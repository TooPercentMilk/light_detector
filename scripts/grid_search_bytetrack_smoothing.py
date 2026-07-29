"""Grid-search ByteTrack and temporal-smoothing settings on COCO val.

This evaluates the full pipeline at fixed detector/classifier weights while
overriding only runtime tracking and smoothing parameters. It is intended for
answering questions like:

    * Does ByteTrack's ``track_thresh`` drop true detector outputs?
    * How much does ``min_consensus`` hurt end-to-end state accuracy?
    * Which runtime settings maximize validation end-to-end accuracy?

Example:

    python scripts/grid_search_bytetrack_smoothing.py \
        --config configs/atlas_medium_state4.yaml \
        --dataset data/coco_atlas_medium_state4 \
        --track-thresh-values 0.1 0.2 0.3 0.5 \
        --window-size-values 1 3 5 \
        --min-consensus-values 1 2 3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from datetime import datetime
from itertools import product
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("runs/eval")

_TAG_TO_CLASS: dict[str, int] = {
    "go": 2,
    "goLeft": 2,
    "goForward": 2,
    "stop": 0,
    "stopLeft": 0,
    "warning": 1,
    "warningLeft": 1,
    "off": 3,
}

_STATE_TO_CLASS = {"red": 0, "yellow": 1, "green": 2, "off": 3}


def _compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _positive_image_ids(coco: dict) -> set[int]:
    return {
        ann["image_id"]
        for ann in coco.get("annotations", [])
        if len(ann.get("bbox", [])) == 4
        and ann["bbox"][2] > 0
        and ann["bbox"][3] > 0
    }


def _infer_detector_num_classes(dataset_path: str | Path) -> int | None:
    ann_path = Path(dataset_path) / "annotations" / "instances_val.json"
    if not ann_path.is_file():
        return None
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)
    categories = coco.get("categories", [])
    return len(categories) if categories else None


def _load_eval_items(
    dataset_path: str | Path,
    positive_images_only: bool,
    max_frames: int | None,
) -> tuple[Path, dict[int, str], dict[int, list[dict]], list[tuple[int, str]]]:
    data_dir = Path(dataset_path).resolve()
    ann_path = data_dir / "annotations" / "instances_val.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Val annotations not found: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    eval_items = sorted(id_to_file.items())
    if positive_images_only:
        positive_ids = _positive_image_ids(coco)
        eval_items = [item for item in eval_items if item[0] in positive_ids]
    if max_frames is not None:
        eval_items = eval_items[:max_frames]

    return data_dir / "val", id_to_file, anns_by_img, eval_items


def _evaluate_combo(
    config_path: str,
    dataset_path: str,
    device: str | None,
    image_size: tuple[int, int],
    positive_images_only: bool,
    max_frames: int | None,
    iou_threshold: float,
    track_thresh: float,
    match_thresh: float,
    track_buffer: int,
    window_size: int,
    min_consensus: int,
) -> dict:
    from adas_perception.traffic_light.config import (
        apply_detector_input_size,
        load_config,
    )
    from adas_perception.traffic_light.node import TrafficLightNode

    cfg = load_config(config_path)
    apply_detector_input_size(cfg, image_size)
    inferred_num_classes = _infer_detector_num_classes(dataset_path)
    if inferred_num_classes is not None:
        cfg.detector.num_classes = inferred_num_classes
    if device:
        cfg.detector.device = device
        cfg.classifier.device = device

    cfg.tracker.track_thresh = track_thresh
    cfg.tracker.match_thresh = match_thresh
    cfg.tracker.track_buffer = track_buffer
    cfg.temporal_smoother.window_size = window_size
    cfg.temporal_smoother.min_consensus = min_consensus

    node = TrafficLightNode(cfg)
    image_dir, _, anns_by_img, eval_items = _load_eval_items(
        dataset_path=dataset_path,
        positive_images_only=positive_images_only,
        max_frames=max_frames,
    )

    total_gt = 0
    total_predictions = 0
    total_detected = 0
    total_correct_state = 0
    skipped_annotations = 0

    for frame_id, (img_id, fname) in enumerate(eval_items):
        image = cv2.imread(str(image_dir / fname))
        if image is None:
            continue

        gt_boxes: list[list[float]] = []
        gt_classes: list[int] = []
        for ann in anns_by_img.get(img_id, []):
            tag = ann.get("attributes", {}).get("lisa_tag")
            if tag is None or tag not in _TAG_TO_CLASS:
                skipped_annotations += 1
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            gt_boxes.append([x, y, x + w, y + h])
            gt_classes.append(_TAG_TO_CLASS[tag])

        total_gt += len(gt_boxes)

        lights = node.process_frame(image, frame_id)
        total_predictions += len(lights)
        if not gt_boxes or not lights:
            continue

        pred_arr = np.array([light.bbox for light in lights], dtype=np.float32)
        gt_arr = np.array(gt_boxes, dtype=np.float32)
        ious = _compute_iou(pred_arr, gt_arr)

        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        order = np.dstack(
            np.unravel_index(np.argsort(-ious, axis=None), ious.shape)
        )[0]
        for p_idx, g_idx in order:
            p_idx, g_idx = int(p_idx), int(g_idx)
            if p_idx in matched_pred or g_idx in matched_gt:
                continue
            if ious[p_idx, g_idx] < iou_threshold:
                break
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)
            total_detected += 1

            pred_cls = _STATE_TO_CLASS.get(lights[p_idx].state.value, -1)
            if pred_cls == gt_classes[g_idx]:
                total_correct_state += 1

    detection_recall = total_detected / total_gt if total_gt else 0.0
    detection_precision = (
        total_detected / total_predictions if total_predictions else 0.0
    )
    state_accuracy = (
        total_correct_state / total_detected if total_detected else 0.0
    )
    end_to_end_accuracy = total_correct_state / total_gt if total_gt else 0.0

    return {
        "track_thresh": track_thresh,
        "match_thresh": match_thresh,
        "track_buffer": track_buffer,
        "window_size": window_size,
        "min_consensus": min_consensus,
        "frames": len(eval_items),
        "total_gt": total_gt,
        "total_predictions": total_predictions,
        "total_detected": total_detected,
        "total_correct_state": total_correct_state,
        "skipped_annotations": skipped_annotations,
        "detection_recall": detection_recall,
        "detection_precision": detection_precision,
        "state_accuracy_on_matched": state_accuracy,
        "end_to_end_accuracy": end_to_end_accuracy,
    }


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search ByteTrack and temporal-smoothing parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/atlas_medium_state4.yaml")
    parser.add_argument("--dataset", default="data/coco_atlas_medium_state4")
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--positive-images-only", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Evaluate only the first N val frames for faster experiments.",
    )
    parser.add_argument(
        "--track-thresh-values",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.5],
    )
    parser.add_argument(
        "--match-thresh-values",
        type=float,
        nargs="+",
        default=[0.8],
    )
    parser.add_argument(
        "--track-buffer-values",
        type=int,
        nargs="+",
        default=[30],
    )
    parser.add_argument(
        "--window-size-values",
        type=int,
        nargs="+",
        default=[1, 3, 5],
    )
    parser.add_argument(
        "--min-consensus-values",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    parser.add_argument(
        "--rank-by",
        choices=[
            "end_to_end_accuracy",
            "detection_recall",
            "detection_precision",
            "state_accuracy_on_matched",
        ],
        default="end_to_end_accuracy",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path; CSV is written next to it.",
    )

    args = parser.parse_args(argv)
    if args.image_size <= 0:
        parser.error("--image-size must be positive")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")

    combos = []
    for combo in product(
        args.track_thresh_values,
        args.match_thresh_values,
        args.track_buffer_values,
        args.window_size_values,
        args.min_consensus_values,
    ):
        track_thresh, match_thresh, track_buffer, window_size, min_consensus = combo
        if min_consensus > window_size:
            continue
        combos.append(combo)

    if not combos:
        parser.error("No valid combinations; min_consensus must be <= window_size")

    logger.info("Evaluating %d parameter combinations", len(combos))
    rows: list[dict] = []
    for index, combo in enumerate(combos, start=1):
        logger.info("Combination %d/%d: %s", index, len(combos), combo)
        row = _evaluate_combo(
            config_path=args.config,
            dataset_path=args.dataset,
            device=args.device,
            image_size=(args.image_size, args.image_size),
            positive_images_only=args.positive_images_only,
            max_frames=args.max_frames,
            iou_threshold=args.iou_threshold,
            track_thresh=combo[0],
            match_thresh=combo[1],
            track_buffer=combo[2],
            window_size=combo[3],
            min_consensus=combo[4],
        )
        rows.append(row)
        logger.info(
            "e2e=%.3f recall=%.3f precision=%.3f state=%.3f",
            row["end_to_end_accuracy"],
            row["detection_recall"],
            row["detection_precision"],
            row["state_accuracy_on_matched"],
        )

    ranked = sorted(rows, key=lambda row: row[args.rank_by], reverse=True)
    best = ranked[0]
    logger.info("")
    logger.info("Best by %s:", args.rank_by)
    logger.info(
        "track_thresh=%.3f match_thresh=%.3f track_buffer=%d "
        "window_size=%d min_consensus=%d | e2e=%.3f recall=%.3f "
        "precision=%.3f state=%.3f",
        best["track_thresh"],
        best["match_thresh"],
        best["track_buffer"],
        best["window_size"],
        best["min_consensus"],
        best["end_to_end_accuracy"],
        best["detection_recall"],
        best["detection_precision"],
        best["state_accuracy_on_matched"],
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = (
        Path(args.output)
        if args.output
        else RESULTS_DIR / f"bytetrack_smoothing_grid_{ts}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": args.config,
        "dataset": args.dataset,
        "device": args.device,
        "rank_by": args.rank_by,
        "search_space": {
            "track_thresh_values": args.track_thresh_values,
            "match_thresh_values": args.match_thresh_values,
            "track_buffer_values": args.track_buffer_values,
            "window_size_values": args.window_size_values,
            "min_consensus_values": args.min_consensus_values,
        },
        "best": best,
        "results": ranked,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _write_csv(ranked, out_path.with_suffix(".csv"))
    logger.info("Wrote JSON -> %s", out_path)
    logger.info("Wrote CSV  -> %s", out_path.with_suffix(".csv"))


if __name__ == "__main__":
    main()
