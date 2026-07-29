"""Evaluate the traffic-light pipeline stratified by ground-truth box size.

For each ground-truth bounding box on the validation set, this script bins
the GT by its sqrt(area) (i.e. roughly its side length in pixels) and
reports:

* **Recall** per GT-size bucket (matched_TP / total_GT in that bucket)
* **End-to-end accuracy** per GT-size bucket
  (matched_TP_with_correct_state / total_GT)
* **State accuracy on matched** per GT-size bucket
  (correct_state / matched_TP)

Predictions are *also* binned by their own predicted-box size so that
**precision** can be reported per predicted-size bucket
(matched_TP / total_predictions in that predicted bucket).

This isolates the "small object" hypothesis: if recall collapses below
some pixel threshold, the detector input resolution is the bottleneck.

Usage::

    python scripts/evaluate_by_size.py \
        --config configs/val_best.yaml \
        --dataset data/coco_tl

    # Write a bar-chart PNG alongside the JSON
    python scripts/evaluate_by_size.py \
        --config configs/val_best.yaml \
        --dataset data/coco_tl \
        --plot

    # Custom bucket edges (sqrt(area) in pixels)
    python scripts/evaluate_by_size.py \
        --config configs/val_best.yaml \
        --dataset data/coco_tl \
        --bin-edges 0 8 12 16 24 32 48 96 1000000
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("runs/eval")

# Mirror the LISA tag → class index mapping used in scripts/evaluate.py.
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

# Default sqrt(area) bin edges in original-image pixels. Designed to spotlight
# the "small object" regime where lower-resolution detector inputs struggle.
DEFAULT_BIN_EDGES: list[float] = [0, 8, 12, 16, 24, 32, 48, 96, 1e9]


def _sequence_from_filename(fname: str) -> str:
    if "--" in fname:
        return fname.split("--", 1)[0]
    return "unknown"


def _positive_image_ids(coco: dict) -> set[int]:
    image_ids: set[int] = set()
    for ann in coco.get("annotations", []):
        bbox = ann.get("bbox", [])
        if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        image_ids.add(ann["image_id"])
    return image_ids


def _compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two ``[N,4]`` and ``[M,4]`` xyxy box arrays."""
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def _bucket_index(side_px: float, edges: list[float]) -> int:
    """Return the bucket index for a side length using ``edges`` as
    half-open intervals ``[edges[i], edges[i+1])``. Anything off the high end
    falls into the last bucket."""
    for i in range(len(edges) - 1):
        if edges[i] <= side_px < edges[i + 1]:
            return i
    return len(edges) - 2


def _bucket_label(edges: list[float], i: int) -> str:
    lo, hi = edges[i], edges[i + 1]
    if hi >= 1e8:
        return f">={int(lo)}"
    return f"{int(lo)}-{int(hi)}"


def evaluate_by_size(
    config_path: str,
    dataset_path: str,
    device: str | None,
    iou_threshold: float,
    bin_edges: list[float],
    positive_images_only: bool,
    top_crop_only: bool = False,
    top_crop_fraction: float | None = None,
    input_size: tuple[int, int] | None = None,
) -> dict:
    """Run the pipeline on the val split and aggregate metrics by box size."""
    from adas_perception.traffic_light.config import (
        apply_detector_input_size,
        load_config,
    )
    from adas_perception.traffic_light.node import TrafficLightNode

    cfg = load_config(config_path)
    if input_size is not None:
        apply_detector_input_size(cfg, input_size)
    if device:
        cfg.detector.device = device
        cfg.classifier.device = device
    if top_crop_fraction is not None:
        cfg.preprocess.top_crop_fraction = top_crop_fraction
    if top_crop_only:
        cfg.preprocess.top_crop_only = True
        cfg.preprocess.top_third_only = False
    logger.info(
        "Top crop: enabled=%s | fraction=%.3f",
        cfg.preprocess.top_crop_only or cfg.preprocess.top_third_only,
        cfg.preprocess.top_crop_fraction,
    )

    node = TrafficLightNode(cfg)

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

    image_dir = data_dir / "val"
    state_to_idx = {"red": 0, "yellow": 1, "green": 2, "off": 3}

    n_buckets = len(bin_edges) - 1

    # GT-side accumulators, keyed by sequence then bucket index.
    seq_gt = defaultdict(lambda: np.zeros(n_buckets, dtype=np.int64))
    seq_matched = defaultdict(lambda: np.zeros(n_buckets, dtype=np.int64))
    seq_correct_state = defaultdict(lambda: np.zeros(n_buckets, dtype=np.int64))

    # Prediction-side accumulators (precision is by *predicted* box size).
    seq_preds = defaultdict(lambda: np.zeros(n_buckets, dtype=np.int64))
    seq_pred_tp = defaultdict(lambda: np.zeros(n_buckets, dtype=np.int64))

    eval_items = sorted(id_to_file.items())
    if positive_images_only:
        positive_ids = _positive_image_ids(coco)
        eval_items = [item for item in eval_items if item[0] in positive_ids]
        logger.info(
            "Restricted to positive images only: %d images",
            len(eval_items),
        )

    logger.info(
        "Bins (sqrt(area), pixels): %s",
        [_bucket_label(bin_edges, i) for i in range(n_buckets)],
    )

    for frame_id, (img_id, fname) in enumerate(eval_items):
        img_path = image_dir / fname
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        seq = _sequence_from_filename(fname)
        gt_anns = anns_by_img.get(img_id, [])

        gt_boxes: list[list[float]] = []
        gt_classes: list[int] = []
        gt_buckets: list[int] = []
        for ann in gt_anns:
            tag = ann.get("attributes", {}).get("lisa_tag")
            if tag is None or tag not in _TAG_TO_CLASS:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            gt_boxes.append([x, y, x + w, y + h])
            gt_classes.append(_TAG_TO_CLASS[tag])
            gt_buckets.append(_bucket_index(math.sqrt(w * h), bin_edges))

        for b in gt_buckets:
            seq_gt[seq][b] += 1

        # Always run the pipeline, even on images with no eligible GT, so
        # that false positives still count toward precision.
        lights = node.process_frame(image, frame_id)

        # Per-prediction bucketing.
        pred_buckets = []
        for light in lights:
            x1, y1, x2, y2 = light.bbox
            side = math.sqrt(max(0.0, (x2 - x1) * (y2 - y1)))
            b = _bucket_index(side, bin_edges)
            pred_buckets.append(b)
            seq_preds[seq][b] += 1

        if not gt_boxes or not lights:
            continue

        gt_arr = np.array(gt_boxes, dtype=np.float32)
        pred_arr = np.array([light.bbox for light in lights], dtype=np.float32)
        ious = _compute_iou(pred_arr, gt_arr)  # (n_pred, n_gt)

        # Greedy match: highest IoU first, each GT and prediction at most once.
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        # Flatten and sort by descending IoU.
        order = np.dstack(np.unravel_index(np.argsort(-ious, axis=None), ious.shape))[0]
        for p_idx, g_idx in order:
            p_idx, g_idx = int(p_idx), int(g_idx)
            if p_idx in matched_pred or g_idx in matched_gt:
                continue
            if ious[p_idx, g_idx] < iou_threshold:
                break
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)

            b_gt = gt_buckets[g_idx]
            seq_matched[seq][b_gt] += 1
            seq_pred_tp[seq][pred_buckets[p_idx]] += 1

            pred_state = lights[p_idx].state.value
            pred_cls = state_to_idx.get(pred_state, -1)
            if pred_cls == gt_classes[g_idx]:
                seq_correct_state[seq][b_gt] += 1

    # ---- aggregate totals across sequences ----
    sequences = sorted(seq_gt.keys() | seq_preds.keys())
    total_gt = np.zeros(n_buckets, dtype=np.int64)
    total_matched = np.zeros(n_buckets, dtype=np.int64)
    total_correct = np.zeros(n_buckets, dtype=np.int64)
    total_preds = np.zeros(n_buckets, dtype=np.int64)
    total_pred_tp = np.zeros(n_buckets, dtype=np.int64)
    for s in sequences:
        total_gt += seq_gt[s]
        total_matched += seq_matched[s]
        total_correct += seq_correct_state[s]
        total_preds += seq_preds[s]
        total_pred_tp += seq_pred_tp[s]

    def _pack(
        gt: np.ndarray,
        matched: np.ndarray,
        correct: np.ndarray,
        preds: np.ndarray,
        pred_tp: np.ndarray,
    ) -> dict:
        out: list[dict] = []
        for i in range(n_buckets):
            out.append(
                {
                    "bucket": _bucket_label(bin_edges, i),
                    "side_min_px": float(bin_edges[i]),
                    "side_max_px": float(bin_edges[i + 1]),
                    "gt_count": int(gt[i]),
                    "matched_count": int(matched[i]),
                    "correct_state_count": int(correct[i]),
                    "predicted_count": int(preds[i]),
                    "predicted_tp_count": int(pred_tp[i]),
                    "recall": (matched[i] / gt[i]) if gt[i] else 0.0,
                    "precision_by_pred_size": (pred_tp[i] / preds[i]) if preds[i] else 0.0,
                    "state_accuracy_on_matched": (correct[i] / matched[i]) if matched[i] else 0.0,
                    "end_to_end_accuracy": (correct[i] / gt[i]) if gt[i] else 0.0,
                }
            )
        return {"buckets": out}

    results = {
        "config": str(config_path),
        "input_size": list(input_size) if input_size is not None else list(cfg.detector.input_size),
        "iou_threshold": iou_threshold,
        "positive_images_only": positive_images_only,
        "top_crop_only": bool(cfg.preprocess.top_crop_only or cfg.preprocess.top_third_only),
        "top_crop_fraction": float(cfg.preprocess.top_crop_fraction),
        "bin_edges": [float(e) for e in bin_edges],
        "total": _pack(total_gt, total_matched, total_correct, total_preds, total_pred_tp),
    }
    for s in sequences:
        results[s] = _pack(
            seq_gt[s],
            seq_matched[s],
            seq_correct_state[s],
            seq_preds[s],
            seq_pred_tp[s],
        )

    return results


def _print_table(label: str, packed: dict) -> None:
    logger.info("")
    logger.info("=" * 96)
    logger.info("%s", label)
    logger.info("=" * 96)
    logger.info(
        "%-10s %8s %8s %8s %8s %8s %8s %8s",
        "bucket", "gt", "matched", "preds", "recall", "prec", "state%", "e2e%",
    )
    logger.info("-" * 96)
    for b in packed["buckets"]:
        logger.info(
            "%-10s %8d %8d %8d %8.3f %8.3f %8.3f %8.3f",
            b["bucket"],
            b["gt_count"],
            b["matched_count"],
            b["predicted_count"],
            b["recall"],
            b["precision_by_pred_size"],
            b["state_accuracy_on_matched"],
            b["end_to_end_accuracy"],
        )


def _maybe_plot(results: dict, out_png: Path) -> None:
    """Render a 2x2 grid: recall, precision, state acc, end-to-end vs bucket."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return

    buckets = results["total"]["buckets"]
    labels = [b["bucket"] for b in buckets]
    x = np.arange(len(labels))

    recall = [b["recall"] for b in buckets]
    precision = [b["precision_by_pred_size"] for b in buckets]
    state_acc = [b["state_accuracy_on_matched"] for b in buckets]
    e2e = [b["end_to_end_accuracy"] for b in buckets]
    gt_counts = [b["gt_count"] for b in buckets]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = [
        ("Recall (by GT size)", recall),
        ("Precision (by predicted size)", precision),
        ("State accuracy on matched", state_acc),
        ("End-to-end accuracy (by GT size)", e2e),
    ]
    for ax, (title, vals) in zip(axes.flat, titles):
        ax.bar(x, vals, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1)
        ax.set_xlabel("sqrt(area) bucket [px]")
        ax.set_ylabel("ratio")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        # Annotate bars with GT support so empty bins aren't misleading.
        for xi, v, n in zip(x, vals, gt_counts):
            ax.text(xi, v + 0.02, f"n={n}", ha="center", fontsize=8)

    fig.suptitle("Detector / pipeline metrics vs. ground-truth box size")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    logger.info("Wrote plot -> %s", out_png)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the pipeline stratified by GT bounding-box size.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/val_best.yaml")
    parser.add_argument("--dataset", default="data/coco_tl")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--image-size",
        type=int,
        default=960,
        help="Square detector/preprocessor image size",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Detector/preprocessor input size; overrides --image-size",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--bin-edges",
        type=float,
        nargs="+",
        default=DEFAULT_BIN_EDGES,
        help="Sqrt(area) bucket edges in pixels (one fewer bucket than edges).",
    )
    parser.add_argument(
        "--positive-images-only",
        action="store_true",
        help="Restrict to images with at least one valid annotation.",
    )
    parser.add_argument(
        "--top-half-only",
        "--top-40-only",
        "--top-third-only",
        dest="top_crop_only",
        action="store_true",
        help="Run detector inference on only the configured top crop of each frame.",
    )
    parser.add_argument(
        "--top-crop-fraction",
        type=float,
        default=None,
        help="Image-height fraction kept when top-crop inference is enabled.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Also write a PNG bar chart next to the JSON output.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: runs/eval/eval_by_size_<ts>.json).",
    )

    args = parser.parse_args(argv)

    if len(args.bin_edges) < 2:
        parser.error("--bin-edges must have at least 2 values")
    if any(args.bin_edges[i] >= args.bin_edges[i + 1] for i in range(len(args.bin_edges) - 1)):
        parser.error("--bin-edges must be strictly increasing")
    if args.top_crop_fraction is not None and not 0.0 < args.top_crop_fraction <= 1.0:
        parser.error("--top-crop-fraction must be in the range (0, 1]")

    from adas_perception.traffic_light.config import detector_input_size_from_args

    try:
        input_size = detector_input_size_from_args(args.image_size, args.input_size)
    except ValueError as exc:
        parser.error(str(exc))

    results = evaluate_by_size(
        config_path=args.config,
        dataset_path=args.dataset,
        device=args.device,
        iou_threshold=args.iou_threshold,
        bin_edges=list(args.bin_edges),
        positive_images_only=args.positive_images_only,
        top_crop_only=args.top_crop_only,
        top_crop_fraction=args.top_crop_fraction,
        input_size=input_size,
    )

    _print_table("TOTAL (all sequences)", results["total"])
    for key in sorted(
        k
        for k in results.keys()
        if k
        not in {
            "total",
            "config",
            "input_size",
            "iou_threshold",
            "positive_images_only",
            "top_crop_only",
            "top_crop_fraction",
            "bin_edges",
        }
    ):
        _print_table(key, results[key])

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = Path(args.output) if args.output else RESULTS_DIR / f"eval_by_size_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote results -> %s", out_path)

    if args.plot:
        _maybe_plot(results, out_path.with_suffix(".png"))


if __name__ == "__main__":
    main()
