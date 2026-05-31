"""Log detector false negatives from a best-model validation run.

The script evaluates the detector from ``configs/val_best.yaml`` on a
COCO-format validation split, matches detections to ground truth, and writes a
review package for every unmatched GT box.

Outputs are written to ``runs/eval/false_negatives_<timestamp>/``:

* ``false_negatives.csv`` with absolute paths and file:// links
* ``false_negatives.json`` with the same per-object records
* ``summary.txt`` with aggregate counts
* ``index.html`` for quick manual browsing
* optional copied originals, annotated overlays, and GT crops

Example:

    python scripts/log_false_negatives.py \
        --config configs/val_best.yaml \
        --dataset data/coco_tl \
        --confidence-threshold 0.1 \
        --iou-threshold 0.5

Use ``--no-save-images`` if you only want CSV/JSON links to the original files.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("runs/eval")
DEFAULT_SIZE_BIN_EDGES = [0.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 96.0, 1e9]

TAG_TO_COLOR: dict[str, str] = {
    "go": "green",
    "goLeft": "green",
    "goForward": "green",
    "stop": "red",
    "stopLeft": "red",
    "warning": "yellow",
    "warningLeft": "yellow",
}


@dataclass(frozen=True)
class GTItem:
    global_idx: int
    ann_id: int
    image_id: int
    file_name: str
    sequence: str
    category_id: int
    lisa_tag: str
    color: str
    bbox_xyxy: tuple[float, float, float, float]
    bbox_xywh: tuple[float, float, float, float]
    image_width: int
    image_height: int


@dataclass(frozen=True)
class DetectionItem:
    global_idx: int
    image_id: int
    category_id: int
    bbox_xyxy: tuple[float, float, float, float]
    bbox_xywh: tuple[float, float, float, float]
    score: float
    objectness: float
    class_confidence: float


@dataclass(frozen=True)
class MatchItem:
    gt_idx: int
    det_idx: int
    iou: float


def _sequence_from_filename(fname: str) -> str:
    if "--" in fname:
        return fname.split("--", 1)[0]
    return "unknown"


def _xywh_to_xyxy(box: tuple[float, float, float, float] | list[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (float(x), float(y), float(x + w), float(y + h))


def _box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou_one_to_many(
    box: tuple[float, float, float, float],
    boxes: list[tuple[float, float, float, float]],
) -> np.ndarray:
    if not boxes:
        return np.zeros((0,), dtype=np.float32)

    box_arr = np.array(box, dtype=np.float32)
    boxes_arr = np.array(boxes, dtype=np.float32)

    x1 = np.maximum(box_arr[0], boxes_arr[:, 0])
    y1 = np.maximum(box_arr[1], boxes_arr[:, 1])
    x2 = np.minimum(box_arr[2], boxes_arr[:, 2])
    y2 = np.minimum(box_arr[3], boxes_arr[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = _box_area(box)
    boxes_area = np.maximum(0.0, boxes_arr[:, 2] - boxes_arr[:, 0]) * np.maximum(
        0.0, boxes_arr[:, 3] - boxes_arr[:, 1]
    )
    union = box_area + boxes_area - inter
    return inter / np.maximum(union, 1e-6)


def _bucket_label(edges: list[float], idx: int) -> str:
    lo, hi = edges[idx], edges[idx + 1]
    if hi >= 1e8:
        return f">={int(lo)}"
    return f"{int(lo)}-{int(hi)}"


def _bucket_value(value: float, edges: list[float]) -> str:
    for idx in range(len(edges) - 1):
        if edges[idx] <= value < edges[idx + 1]:
            return _bucket_label(edges, idx)
    return _bucket_label(edges, len(edges) - 2)


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _safe_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    return "".join(ch if ch in allowed else "_" for ch in value)


def _path_uri(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return path.resolve().as_uri()
    except ValueError:
        return str(path)


def _load_detector(
    config_path: str,
    device_override: str | None,
    input_size_override: tuple[int, int] | None = None,
):
    from adas_perception.traffic_light.config import (
        apply_detector_input_size,
        load_config,
    )
    from yolox.exp import get_exp

    cfg = load_config(config_path)
    if input_size_override is not None:
        apply_detector_input_size(cfg, input_size_override)
    device_name = device_override or cfg.detector.device
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Config requested CUDA but CUDA is not available; using CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    exp = get_exp(None, cfg.detector.exp_name)
    exp.num_classes = cfg.detector.num_classes
    exp.input_size = tuple(cfg.detector.input_size)
    exp.test_size = tuple(cfg.detector.input_size)
    model = exp.get_model()

    weights_path = Path(cfg.detector.model_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Detector weights not found: {weights_path}")

    try:
        ckpt = torch.load(str(weights_path), map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(str(weights_path), map_location=device)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info("Loaded detector weights: %s", weights_path)
    logger.info("Input size: %s | device: %s", tuple(cfg.detector.input_size), device)
    return cfg, model, device


def _collect_detections(
    model: torch.nn.Module,
    dataset_path: str,
    input_size: tuple[int, int],
    candidate_confidence_min: float,
    nms_threshold: float,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    positive_images_only: bool,
    max_images: int | None,
    sequences: list[str] | None,
    top_crop_only: bool,
    top_crop_fraction: float,
) -> tuple[list[DetectionItem], Any, list[int]]:
    from adas_perception.traffic_light.detector.evaluator import _COCOValDataset, _collate_fn
    from yolox.utils import postprocess

    val_dataset = _COCOValDataset(
        data_dir=str(Path(dataset_path).resolve()),
        json_file="instances_val.json",
        name="val",
        img_size=input_size,
        positive_images_only=positive_images_only,
        top_crop_only=top_crop_only,
        top_crop_fraction=top_crop_fraction,
    )

    if sequences:
        sequence_set = set(sequences)
        val_dataset.img_ids = [
            img_id
            for img_id in val_dataset.img_ids
            if _sequence_from_filename(val_dataset.coco.loadImgs(img_id)[0]["file_name"]) in sequence_set
        ]

    if max_images is not None:
        val_dataset.img_ids = val_dataset.img_ids[:max_images]

    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty after filtering")

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type != "cpu"),
        collate_fn=_collate_fn,
    )

    detections: list[DetectionItem] = []
    num_classes = len(val_dataset.class_ids)

    logger.info(
        "Running inference at candidate confidence %.4f over %d images",
        candidate_confidence_min,
        len(val_dataset),
    )

    for batch_idx, (imgs, _, info_imgs, ids) in enumerate(val_loader, start=1):
        with torch.no_grad():
            outputs = model(imgs.to(device))

        outputs = postprocess(outputs, num_classes, candidate_confidence_min, nms_threshold)

        for output, img_h, img_w, img_id in zip(outputs, info_imgs[0], info_imgs[1], ids):
            if output is None:
                continue

            output = output.cpu()
            bboxes = output[:, 0:4]
            scale = min(input_size[0] / float(img_h), input_size[1] / float(img_w))
            bboxes /= scale
            bboxes[:, 0::2].clamp_(0, float(img_w))
            bboxes[:, 1::2].clamp_(0, float(img_h))

            bboxes_xywh = bboxes.clone()
            bboxes_xywh[:, 2] -= bboxes_xywh[:, 0]
            bboxes_xywh[:, 3] -= bboxes_xywh[:, 1]

            objectness = output[:, 4].numpy()
            class_conf = output[:, 5].numpy()
            scores = objectness * class_conf
            cls = output[:, 6].numpy().astype(int)

            for idx in range(len(bboxes_xywh)):
                if bboxes_xywh[idx, 2] <= 0 or bboxes_xywh[idx, 3] <= 0:
                    continue
                cat_idx = int(cls[idx])
                if cat_idx >= len(val_dataset.class_ids):
                    continue
                bbox_xywh = tuple(float(v) for v in bboxes_xywh[idx].numpy().tolist())
                detections.append(
                    DetectionItem(
                        global_idx=len(detections),
                        image_id=int(img_id),
                        category_id=int(val_dataset.class_ids[cat_idx]),
                        bbox_xyxy=_xywh_to_xyxy(bbox_xywh),
                        bbox_xywh=bbox_xywh,
                        score=float(scores[idx]),
                        objectness=float(objectness[idx]),
                        class_confidence=float(class_conf[idx]),
                    )
                )

        if batch_idx % 25 == 0:
            logger.info("Processed %d batches; cached %d detections", batch_idx, len(detections))

    logger.info("Cached %d detections at candidate confidence", len(detections))
    return detections, val_dataset.coco, list(val_dataset.img_ids)


def _load_ground_truth(coco_gt: Any, image_ids: list[int]) -> tuple[dict[int, list[GTItem]], list[GTItem]]:
    image_id_set = set(int(v) for v in image_ids)
    image_by_id = {int(img["id"]): img for img in coco_gt.dataset.get("images", [])}
    valid_cat_ids = set(int(v) for v in coco_gt.getCatIds())

    gt_by_img: dict[int, list[GTItem]] = defaultdict(list)
    all_gt: list[GTItem] = []

    for ann in coco_gt.dataset.get("annotations", []):
        image_id = int(ann.get("image_id"))
        if image_id not in image_id_set:
            continue
        if int(ann.get("category_id")) not in valid_cat_ids:
            continue

        bbox = ann.get("bbox", [])
        if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue

        image_info = image_by_id.get(image_id)
        if image_info is None:
            continue

        lisa_tag = str(ann.get("attributes", {}).get("lisa_tag", "unknown"))
        bbox_xywh = tuple(float(v) for v in bbox)
        item = GTItem(
            global_idx=len(all_gt),
            ann_id=int(ann.get("id", len(all_gt))),
            image_id=image_id,
            file_name=str(image_info["file_name"]),
            sequence=_sequence_from_filename(str(image_info["file_name"])),
            category_id=int(ann["category_id"]),
            lisa_tag=lisa_tag,
            color=TAG_TO_COLOR.get(lisa_tag, "unknown"),
            bbox_xyxy=_xywh_to_xyxy(bbox_xywh),
            bbox_xywh=bbox_xywh,
            image_width=int(image_info["width"]),
            image_height=int(image_info["height"]),
        )
        gt_by_img[image_id].append(item)
        all_gt.append(item)

    return gt_by_img, all_gt


def _match_detections(
    detections: list[DetectionItem],
    gt_by_img: dict[int, list[GTItem]],
    iou_threshold: float,
) -> tuple[list[MatchItem], set[int], set[int]]:
    candidates: list[tuple[float, float, int, int]] = []

    detections_by_img: dict[int, list[DetectionItem]] = defaultdict(list)
    for det in detections:
        detections_by_img[det.image_id].append(det)

    for image_id, gt_items in gt_by_img.items():
        image_dets = detections_by_img.get(image_id, [])
        if not image_dets:
            continue

        for gt in gt_items:
            same_cat_dets = [det for det in image_dets if det.category_id == gt.category_id]
            if not same_cat_dets:
                continue
            ious = _iou_one_to_many(gt.bbox_xyxy, [det.bbox_xyxy for det in same_cat_dets])
            for det, iou in zip(same_cat_dets, ious):
                iou_float = float(iou)
                if iou_float >= iou_threshold:
                    candidates.append((iou_float, det.score, gt.global_idx, det.global_idx))

    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    used_gt: set[int] = set()
    used_det: set[int] = set()
    matches: list[MatchItem] = []

    for iou, _, gt_idx, det_idx in candidates:
        if gt_idx in used_gt or det_idx in used_det:
            continue
        used_gt.add(gt_idx)
        used_det.add(det_idx)
        matches.append(MatchItem(gt_idx=gt_idx, det_idx=det_idx, iou=iou))

    return matches, used_gt, used_det


def _best_detection_for_gt(gt: GTItem, detections: list[DetectionItem]) -> tuple[DetectionItem | None, float | None]:
    same_image_cat = [
        det
        for det in detections
        if det.image_id == gt.image_id and det.category_id == gt.category_id
    ]
    if not same_image_cat:
        return None, None
    ious = _iou_one_to_many(gt.bbox_xyxy, [det.bbox_xyxy for det in same_image_cat])
    best_idx = int(ious.argmax())
    return same_image_cat[best_idx], float(ious[best_idx])


def _miss_reason(
    gt: GTItem,
    all_detections: list[DetectionItem],
    final_detections: list[DetectionItem],
    matched_det_ids: set[int],
    confidence_threshold: float,
    iou_threshold: float,
) -> str:
    best_final, best_final_iou = _best_detection_for_gt(gt, final_detections)
    if best_final is not None and best_final_iou is not None:
        if best_final_iou >= iou_threshold:
            if best_final.global_idx in matched_det_ids:
                return "duplicate_or_shared_detection"
            return "unmatched_after_greedy_matching"
        return "low_iou"

    best_any, best_any_iou = _best_detection_for_gt(gt, all_detections)
    if best_any is None or best_any_iou is None:
        return "no_candidate_detection"
    if best_any.score < confidence_threshold:
        return "below_confidence"
    return "no_candidate_detection"


def _crop_with_padding(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    pad_fraction: float,
) -> np.ndarray | None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad_x = max(2, int(round(bw * pad_fraction)))
    pad_y = max(2, int(round(bh * pad_fraction)))

    ix1 = max(0, int(math.floor(x1)) - pad_x)
    iy1 = max(0, int(math.floor(y1)) - pad_y)
    ix2 = min(w, int(math.ceil(x2)) + pad_x)
    iy2 = min(h, int(math.ceil(y2)) + pad_y)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return image[iy1:iy2, ix1:ix2]


def _clip_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (
        max(0, min(width - 1, int(round(x1)))),
        max(0, min(height - 1, int(round(y1)))),
        max(0, min(width - 1, int(round(x2)))),
        max(0, min(height - 1, int(round(y2)))),
    )


def _draw_box(
    image: np.ndarray,
    box: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = _clip_box(box, w, h)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    text_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    top = max(0, y1 - text_h - baseline - 4)
    cv2.rectangle(image, (x1, top), (min(w - 1, x1 + text_w + 4), top + text_h + baseline + 4), color, -1)
    cv2.putText(
        image,
        label,
        (x1 + 2, top + text_h + 1),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def _write_review_images(
    gt: GTItem,
    row: dict[str, Any],
    image: np.ndarray,
    image_path: Path,
    final_detections_by_img: dict[int, list[DetectionItem]],
    best_candidate: DetectionItem | None,
    output_dir: Path,
) -> tuple[Path | None, Path | None, Path | None]:
    originals_dir = output_dir / "originals"
    overlays_dir = output_dir / "overlays"
    crops_dir = output_dir / "crops"
    originals_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    base_name = _safe_name(f"{gt.sequence}__img_{gt.image_id}__ann_{gt.ann_id}__{Path(gt.file_name).stem}")
    original_path = originals_dir / _safe_name(f"img_{gt.image_id}__{Path(gt.file_name).name}")
    if not original_path.exists():
        shutil.copy2(image_path, original_path)

    overlay = image.copy()
    for det in final_detections_by_img.get(gt.image_id, []):
        _draw_box(overlay, det.bbox_xyxy, f"det {det.score:.2f}", (255, 170, 0), 1)
    if best_candidate is not None and best_candidate.score < row["confidence_threshold"]:
        _draw_box(
            overlay,
            best_candidate.bbox_xyxy,
            f"best low {best_candidate.score:.2f}",
            (0, 220, 255),
            1,
        )
    _draw_box(overlay, gt.bbox_xyxy, f"FN ann {gt.ann_id} {gt.lisa_tag}", (0, 0, 255), 3)

    overlay_path = overlays_dir / f"{base_name}.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    crop = _crop_with_padding(image, gt.bbox_xyxy, pad_fraction=1.0)
    crop_path = None
    if crop is not None:
        crop_path = crops_dir / f"{base_name}.jpg"
        cv2.imwrite(str(crop_path), crop)

    return original_path, overlay_path, crop_path


def _build_false_negative_rows(
    false_negatives: list[GTItem],
    all_detections: list[DetectionItem],
    final_detections: list[DetectionItem],
    matched_det_ids: set[int],
    dataset_path: str,
    output_dir: Path,
    save_images: bool,
    max_save_images: int | None,
    confidence_threshold: float,
    iou_threshold: float,
    size_bin_edges: list[float],
) -> list[dict[str, Any]]:
    image_dir = Path(dataset_path).resolve() / "val"
    final_detections_by_img: dict[int, list[DetectionItem]] = defaultdict(list)
    for det in final_detections:
        final_detections_by_img[det.image_id].append(det)

    image_cache: dict[int, np.ndarray | None] = {}
    rows: list[dict[str, Any]] = []

    for idx, gt in enumerate(false_negatives):
        image_path = image_dir / gt.file_name
        best_any, best_any_iou = _best_detection_for_gt(gt, all_detections)
        best_final, best_final_iou = _best_detection_for_gt(gt, final_detections)

        gt_w = max(0.0, gt.bbox_xyxy[2] - gt.bbox_xyxy[0])
        gt_h = max(0.0, gt.bbox_xyxy[3] - gt.bbox_xyxy[1])
        gt_area = gt_w * gt_h
        gt_sqrt_area = math.sqrt(gt_area)

        row: dict[str, Any] = {
            "rank": idx + 1,
            "image_id": gt.image_id,
            "ann_id": gt.ann_id,
            "file_name": gt.file_name,
            "image_path": str(image_path),
            "image_uri": _path_uri(image_path),
            "sequence": gt.sequence,
            "category_id": gt.category_id,
            "lisa_tag": gt.lisa_tag,
            "color": gt.color,
            "gt_bbox_xywh": [float(v) for v in gt.bbox_xywh],
            "gt_width": float(gt_w),
            "gt_height": float(gt_h),
            "gt_area": float(gt_area),
            "gt_sqrt_area": float(gt_sqrt_area),
            "size_bucket": _bucket_value(gt_sqrt_area, size_bin_edges),
            "center_x_norm": float(((gt.bbox_xyxy[0] + gt.bbox_xyxy[2]) * 0.5) / gt.image_width),
            "center_y_norm": float(((gt.bbox_xyxy[1] + gt.bbox_xyxy[3]) * 0.5) / gt.image_height),
            "confidence_threshold": float(confidence_threshold),
            "iou_threshold": float(iou_threshold),
            "miss_reason": _miss_reason(
                gt,
                all_detections,
                final_detections,
                matched_det_ids,
                confidence_threshold,
                iou_threshold,
            ),
            "best_candidate_iou": best_any_iou,
            "best_candidate_score": best_any.score if best_any is not None else None,
            "best_candidate_objectness": best_any.objectness if best_any is not None else None,
            "best_candidate_class_confidence": best_any.class_confidence if best_any is not None else None,
            "best_candidate_bbox_xywh": list(best_any.bbox_xywh) if best_any is not None else None,
            "best_final_iou": best_final_iou,
            "best_final_score": best_final.score if best_final is not None else None,
            "best_final_bbox_xywh": list(best_final.bbox_xywh) if best_final is not None else None,
            "original_copy_path": "",
            "overlay_path": "",
            "crop_path": "",
            "original_copy_uri": "",
            "overlay_uri": "",
            "crop_uri": "",
        }

        should_save = save_images and (max_save_images is None or idx < max_save_images)
        if should_save:
            if gt.image_id not in image_cache:
                image_cache[gt.image_id] = cv2.imread(str(image_path))
            image = image_cache[gt.image_id]
            if image is None:
                logger.warning("Could not read image for review artifact: %s", image_path)
            else:
                original_path, overlay_path, crop_path = _write_review_images(
                    gt=gt,
                    row=row,
                    image=image,
                    image_path=image_path,
                    final_detections_by_img=final_detections_by_img,
                    best_candidate=best_any,
                    output_dir=output_dir,
                )
                row["original_copy_path"] = str(original_path) if original_path else ""
                row["overlay_path"] = str(overlay_path) if overlay_path else ""
                row["crop_path"] = str(crop_path) if crop_path else ""
                row["original_copy_uri"] = _path_uri(original_path)
                row["overlay_uri"] = _path_uri(overlay_path)
                row["crop_uri"] = _path_uri(crop_path)

        rows.append(row)

    return rows


def _write_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "image_id",
        "ann_id",
        "file_name",
        "image_path",
        "image_uri",
        "sequence",
        "category_id",
        "lisa_tag",
        "color",
        "gt_bbox_xywh",
        "gt_width",
        "gt_height",
        "gt_area",
        "gt_sqrt_area",
        "size_bucket",
        "center_x_norm",
        "center_y_norm",
        "confidence_threshold",
        "iou_threshold",
        "miss_reason",
        "best_candidate_iou",
        "best_candidate_score",
        "best_candidate_objectness",
        "best_candidate_class_confidence",
        "best_candidate_bbox_xywh",
        "best_final_iou",
        "best_final_score",
        "best_final_bbox_xywh",
        "original_copy_path",
        "overlay_path",
        "crop_path",
        "original_copy_uri",
        "overlay_uri",
        "crop_uri",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fieldnames}
            out["gt_bbox_xywh"] = json.dumps(row.get("gt_bbox_xywh", []))
            out["best_candidate_bbox_xywh"] = json.dumps(row.get("best_candidate_bbox_xywh"))
            out["best_final_bbox_xywh"] = json.dumps(row.get("best_final_bbox_xywh"))
            writer.writerow(out)


def _counter_table(counter: Counter[str], total: int) -> list[str]:
    lines = []
    for name, count in counter.most_common():
        lines.append(f"{name:<28} {count:>7}  ({_safe_rate(count, total) * 100:5.1f}%)")
    return lines


def _write_summary(
    summary_path: Path,
    rows: list[dict[str, Any]],
    parameters: dict[str, Any],
    counts: dict[str, int | float],
) -> None:
    total_fn = len(rows)
    reason_counter = Counter(str(row["miss_reason"]) for row in rows)
    sequence_counter = Counter(str(row["sequence"]) for row in rows)
    color_counter = Counter(str(row["color"]) for row in rows)
    size_counter = Counter(str(row["size_bucket"]) for row in rows)

    lines = [
        "FALSE NEGATIVE DETECTION REVIEW",
        "=" * 38,
        f"Config: {parameters['config']}",
        f"Dataset: {parameters['dataset']}",
        f"Detector weights: {parameters['detector_weights']}",
        f"Images evaluated: {counts['num_images']}",
        f"GT objects: {counts['gt_count']}",
        f"Detections at candidate confidence: {counts['candidate_detection_count']}",
        f"Detections at eval confidence: {counts['final_detection_count']}",
        f"Matched GT: {counts['matched_gt_count']}",
        f"False negatives: {total_fn}",
        f"Recall at IoU >= {parameters['iou_threshold']}: {counts['recall'] * 100:.2f}%",
        "",
        "Parameters",
        f"Confidence threshold: {parameters['confidence_threshold']}",
        f"Candidate confidence min: {parameters['candidate_confidence_min']}",
        f"NMS threshold: {parameters['nms_threshold']}",
        f"Positive images only: {parameters['positive_images_only']}",
        f"Top crop only: {parameters['top_crop_only']}",
        f"Top crop fraction: {parameters['top_crop_fraction']}",
        f"Sequences: {parameters['sequences']}",
        "",
        "Miss reasons",
        *(_counter_table(reason_counter, total_fn) if total_fn else ["none"]),
        "",
        "By color",
        *(_counter_table(color_counter, total_fn) if total_fn else ["none"]),
        "",
        "By size bucket, sqrt(area) px",
        *(_counter_table(size_counter, total_fn) if total_fn else ["none"]),
        "",
        "By sequence",
        *(_counter_table(sequence_counter, total_fn) if total_fn else ["none"]),
    ]

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def _write_json(payload: dict[str, Any], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _rel_or_uri(path_value: str, base_dir: Path, fallback_uri: str) -> str:
    if not path_value:
        return fallback_uri
    path = Path(path_value)
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return _path_uri(path)


def _write_html(rows: list[dict[str, Any]], html_path: Path) -> None:
    cards: list[str] = []
    base_dir = html_path.parent
    for row in rows:
        visual_src = _rel_or_uri(str(row.get("overlay_path", "")), base_dir, str(row.get("image_uri", "")))
        crop_src = _rel_or_uri(str(row.get("crop_path", "")), base_dir, "")
        original_href = row.get("original_copy_uri") or row.get("image_uri") or ""

        visual_img = (
            f'<a href="{html.escape(str(original_href))}"><img src="{html.escape(visual_src)}" alt="overlay"></a>'
            if visual_src
            else ""
        )
        crop_img = f'<img src="{html.escape(crop_src)}" alt="crop">' if crop_src else ""
        candidate = row.get("best_candidate_score")
        candidate_text = "none" if candidate is None else f"{float(candidate):.3f}"
        iou = row.get("best_candidate_iou")
        iou_text = "none" if iou is None else f"{float(iou):.3f}"
        bbox = html.escape(json.dumps(row.get("gt_bbox_xywh")))

        cards.append(
            "\n".join(
                [
                    '<section class="card">',
                    '<div class="media">',
                    visual_img,
                    crop_img,
                    "</div>",
                    "<dl>",
                    f"<dt>Rank</dt><dd>{row['rank']}</dd>",
                    f"<dt>Image</dt><dd><a href=\"{html.escape(str(row.get('image_uri', '')))}\">{html.escape(str(row['file_name']))}</a></dd>",
                    f"<dt>Annotation</dt><dd>{row['ann_id']} / image {row['image_id']}</dd>",
                    f"<dt>Sequence</dt><dd>{html.escape(str(row['sequence']))}</dd>",
                    f"<dt>Tag</dt><dd>{html.escape(str(row['lisa_tag']))} ({html.escape(str(row['color']))})</dd>",
                    f"<dt>Reason</dt><dd>{html.escape(str(row['miss_reason']))}</dd>",
                    f"<dt>GT bbox</dt><dd><code>{bbox}</code></dd>",
                    f"<dt>Size</dt><dd>{float(row['gt_sqrt_area']):.1f} sqrt-area px ({html.escape(str(row['size_bucket']))})</dd>",
                    f"<dt>Best candidate</dt><dd>score {candidate_text}, IoU {iou_text}</dd>",
                    "</dl>",
                    "</section>",
                ]
            )
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>False Negative Review</title>
  <style>
    body {{
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #16202a;
      background: #f5f6f8;
    }}
    header {{
      position: sticky;
      top: 0;
      padding: 16px 24px;
      background: #ffffff;
      border-bottom: 1px solid #d9dee7;
      z-index: 1;
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 700;
    }}
    main {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
      gap: 16px;
      padding: 16px;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d9dee7;
      border-radius: 8px;
      overflow: hidden;
    }}
    .media {{
      display: grid;
      grid-template-columns: 1fr minmax(120px, 0.35fr);
      gap: 8px;
      padding: 8px;
      background: #111820;
      align-items: start;
    }}
    img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 4px;
    }}
    dl {{
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 6px 12px;
      margin: 0;
      padding: 12px;
      font-size: 13px;
    }}
    dt {{
      color: #5b6775;
      font-weight: 650;
    }}
    dd {{
      margin: 0;
      min-width: 0;
      overflow-wrap: anywhere;
    }}
    code {{
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>False Negative Review ({len(rows)} objects)</h1>
  </header>
  <main>
    {"".join(cards)}
  </main>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(doc)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Log validation false negatives for manual detector review.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/val_best.yaml")
    parser.add_argument("--dataset", default="data/coco_tl")
    parser.add_argument("--device", default=None, help="Override detector device, e.g. cpu or cuda")
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
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--confidence-threshold", type=float, default=None)
    parser.add_argument("--candidate-confidence-min", type=float, default=0.01)
    parser.add_argument("--nms-threshold", type=float, default=None)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--positive-images-only", action="store_true")
    parser.add_argument(
        "--top-half-only",
        "--top-40-only",
        "--top-third-only",
        dest="top_crop_only",
        action="store_true",
        help="Feed detector inference the configured top crop while keeping full-frame ground truth",
    )
    parser.add_argument(
        "--top-crop-fraction",
        type=float,
        default=None,
        help="Image-height fraction kept when top-crop inference is enabled",
    )
    parser.add_argument("--max-images", type=int, default=None, help="Optional debug limit on evaluated images")
    parser.add_argument("--sequences", nargs="+", default=None, help="Optional sequence filter, e.g. daySequence1")
    parser.add_argument("--size-bin-edges", type=float, nargs="+", default=DEFAULT_SIZE_BIN_EDGES)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-save-images", type=int, default=None, help="Limit copied overlays/crops, CSV still lists all FNs")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: runs/eval/false_negatives_<timestamp>",
    )
    args = parser.parse_args(argv)

    if args.iou_threshold <= 0 or args.iou_threshold > 1:
        parser.error("--iou-threshold must be in (0, 1]")
    if args.candidate_confidence_min < 0 or args.candidate_confidence_min > 1:
        parser.error("--candidate-confidence-min must be in [0, 1]")
    if args.confidence_threshold is not None and (
        args.confidence_threshold < 0 or args.confidence_threshold > 1
    ):
        parser.error("--confidence-threshold must be in [0, 1]")
    if len(args.size_bin_edges) < 2:
        parser.error("--size-bin-edges must contain at least two values")
    if any(args.size_bin_edges[i] >= args.size_bin_edges[i + 1] for i in range(len(args.size_bin_edges) - 1)):
        parser.error("--size-bin-edges must be strictly increasing")
    if args.top_crop_fraction is not None and not 0.0 < args.top_crop_fraction <= 1.0:
        parser.error("--top-crop-fraction must be in the range (0, 1]")

    from adas_perception.traffic_light.config import detector_input_size_from_args

    try:
        requested_input_size = detector_input_size_from_args(args.image_size, args.input_size)
    except ValueError as exc:
        parser.error(str(exc))

    cfg, model, device = _load_detector(
        args.config,
        args.device,
        input_size_override=requested_input_size,
    )
    input_size = tuple(int(v) for v in cfg.detector.input_size)
    confidence_threshold = (
        float(args.confidence_threshold)
        if args.confidence_threshold is not None
        else float(cfg.detector.confidence_threshold)
    )
    candidate_confidence_min = min(float(args.candidate_confidence_min), confidence_threshold)
    nms_threshold = float(args.nms_threshold) if args.nms_threshold is not None else float(cfg.detector.nms_threshold)
    top_crop_only = bool(
        args.top_crop_only
        or cfg.preprocess.top_crop_only
        or cfg.preprocess.top_third_only
    )
    top_crop_fraction = (
        float(cfg.preprocess.top_crop_fraction)
        if args.top_crop_fraction is None
        else float(args.top_crop_fraction)
    )
    logger.info("Top crop: enabled=%s | fraction=%.3f", top_crop_only, top_crop_fraction)

    detections, coco_gt, image_ids = _collect_detections(
        model=model,
        dataset_path=args.dataset,
        input_size=input_size,
        candidate_confidence_min=candidate_confidence_min,
        nms_threshold=nms_threshold,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        positive_images_only=args.positive_images_only,
        max_images=args.max_images,
        sequences=args.sequences,
        top_crop_only=top_crop_only,
        top_crop_fraction=top_crop_fraction,
    )

    gt_by_img, all_gt = _load_ground_truth(coco_gt, image_ids)
    final_detections = [det for det in detections if det.score >= confidence_threshold]
    matches, matched_gt_ids, matched_det_ids = _match_detections(
        detections=final_detections,
        gt_by_img=gt_by_img,
        iou_threshold=float(args.iou_threshold),
    )
    false_negatives = [gt for gt in all_gt if gt.global_idx not in matched_gt_ids]

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR / f"false_negatives_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _build_false_negative_rows(
        false_negatives=false_negatives,
        all_detections=detections,
        final_detections=final_detections,
        matched_det_ids=matched_det_ids,
        dataset_path=args.dataset,
        output_dir=output_dir,
        save_images=bool(args.save_images),
        max_save_images=args.max_save_images,
        confidence_threshold=confidence_threshold,
        iou_threshold=float(args.iou_threshold),
        size_bin_edges=[float(v) for v in args.size_bin_edges],
    )

    counts = {
        "num_images": len(image_ids),
        "gt_count": len(all_gt),
        "candidate_detection_count": len(detections),
        "final_detection_count": len(final_detections),
        "matched_gt_count": len(matched_gt_ids),
        "matched_detection_count": len(matched_det_ids),
        "false_negative_count": len(false_negatives),
        "recall": _safe_rate(len(matched_gt_ids), len(all_gt)),
    }
    parameters = {
        "config": str(args.config),
        "dataset": str(args.dataset),
        "detector_weights": cfg.detector.model_path,
        "input_size": list(input_size),
        "confidence_threshold": confidence_threshold,
        "candidate_confidence_min": candidate_confidence_min,
        "nms_threshold": nms_threshold,
        "iou_threshold": float(args.iou_threshold),
        "positive_images_only": bool(args.positive_images_only),
        "top_crop_only": bool(top_crop_only),
        "top_crop_fraction": float(top_crop_fraction),
        "full_frame_ground_truth": True,
        "max_images": args.max_images,
        "sequences": args.sequences,
        "size_bin_edges": [float(v) for v in args.size_bin_edges],
        "save_images": bool(args.save_images),
        "max_save_images": args.max_save_images,
    }
    payload = {
        "parameters": parameters,
        "counts": counts,
        "matches": [match.__dict__ for match in matches],
        "false_negatives": rows,
    }

    csv_path = output_dir / "false_negatives.csv"
    json_path = output_dir / "false_negatives.json"
    summary_path = output_dir / "summary.txt"
    html_path = output_dir / "index.html"

    _write_csv(rows, csv_path)
    _write_json(payload, json_path)
    _write_summary(summary_path, rows, parameters, counts)
    _write_html(rows, html_path)

    logger.info("GT objects: %d", len(all_gt))
    logger.info("Matched GT: %d", len(matched_gt_ids))
    logger.info("False negatives: %d", len(false_negatives))
    logger.info("Wrote CSV     -> %s", csv_path)
    logger.info("Wrote JSON    -> %s", json_path)
    logger.info("Wrote summary -> %s", summary_path)
    logger.info("Wrote gallery -> %s", html_path)


if __name__ == "__main__":
    main()
