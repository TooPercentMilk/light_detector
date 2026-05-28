"""Analyze low-confidence detections that still match ground truth.

This script looks for "almost good" detections: detector outputs whose
combined YOLOX confidence is in a low band, but whose box matches a ground
truth traffic light at a fixed IoU threshold.

It writes:

* a text report that answers the investigation questions,
* a JSON file with the same aggregates, and
* a CSV of every matched low-confidence detection for manual review.

Example:

    python scripts/analyze_low_conf_detections.py \
        --config configs/val_best.yaml \
        --dataset data/coco_tl \
        --confidence-min 0.01 \
        --confidence-max 0.25 \
        --iou-threshold 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
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

TAG_TO_COLOR: dict[str, str] = {
    "go": "green",
    "goLeft": "green",
    "goForward": "green",
    "stop": "red",
    "stopLeft": "red",
    "warning": "yellow",
    "warningLeft": "yellow",
}

DEFAULT_SIZE_BIN_EDGES = [0.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 96.0, 1e9]
COLOR_ORDER = ["red", "yellow", "green", "unknown"]


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
class MatchItem:
    det_index: int
    detection: dict[str, Any]
    gt: GTItem
    iou: float


def _sequence_from_filename(fname: str) -> str:
    if "--" in fname:
        return fname.split("--", 1)[0]
    return "unknown"


def _xywh_to_xyxy(box: list[float] | tuple[float, float, float, float]) -> tuple[float, float, float, float]:
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


def _intersection_over_min_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    min_area = max(1e-6, min(_box_area(a), _box_area(b)))
    return inter / min_area


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


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_count_rate(count: int, total: int) -> str:
    return f"{count}/{total} ({_pct(_safe_rate(count, total))})"


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    clean = np.array([float(v) for v in values if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if clean.size == 0:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "mean": None,
            "p75": None,
            "p90": None,
            "max": None,
        }

    return {
        "count": int(clean.size),
        "min": float(np.min(clean)),
        "p10": float(np.percentile(clean, 10)),
        "p25": float(np.percentile(clean, 25)),
        "median": float(np.median(clean)),
        "mean": float(np.mean(clean)),
        "p75": float(np.percentile(clean, 75)),
        "p90": float(np.percentile(clean, 90)),
        "max": float(np.max(clean)),
    }


def _counter_rows(
    low_counter: Counter[str],
    low_total: int,
    baseline_counter: Counter[str],
    baseline_total: int,
    order: list[str] | None = None,
) -> list[dict[str, float | int | str | None]]:
    keys = order or sorted(set(low_counter) | set(baseline_counter))
    rows: list[dict[str, float | int | str | None]] = []
    for key in keys:
        low_count = int(low_counter.get(key, 0))
        base_count = int(baseline_counter.get(key, 0))
        low_rate = _safe_rate(low_count, low_total)
        base_rate = _safe_rate(base_count, baseline_total)
        enrichment = low_rate / base_rate if base_rate else None
        rows.append(
            {
                "name": key,
                "low_conf_matches": low_count,
                "low_conf_share": low_rate,
                "baseline_gt": base_count,
                "baseline_share": base_rate,
                "enrichment_vs_baseline": enrichment,
            }
        )
    return rows


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
    min_conf_threshold: float,
    nms_threshold: float,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    positive_images_only: bool,
    max_images: int | None,
    sequences: list[str] | None,
):
    from adas_perception.traffic_light.detector.evaluator import _COCOValDataset, _collate_fn
    from yolox.utils import postprocess

    val_dataset = _COCOValDataset(
        data_dir=str(Path(dataset_path).resolve()),
        json_file="instances_val.json",
        name="val",
        img_size=input_size,
        positive_images_only=positive_images_only,
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

    detections: list[dict[str, Any]] = []
    num_classes = len(val_dataset.class_ids)

    logger.info(
        "Running inference at min confidence %.4f over %d images",
        min_conf_threshold,
        len(val_dataset),
    )

    for batch_idx, (imgs, _, info_imgs, ids) in enumerate(val_loader, start=1):
        with torch.no_grad():
            outputs = model(imgs.to(device))

        outputs = postprocess(outputs, num_classes, min_conf_threshold, nms_threshold)

        for output, img_h, img_w, img_id in zip(outputs, info_imgs[0], info_imgs[1], ids):
            if output is None:
                continue

            output = output.cpu()
            bboxes = output[:, 0:4]
            scale = min(input_size[0] / float(img_h), input_size[1] / float(img_w))
            bboxes /= scale

            bboxes_xywh = bboxes.clone()
            bboxes_xywh[:, 2] -= bboxes_xywh[:, 0]
            bboxes_xywh[:, 3] -= bboxes_xywh[:, 1]

            objectness = output[:, 4].numpy()
            class_conf = output[:, 5].numpy()
            scores = objectness * class_conf
            cls = output[:, 6].numpy().astype(int)

            for idx in range(len(bboxes_xywh)):
                cat_idx = int(cls[idx])
                if cat_idx >= len(val_dataset.class_ids):
                    continue
                detections.append(
                    {
                        "image_id": int(img_id),
                        "category_id": int(val_dataset.class_ids[cat_idx]),
                        "bbox": [float(v) for v in bboxes_xywh[idx].numpy().tolist()],
                        "score": float(scores[idx]),
                        "objectness": float(objectness[idx]),
                        "class_confidence": float(class_conf[idx]),
                    }
                )

        if batch_idx % 25 == 0:
            logger.info("Processed %d batches; cached %d detections", batch_idx, len(detections))

    logger.info("Cached %d detections before low-confidence filtering", len(detections))
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
        color = TAG_TO_COLOR.get(lisa_tag, "unknown")
        bbox_xywh = tuple(float(v) for v in bbox)
        item = GTItem(
            global_idx=len(all_gt),
            ann_id=int(ann.get("id", len(all_gt))),
            image_id=image_id,
            file_name=str(image_info["file_name"]),
            sequence=_sequence_from_filename(str(image_info["file_name"])),
            category_id=int(ann["category_id"]),
            lisa_tag=lisa_tag,
            color=color,
            bbox_xyxy=_xywh_to_xyxy(bbox_xywh),
            bbox_xywh=bbox_xywh,
            image_width=int(image_info["width"]),
            image_height=int(image_info["height"]),
        )
        gt_by_img[image_id].append(item)
        all_gt.append(item)

    return gt_by_img, all_gt


def _match_low_conf_detections(
    detections: list[dict[str, Any]],
    gt_by_img: dict[int, list[GTItem]],
    confidence_min: float,
    confidence_max: float,
    iou_threshold: float,
) -> tuple[list[MatchItem], int]:
    low_conf = [
        (idx, det)
        for idx, det in enumerate(detections)
        if confidence_min <= float(det["score"]) <= confidence_max
    ]

    candidates: list[tuple[float, float, int, int, dict[str, Any], GTItem]] = []
    for det_index, det in low_conf:
        img_id = int(det["image_id"])
        gt_items = [gt for gt in gt_by_img.get(img_id, []) if gt.category_id == int(det["category_id"])]
        if not gt_items:
            continue

        pred_box = _xywh_to_xyxy(tuple(float(v) for v in det["bbox"]))
        ious = _iou_one_to_many(pred_box, [gt.bbox_xyxy for gt in gt_items])
        for gt, iou in zip(gt_items, ious):
            iou_float = float(iou)
            if iou_float >= iou_threshold:
                candidates.append((iou_float, float(det["score"]), det_index, gt.global_idx, det, gt))

    # Use one-to-one matching so one easy GT cannot dominate the analysis with
    # duplicate low-confidence boxes.
    candidates.sort(key=lambda row: (row[0], row[1]), reverse=True)
    used_det: set[int] = set()
    used_gt: set[int] = set()
    matches: list[MatchItem] = []
    for iou, _, det_index, gt_index, det, gt in candidates:
        if det_index in used_det or gt_index in used_gt:
            continue
        used_det.add(det_index)
        used_gt.add(gt_index)
        matches.append(MatchItem(det_index=det_index, detection=det, gt=gt, iou=iou))

    return matches, len(low_conf)


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


def _laplacian_blur_score(crop: np.ndarray | None) -> float | None:
    if crop is None or crop.size == 0:
        return None
    if crop.shape[0] < 3 or crop.shape[1] < 3:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _compute_gt_features(
    gt_by_img: dict[int, list[GTItem]],
    dataset_path: str,
    size_bin_edges: list[float],
    tiny_side_threshold: float,
    distant_side_threshold: float,
    edge_margin_pct: float,
    edge_margin_px: float,
    overlap_min_area_ratio: float,
    blur_crop_padding: float,
    skip_blur: bool,
) -> dict[int, dict[str, Any]]:
    image_dir = Path(dataset_path).resolve() / "val"
    features: dict[int, dict[str, Any]] = {}

    for image_id, gt_items in sorted(gt_by_img.items()):
        image = None
        if not skip_blur and gt_items:
            image = cv2.imread(str(image_dir / gt_items[0].file_name))
            if image is None:
                logger.warning("Could not read image for blur analysis: %s", image_dir / gt_items[0].file_name)

        for gt in gt_items:
            x1, y1, x2, y2 = gt.bbox_xyxy
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            area = bw * bh
            sqrt_area = math.sqrt(area)
            image_area = max(1, gt.image_width * gt.image_height)
            margin_x = max(edge_margin_px, edge_margin_pct * gt.image_width)
            margin_y = max(edge_margin_px, edge_margin_pct * gt.image_height)

            edge_sides: list[str] = []
            if x1 <= margin_x:
                edge_sides.append("left")
            if y1 <= margin_y:
                edge_sides.append("top")
            if x2 >= gt.image_width - margin_x:
                edge_sides.append("right")
            if y2 >= gt.image_height - margin_y:
                edge_sides.append("bottom")

            clipped_edge = x1 <= 0 or y1 <= 0 or x2 >= gt.image_width - 1 or y2 >= gt.image_height - 1

            overlaps_other_gt = any(
                other.global_idx != gt.global_idx
                and _intersection_over_min_area(gt.bbox_xyxy, other.bbox_xyxy) >= overlap_min_area_ratio
                for other in gt_items
            )

            blur_score = None
            if image is not None:
                blur_score = _laplacian_blur_score(_crop_with_padding(image, gt.bbox_xyxy, blur_crop_padding))

            features[gt.global_idx] = {
                "ann_id": gt.ann_id,
                "image_id": gt.image_id,
                "file_name": gt.file_name,
                "sequence": gt.sequence,
                "lisa_tag": gt.lisa_tag,
                "color": gt.color,
                "gt_bbox_xywh": list(gt.bbox_xywh),
                "gt_width": float(bw),
                "gt_height": float(bh),
                "gt_area": float(area),
                "gt_sqrt_area": float(sqrt_area),
                "gt_area_fraction": float(area / image_area),
                "size_bucket": _bucket_value(sqrt_area, size_bin_edges),
                "tiny": bool(sqrt_area < tiny_side_threshold),
                "distant_proxy": bool(sqrt_area < distant_side_threshold),
                "center_x_norm": float(((x1 + x2) * 0.5) / gt.image_width),
                "center_y_norm": float(((y1 + y2) * 0.5) / gt.image_height),
                "edge": bool(edge_sides),
                "edge_sides": edge_sides,
                "clipped_edge": bool(clipped_edge),
                "overlaps_other_gt_proxy": bool(overlaps_other_gt),
                "blur_score": blur_score,
            }

    return features


def _add_match_features(
    matches: list[MatchItem],
    gt_features: dict[int, dict[str, Any]],
    poor_iou_threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in matches:
        gt = match.gt
        det = match.detection
        pred_xywh = tuple(float(v) for v in det["bbox"])
        pred_xyxy = _xywh_to_xyxy(pred_xywh)
        gx1, gy1, gx2, gy2 = gt.bbox_xyxy
        px1, py1, px2, py2 = pred_xyxy
        gw, gh = max(1e-6, gx2 - gx1), max(1e-6, gy2 - gy1)
        pw, ph = max(1e-6, px2 - px1), max(1e-6, py2 - py1)
        gt_area = gw * gh
        pred_area = pw * ph

        gt_cx, gt_cy = (gx1 + gx2) * 0.5, (gy1 + gy2) * 0.5
        pred_cx, pred_cy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
        gt_diag = max(1e-6, math.sqrt(gw * gw + gh * gh))
        center_error_norm = math.hypot(pred_cx - gt_cx, pred_cy - gt_cy) / gt_diag

        row = dict(gt_features[gt.global_idx])
        row.update(
            {
                "det_index": int(match.det_index),
                "score": float(det["score"]),
                "objectness": float(det["objectness"]),
                "class_confidence": float(det["class_confidence"]),
                "iou": float(match.iou),
                "iou_bucket": _iou_bucket(match.iou),
                "poor_box_proxy": bool(match.iou < poor_iou_threshold),
                "pred_bbox_xywh": list(pred_xywh),
                "pred_width": float(pw),
                "pred_height": float(ph),
                "pred_area": float(pred_area),
                "pred_sqrt_area": float(math.sqrt(pred_area)),
                "center_error_norm": float(center_error_norm),
                "area_ratio_pred_to_gt": float(pred_area / gt_area),
                "width_ratio_pred_to_gt": float(pw / gw),
                "height_ratio_pred_to_gt": float(ph / gh),
            }
        )
        rows.append(row)
    return rows


def _iou_bucket(iou: float) -> str:
    if iou < 0.6:
        return "0.50-0.60"
    if iou < 0.75:
        return "0.60-0.75"
    if iou < 0.9:
        return "0.75-0.90"
    return "0.90-1.00"


def _feature_rate(
    key: str,
    low_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> dict[str, float | int | None]:
    low_count = sum(1 for row in low_rows if row.get(key))
    baseline_count = sum(1 for row in baseline_rows if row.get(key))
    low_total = len(low_rows)
    baseline_total = len(baseline_rows)
    low_rate = _safe_rate(low_count, low_total)
    baseline_rate = _safe_rate(baseline_count, baseline_total)
    return {
        "low_conf_count": int(low_count),
        "low_conf_total": int(low_total),
        "low_conf_rate": low_rate,
        "baseline_count": int(baseline_count),
        "baseline_total": int(baseline_total),
        "baseline_rate": baseline_rate,
        "enrichment_vs_baseline": (low_rate / baseline_rate if baseline_rate else None),
    }


def _build_analysis(
    low_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    parameters: dict[str, Any],
    low_conf_detection_count: int,
    total_detection_count: int,
    blur_threshold_override: float | None,
    blur_percentile: float,
    sequence_enrichment_threshold: float,
    min_sequence_count: int,
) -> dict[str, Any]:
    low_total = len(low_rows)
    baseline_total = len(baseline_rows)

    baseline_blur_scores = [
        float(row["blur_score"])
        for row in baseline_rows
        if row.get("blur_score") is not None and math.isfinite(float(row["blur_score"]))
    ]
    low_blur_scores = [
        float(row["blur_score"])
        for row in low_rows
        if row.get("blur_score") is not None and math.isfinite(float(row["blur_score"]))
    ]
    if blur_threshold_override is not None:
        blur_threshold = float(blur_threshold_override)
    elif baseline_blur_scores:
        blur_threshold = float(np.percentile(np.array(baseline_blur_scores), blur_percentile))
    else:
        blur_threshold = None

    if blur_threshold is not None:
        for row in low_rows:
            row["low_blur_proxy"] = (
                row.get("blur_score") is not None and float(row["blur_score"]) <= blur_threshold
            )
        for row in baseline_rows:
            row["low_blur_proxy"] = (
                row.get("blur_score") is not None and float(row["blur_score"]) <= blur_threshold
            )

    color_rows = _counter_rows(
        Counter(str(row["color"]) for row in low_rows),
        low_total,
        Counter(str(row["color"]) for row in baseline_rows),
        baseline_total,
        COLOR_ORDER,
    )

    size_order = [_bucket_label(parameters["size_bin_edges"], i) for i in range(len(parameters["size_bin_edges"]) - 1)]
    size_rows = _counter_rows(
        Counter(str(row["size_bucket"]) for row in low_rows),
        low_total,
        Counter(str(row["size_bucket"]) for row in baseline_rows),
        baseline_total,
        size_order,
    )

    sequence_rows = _counter_rows(
        Counter(str(row["sequence"]) for row in low_rows),
        low_total,
        Counter(str(row["sequence"]) for row in baseline_rows),
        baseline_total,
    )
    sequence_rows.sort(key=lambda row: (int(row["low_conf_matches"]), float(row["enrichment_vs_baseline"] or 0)), reverse=True)
    overrepresented_sequences = [
        row
        for row in sequence_rows
        if int(row["low_conf_matches"]) >= min_sequence_count
        and row["enrichment_vs_baseline"] is not None
        and float(row["enrichment_vs_baseline"]) >= sequence_enrichment_threshold
    ]

    iou_rows = _counter_rows(
        Counter(str(row["iou_bucket"]) for row in low_rows),
        low_total,
        Counter(),
        0,
        ["0.50-0.60", "0.60-0.75", "0.75-0.90", "0.90-1.00"],
    )

    red_yellow_green = [row for row in color_rows if row["name"] in {"red", "yellow", "green"}]
    color_shares = [float(row["low_conf_share"]) for row in red_yellow_green]
    max_equal_delta = max((abs(share - (1.0 / 3.0)) for share in color_shares), default=0.0)

    answers = {
        "tiny_lights": _feature_rate("tiny", low_rows, baseline_rows),
        "distant_proxy": _feature_rate("distant_proxy", low_rows, baseline_rows),
        "partial_occlusion_proxy": {
            "note": (
                "LISA COCO annotations do not include occlusion labels. "
                "This uses clipped-at-image-edge and overlap-with-neighbor-GT proxies only."
            ),
            "clipped_edge": _feature_rate("clipped_edge", low_rows, baseline_rows),
            "overlaps_other_gt_proxy": _feature_rate("overlaps_other_gt_proxy", low_rows, baseline_rows),
        },
        "motion_blur_proxy": {
            "note": (
                "Uses variance of Laplacian on a padded GT crop; lower scores are blurrier. "
                "This is a blur/sharpness proxy, not a true motion-label."
            ),
            "threshold": blur_threshold,
            "threshold_source": (
                "manual" if blur_threshold_override is not None else f"baseline_p{blur_percentile:g}"
            ),
            "low_blur": _feature_rate("low_blur_proxy", low_rows, baseline_rows) if blur_threshold is not None else None,
            "low_conf_blur_summary": _numeric_summary(low_blur_scores),
            "baseline_blur_summary": _numeric_summary(baseline_blur_scores),
        },
        "color_distribution": {
            "rows": color_rows,
            "max_abs_delta_from_equal_rgb": float(max_equal_delta),
            "roughly_equal_rgb_proxy": bool(max_equal_delta <= 0.15),
        },
        "image_edges": _feature_rate("edge", low_rows, baseline_rows),
        "box_quality": {
            "poor_box_proxy": _feature_rate("poor_box_proxy", low_rows, []),
            "iou_summary": _numeric_summary([float(row["iou"]) for row in low_rows]),
            "iou_buckets": iou_rows,
            "center_error_norm_summary": _numeric_summary([float(row["center_error_norm"]) for row in low_rows]),
            "area_ratio_pred_to_gt_summary": _numeric_summary([float(row["area_ratio_pred_to_gt"]) for row in low_rows]),
        },
        "lisa_sequences": {
            "rows": sequence_rows,
            "overrepresented_sequences": overrepresented_sequences,
            "enrichment_threshold": float(sequence_enrichment_threshold),
            "min_sequence_count": int(min_sequence_count),
        },
        "size_buckets": {"rows": size_rows},
        "score_decomposition": {
            "score_summary": _numeric_summary([float(row["score"]) for row in low_rows]),
            "objectness_summary": _numeric_summary([float(row["objectness"]) for row in low_rows]),
            "class_confidence_summary": _numeric_summary([float(row["class_confidence"]) for row in low_rows]),
        },
    }

    return {
        "parameters": parameters,
        "counts": {
            "num_images": int(parameters["num_images"]),
            "baseline_gt_count": int(baseline_total),
            "total_detections_at_min_conf": int(total_detection_count),
            "low_conf_detection_count": int(low_conf_detection_count),
            "low_conf_matched_count": int(low_total),
            "low_conf_unmatched_count": int(low_conf_detection_count - low_total),
            "low_conf_match_rate": _safe_rate(low_total, low_conf_detection_count),
        },
        "answers": answers,
    }


def _yes_no_mostly(rate: float, enrichment: float | None, mostly_threshold: float = 0.5) -> str:
    if rate >= mostly_threshold:
        return "Yes"
    if enrichment is not None and enrichment >= 1.5:
        return "Overrepresented, but not most"
    return "No"


def _yes_no_enriched(rate: float, enrichment: float | None, rate_threshold: float = 0.25) -> str:
    if rate >= rate_threshold and enrichment is not None and enrichment >= 1.25:
        return "Yes"
    if enrichment is not None and enrichment >= 1.5:
        return "Overrepresented, but uncommon"
    return "No"


def _append_feature_answer(lines: list[str], title: str, feature: dict[str, Any], threshold_desc: str) -> None:
    low_count = int(feature["low_conf_count"])
    low_total = int(feature["low_conf_total"])
    base_count = int(feature["baseline_count"])
    base_total = int(feature["baseline_total"])
    low_rate = float(feature["low_conf_rate"])
    enrichment = feature["enrichment_vs_baseline"]
    answer = _yes_no_mostly(low_rate, float(enrichment) if enrichment is not None else None)
    enrich_txt = "n/a" if enrichment is None else f"{float(enrichment):.2f}x"
    lines.extend(
        [
            "",
            title,
            f"Answer: {answer}.",
            f"Low-confidence matches: {_fmt_count_rate(low_count, low_total)}.",
            f"All GT baseline: {_fmt_count_rate(base_count, base_total)}.",
            f"Enrichment vs baseline: {enrich_txt}.",
            f"Definition: {threshold_desc}.",
        ]
    )


def _format_rows_table(
    rows: list[dict[str, Any]],
    name_col: str,
    max_rows: int | None = None,
) -> list[str]:
    selected = rows if max_rows is None else rows[:max_rows]
    out = [f"{name_col:<18} {'low':>8} {'low%':>8} {'gt':>8} {'gt%':>8} {'enrich':>8}"]
    out.append("-" * 64)
    for row in selected:
        enrich = row["enrichment_vs_baseline"]
        enrich_txt = "n/a" if enrich is None else f"{float(enrich):.2f}x"
        out.append(
            f"{str(row['name']):<18} "
            f"{int(row['low_conf_matches']):>8} "
            f"{_pct(float(row['low_conf_share'])):>8} "
            f"{int(row['baseline_gt']):>8} "
            f"{_pct(float(row['baseline_share'])):>8} "
            f"{enrich_txt:>8}"
        )
    return out


def _summary_value(summary: dict[str, Any], key: str) -> str:
    value = summary.get(key)
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4g}"


def _write_report(analysis: dict[str, Any], report_path: Path) -> None:
    counts = analysis["counts"]
    params = analysis["parameters"]
    answers = analysis["answers"]
    lines: list[str] = []

    lines.extend(
        [
            "LOW-CONFIDENCE TRUE-POSITIVE DETECTION ANALYSIS",
            "=" * 55,
            f"Config: {params['config']}",
            f"Dataset: {params['dataset']}",
            f"Detector weights: {params['detector_weights']}",
            f"Images evaluated: {counts['num_images']}",
            f"Definition: score in [{params['confidence_min']}, {params['confidence_max']}] "
            f"and one-to-one GT match at IoU >= {params['iou_threshold']}",
            "",
            "Important caveat: LISA conversion keeps color tags and sequence names, "
            "but not true occlusion, depth, or motion-blur labels. Those questions "
            "are answered with explicit proxies below.",
            "",
            "Overall counts",
            f"All GT objects in evaluated images: {counts['baseline_gt_count']}",
            f"Detections at min confidence: {counts['total_detections_at_min_conf']}",
            f"Low-confidence detections: {counts['low_conf_detection_count']}",
            f"Matched low-confidence detections: {counts['low_conf_matched_count']}",
            f"Unmatched low-confidence detections: {counts['low_conf_unmatched_count']}",
            f"Low-confidence match rate: {_pct(float(counts['low_conf_match_rate']))}",
        ]
    )

    _append_feature_answer(
        lines,
        "1. Are they mostly tiny lights?",
        answers["tiny_lights"],
        f"tiny = sqrt(GT bbox area) < {params['tiny_side_threshold']} px.",
    )

    lines.extend(["", "Size buckets"])
    lines.extend(_format_rows_table(answers["size_buckets"]["rows"], "sqrt(area) px"))

    _append_feature_answer(
        lines,
        "2. Are they distant?",
        answers["distant_proxy"],
        f"distant proxy = sqrt(GT bbox area) < {params['distant_side_threshold']} px.",
    )

    occl = answers["partial_occlusion_proxy"]
    clipped = occl["clipped_edge"]
    overlap = occl["overlaps_other_gt_proxy"]
    lines.extend(
        [
            "",
            "3. Are they partially occluded?",
            "Answer: Not directly measurable from these annotations.",
            f"Proxy A, clipped at image boundary: {_fmt_count_rate(int(clipped['low_conf_count']), int(clipped['low_conf_total']))} "
            f"vs baseline {_fmt_count_rate(int(clipped['baseline_count']), int(clipped['baseline_total']))}.",
            f"Proxy B, overlaps another GT box: {_fmt_count_rate(int(overlap['low_conf_count']), int(overlap['low_conf_total']))} "
            f"vs baseline {_fmt_count_rate(int(overlap['baseline_count']), int(overlap['baseline_total']))}.",
            f"Note: {occl['note']}",
        ]
    )

    blur = answers["motion_blur_proxy"]
    low_blur = blur["low_blur"]
    if low_blur is not None:
        answer = _yes_no_enriched(
            float(low_blur["low_conf_rate"]),
            float(low_blur["enrichment_vs_baseline"]) if low_blur["enrichment_vs_baseline"] is not None else None,
        )
        blur_threshold = "n/a" if blur["threshold"] is None else f"{float(blur['threshold']):.4g}"
        lines.extend(
            [
                "",
                "4. Are they motion-blurred?",
                f"Answer: {answer}, using the blur proxy.",
                f"Low-blur matches: {_fmt_count_rate(int(low_blur['low_conf_count']), int(low_blur['low_conf_total']))}.",
                f"All GT low-blur baseline: {_fmt_count_rate(int(low_blur['baseline_count']), int(low_blur['baseline_total']))}.",
                f"Blur threshold: {blur_threshold} ({blur['threshold_source']}).",
                "Blur score summary, low-confidence matches: "
                f"median={_summary_value(blur['low_conf_blur_summary'], 'median')}, "
                f"mean={_summary_value(blur['low_conf_blur_summary'], 'mean')}.",
                "Blur score summary, all GT baseline: "
                f"median={_summary_value(blur['baseline_blur_summary'], 'median')}, "
                f"mean={_summary_value(blur['baseline_blur_summary'], 'mean')}.",
                f"Note: {blur['note']}",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "4. Are they motion-blurred?",
                "Answer: Blur analysis was skipped or no valid crops were available.",
            ]
        )

    color = answers["color_distribution"]
    lines.extend(
        [
            "",
            "5. Are yellow/red/green equally represented?",
            f"Answer: {'Roughly yes' if color['roughly_equal_rgb_proxy'] else 'No'} "
            f"(max absolute delta from an even RGB split = {_pct(float(color['max_abs_delta_from_equal_rgb']))}).",
        ]
    )
    lines.extend(_format_rows_table(color["rows"], "color"))

    edge = answers["image_edges"]
    edge_answer = _yes_no_enriched(
        float(edge["low_conf_rate"]),
        float(edge["enrichment_vs_baseline"]) if edge["enrichment_vs_baseline"] is not None else None,
    )
    lines.extend(
        [
            "",
            "6. Are they at image edges?",
            f"Answer: {edge_answer}.",
            f"Low-confidence edge matches: {_fmt_count_rate(int(edge['low_conf_count']), int(edge['low_conf_total']))}.",
            f"All GT edge baseline: {_fmt_count_rate(int(edge['baseline_count']), int(edge['baseline_total']))}.",
            f"Definition: GT box touches a margin of max({params['edge_margin_px']} px, "
            f"{params['edge_margin_pct'] * 100:.1f}% of image dimension).",
        ]
    )

    box = answers["box_quality"]
    poor = box["poor_box_proxy"]
    poor_rate = float(poor["low_conf_rate"])
    box_answer = "Yes" if poor_rate >= 0.5 else ("Some are borderline" if poor_rate >= 0.25 else "Mostly no")
    lines.extend(
        [
            "",
            "7. Are they poorly boxed?",
            f"Answer: {box_answer}.",
            f"Poor-box proxy: {_fmt_count_rate(int(poor['low_conf_count']), int(poor['low_conf_total']))} "
            f"with IoU < {params['poor_iou_threshold']}.",
            "IoU summary: "
            f"median={_summary_value(box['iou_summary'], 'median')}, "
            f"mean={_summary_value(box['iou_summary'], 'mean')}, "
            f"p10={_summary_value(box['iou_summary'], 'p10')}.",
            "Center error normalized by GT diagonal: "
            f"median={_summary_value(box['center_error_norm_summary'], 'median')}, "
            f"p90={_summary_value(box['center_error_norm_summary'], 'p90')}.",
            "Pred/GT area ratio: "
            f"median={_summary_value(box['area_ratio_pred_to_gt_summary'], 'median')}, "
            f"p10={_summary_value(box['area_ratio_pred_to_gt_summary'], 'p10')}, "
            f"p90={_summary_value(box['area_ratio_pred_to_gt_summary'], 'p90')}.",
            "",
            "IoU buckets",
        ]
    )
    lines.extend(_format_rows_table(box["iou_buckets"], "iou", max_rows=None))

    seq = answers["lisa_sequences"]
    lines.extend(
        [
            "",
            "8. Are they from specific LISA sequences?",
            f"Answer: {'Yes' if seq['overrepresented_sequences'] else 'No strong sequence concentration by the configured rule'}.",
            f"Configured rule: sequence enrichment >= {seq['enrichment_threshold']}x "
            f"with at least {seq['min_sequence_count']} matches.",
            "",
            "Sequence distribution",
        ]
    )
    lines.extend(_format_rows_table(seq["rows"], "sequence"))

    if seq["overrepresented_sequences"]:
        lines.extend(["", "Overrepresented sequences"])
        lines.extend(_format_rows_table(seq["overrepresented_sequences"], "sequence"))

    score = answers["score_decomposition"]
    lines.extend(
        [
            "",
            "Score decomposition",
            "YOLOX combined score = objectness * class confidence.",
            "Combined score: "
            f"median={_summary_value(score['score_summary'], 'median')}, "
            f"mean={_summary_value(score['score_summary'], 'mean')}.",
            "Objectness: "
            f"median={_summary_value(score['objectness_summary'], 'median')}, "
            f"mean={_summary_value(score['objectness_summary'], 'mean')}.",
            "Class confidence: "
            f"median={_summary_value(score['class_confidence_summary'], 'median')}, "
            f"mean={_summary_value(score['class_confidence_summary'], 'mean')}.",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def _write_json(analysis: dict[str, Any], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)


def _write_matches_csv(rows: list[dict[str, Any]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file_name",
        "sequence",
        "image_id",
        "ann_id",
        "lisa_tag",
        "color",
        "score",
        "objectness",
        "class_confidence",
        "iou",
        "iou_bucket",
        "poor_box_proxy",
        "gt_bbox_xywh",
        "pred_bbox_xywh",
        "gt_sqrt_area",
        "gt_area_fraction",
        "size_bucket",
        "tiny",
        "distant_proxy",
        "edge",
        "edge_sides",
        "clipped_edge",
        "overlaps_other_gt_proxy",
        "blur_score",
        "low_blur_proxy",
        "center_error_norm",
        "area_ratio_pred_to_gt",
        "width_ratio_pred_to_gt",
        "height_ratio_pred_to_gt",
        "center_x_norm",
        "center_y_norm",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in fieldnames}
            out["edge_sides"] = "+".join(row.get("edge_sides", []))
            out["gt_bbox_xywh"] = json.dumps(row.get("gt_bbox_xywh", []))
            out["pred_bbox_xywh"] = json.dumps(row.get("pred_bbox_xywh", []))
            writer.writerow(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Analyze low-confidence detections that match GT at a fixed IoU.",
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
    parser.add_argument("--nms-threshold", type=float, default=None)
    parser.add_argument("--confidence-min", type=float, default=0.01)
    parser.add_argument("--confidence-max", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--tiny-side-threshold", type=float, default=16.0)
    parser.add_argument("--distant-side-threshold", type=float, default=12.0)
    parser.add_argument("--size-bin-edges", type=float, nargs="+", default=DEFAULT_SIZE_BIN_EDGES)
    parser.add_argument("--edge-margin-pct", type=float, default=0.05)
    parser.add_argument("--edge-margin-px", type=float, default=8.0)
    parser.add_argument("--overlap-min-area-ratio", type=float, default=0.1)
    parser.add_argument("--poor-iou-threshold", type=float, default=0.6)
    parser.add_argument("--blur-crop-padding", type=float, default=0.5)
    parser.add_argument("--blur-threshold", type=float, default=None)
    parser.add_argument("--blur-percentile", type=float, default=20.0)
    parser.add_argument("--skip-blur", action="store_true")
    parser.add_argument("--positive-images-only", action="store_true")
    parser.add_argument("--max-images", type=int, default=None, help="Optional debug limit on evaluated images")
    parser.add_argument("--sequences", nargs="+", default=None, help="Optional sequence filter, e.g. daySequence1")
    parser.add_argument("--sequence-enrichment-threshold", type=float, default=1.5)
    parser.add_argument("--min-sequence-count", type=int, default=5)
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix without extension. Default: runs/eval/low_conf_detections_<timestamp>",
    )
    args = parser.parse_args(argv)

    if args.confidence_min < 0 or args.confidence_max > 1 or args.confidence_min > args.confidence_max:
        parser.error("Confidence bounds must satisfy 0 <= min <= max <= 1")
    if len(args.size_bin_edges) < 2:
        parser.error("--size-bin-edges must contain at least two values")
    if any(args.size_bin_edges[i] >= args.size_bin_edges[i + 1] for i in range(len(args.size_bin_edges) - 1)):
        parser.error("--size-bin-edges must be strictly increasing")
    if args.iou_threshold <= 0 or args.iou_threshold > 1:
        parser.error("--iou-threshold must be in (0, 1]")

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
    nms_threshold = args.nms_threshold if args.nms_threshold is not None else cfg.detector.nms_threshold

    detections, coco_gt, image_ids = _collect_detections(
        model=model,
        dataset_path=args.dataset,
        input_size=input_size,
        min_conf_threshold=args.confidence_min,
        nms_threshold=nms_threshold,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        positive_images_only=args.positive_images_only,
        max_images=args.max_images,
        sequences=args.sequences,
    )

    gt_by_img, all_gt = _load_ground_truth(coco_gt, image_ids)
    gt_features = _compute_gt_features(
        gt_by_img=gt_by_img,
        dataset_path=args.dataset,
        size_bin_edges=[float(v) for v in args.size_bin_edges],
        tiny_side_threshold=args.tiny_side_threshold,
        distant_side_threshold=args.distant_side_threshold,
        edge_margin_pct=args.edge_margin_pct,
        edge_margin_px=args.edge_margin_px,
        overlap_min_area_ratio=args.overlap_min_area_ratio,
        blur_crop_padding=args.blur_crop_padding,
        skip_blur=args.skip_blur,
    )

    baseline_rows = [gt_features[gt.global_idx] for gt in all_gt]
    matches, low_conf_detection_count = _match_low_conf_detections(
        detections=detections,
        gt_by_img=gt_by_img,
        confidence_min=args.confidence_min,
        confidence_max=args.confidence_max,
        iou_threshold=args.iou_threshold,
    )
    low_rows = _add_match_features(matches, gt_features, args.poor_iou_threshold)

    logger.info(
        "Matched %d of %d low-confidence detections to GT at IoU >= %.2f",
        len(low_rows),
        low_conf_detection_count,
        args.iou_threshold,
    )

    parameters = {
        "config": str(args.config),
        "dataset": str(args.dataset),
        "detector_weights": cfg.detector.model_path,
        "input_size": list(input_size),
        "nms_threshold": float(nms_threshold),
        "confidence_min": float(args.confidence_min),
        "confidence_max": float(args.confidence_max),
        "iou_threshold": float(args.iou_threshold),
        "tiny_side_threshold": float(args.tiny_side_threshold),
        "distant_side_threshold": float(args.distant_side_threshold),
        "size_bin_edges": [float(v) for v in args.size_bin_edges],
        "edge_margin_pct": float(args.edge_margin_pct),
        "edge_margin_px": float(args.edge_margin_px),
        "overlap_min_area_ratio": float(args.overlap_min_area_ratio),
        "poor_iou_threshold": float(args.poor_iou_threshold),
        "blur_crop_padding": float(args.blur_crop_padding),
        "skip_blur": bool(args.skip_blur),
        "positive_images_only": bool(args.positive_images_only),
        "max_images": args.max_images,
        "sequences": args.sequences,
        "num_images": len(image_ids),
    }
    analysis = _build_analysis(
        low_rows=low_rows,
        baseline_rows=baseline_rows,
        parameters=parameters,
        low_conf_detection_count=low_conf_detection_count,
        total_detection_count=len(detections),
        blur_threshold_override=args.blur_threshold,
        blur_percentile=args.blur_percentile,
        sequence_enrichment_threshold=args.sequence_enrichment_threshold,
        min_sequence_count=args.min_sequence_count,
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_prefix = Path(args.output_prefix) if args.output_prefix else RESULTS_DIR / f"low_conf_detections_{timestamp}"
    report_path = out_prefix.with_suffix(".txt")
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_name(out_prefix.name + "_matches").with_suffix(".csv")

    _write_report(analysis, report_path)
    _write_json(analysis, json_path)
    _write_matches_csv(low_rows, csv_path)

    logger.info("Wrote report -> %s", report_path)
    logger.info("Wrote JSON   -> %s", json_path)
    logger.info("Wrote CSV    -> %s", csv_path)


if __name__ == "__main__":
    main()
