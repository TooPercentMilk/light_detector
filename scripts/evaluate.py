"""Evaluate the full traffic-light pipeline on the COCO validation set.

Runs both:
1. **Detector evaluation** — COCO mAP metrics (mAP, mAP_50, mAP_75) via
   ``pycocotools`` against the ground-truth bounding boxes.
2. **Classifier evaluation** — per-class and overall accuracy / precision /
   recall / F1 on ROIs cropped from ground-truth bounding boxes.

Usage::

    # Use best trained weights (val_best.yaml)
    python scripts/evaluate.py --config configs/val_best.yaml --dataset data/coco_tl

    # Override device
    python scripts/evaluate.py --config configs/val_best.yaml --dataset data/coco_tl --device cpu
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Detector evaluation (COCO mAP)
# ------------------------------------------------------------------

def evaluate_detector(
    config_path: str,
    dataset_path: str,
    device: str | None = None,
    batch_size: int = 8,
) -> dict[str, float]:
    """Load the detector from *config_path* and evaluate on the val split."""
    from adas_perception.traffic_light.config import load_config
    from adas_perception.traffic_light.detector.evaluator import evaluate

    cfg = load_config(config_path)
    dev = device or cfg.detector.device

    # Build the raw YOLOX model (same path the wrapper uses)
    from yolox.exp import get_exp

    exp = get_exp(None, cfg.detector.exp_name)
    exp.num_classes = cfg.detector.num_classes
    exp.test_size = tuple(cfg.detector.input_size)
    model = exp.get_model()

    weights_path = Path(cfg.detector.model_path)
    if not weights_path.is_file():
        logger.error("Detector weights not found: %s", weights_path)
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

    ckpt = torch.load(str(weights_path), map_location=dev, weights_only=True)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.to(dev)
    model.eval()

    logger.info("Evaluating detector on %s …", dataset_path)
    metrics = evaluate(
        model,
        dataset_path=dataset_path,
        input_size=tuple(cfg.detector.input_size),
        batch_size=batch_size,
        device=dev,
    )
    return metrics


# ------------------------------------------------------------------
# Classifier evaluation (accuracy / precision / recall / F1)
# ------------------------------------------------------------------

_TAG_TO_CLASS: dict[str, int] = {
    "go": 2,         # green
    "goLeft": 2,
    "goForward": 2,
    "stop": 0,        # red
    "stopLeft": 0,
    "warning": 1,     # yellow
    "warningLeft": 1,
}

_CLASS_NAMES = {0: "red", 1: "yellow", 2: "green", 3: "off"}


def evaluate_classifier(
    config_path: str,
    dataset_path: str,
    device: str | None = None,
) -> dict[str, float]:
    """Evaluate the state classifier on ground-truth crops from the val split."""
    from adas_perception.traffic_light.config import load_config
    from adas_perception.traffic_light.state.classifier import StateClassifier

    cfg = load_config(config_path)
    dev = device or cfg.classifier.device

    classifier = StateClassifier(cfg.classifier)
    classifier.load_model(cfg.classifier.model_path, dev)

    # Load validation annotations
    data_dir = Path(dataset_path).resolve()
    ann_path = data_dir / "annotations" / "instances_val.json"
    if not ann_path.is_file():
        logger.error("Val annotations not found: %s", ann_path)
        return {}

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    image_dir = data_dir / "val"

    # Collect (ground_truth_class, predicted_class) pairs
    y_true: list[int] = []
    y_pred: list[int] = []
    skipped = 0

    # Cache loaded images to avoid re-reading for multiple annotations
    image_cache: dict[int, np.ndarray | None] = {}

    for ann in coco["annotations"]:
        tag = ann.get("attributes", {}).get("lisa_tag")
        if tag is None or tag not in _TAG_TO_CLASS:
            skipped += 1
            continue

        gt_cls = _TAG_TO_CLASS[tag]
        img_id = ann["image_id"]

        # Load image (cached)
        if img_id not in image_cache:
            fname = id_to_file.get(img_id)
            if fname is None:
                image_cache[img_id] = None
            else:
                image_cache[img_id] = cv2.imread(str(image_dir / fname))
        image = image_cache[img_id]
        if image is None:
            continue

        # Crop ROI from ground-truth bbox (COCO [x, y, w, h])
        x, y, w, h = [int(round(v)) for v in ann["bbox"]]
        ih, iw = image.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + w), min(ih, y + h)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        pred_state, _ = classifier.classify(roi)
        # Map LightState enum back to class index
        state_to_idx = {"red": 0, "yellow": 1, "green": 2, "off": 3}
        pred_cls = state_to_idx.get(pred_state.value, -1)
        if pred_cls < 0:
            continue

        y_true.append(gt_cls)
        y_pred.append(pred_cls)

    if skipped:
        logger.info("Skipped %d annotations with unknown/missing tags", skipped)

    if not y_true:
        logger.warning("No valid classification samples found")
        return {}

    # Compute per-class precision / recall / F1
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    overall_correct = int((y_true_arr == y_pred_arr).sum())
    overall_acc = overall_correct / len(y_true_arr)

    metrics: dict[str, float] = {"accuracy": overall_acc}
    present_classes = sorted(set(y_true) | set(y_pred))

    logger.info("")
    logger.info("%-10s  %6s  %6s  %6s  %7s", "Class", "Prec", "Recall", "F1", "Support")
    logger.info("-" * 48)

    precisions, recalls, f1s, supports = [], [], [], []
    for cls_idx in present_classes:
        tp = int(((y_true_arr == cls_idx) & (y_pred_arr == cls_idx)).sum())
        fp = int(((y_true_arr != cls_idx) & (y_pred_arr == cls_idx)).sum())
        fn = int(((y_true_arr == cls_idx) & (y_pred_arr != cls_idx)).sum())
        support = int((y_true_arr == cls_idx).sum())

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        name = _CLASS_NAMES.get(cls_idx, f"class_{cls_idx}")
        logger.info("%-10s  %6.3f  %6.3f  %6.3f  %7d", name, prec, rec, f1, support)

        metrics[f"{name}_precision"] = prec
        metrics[f"{name}_recall"] = rec
        metrics[f"{name}_f1"] = f1

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(support)

    # Macro averages
    macro_prec = np.mean(precisions) if precisions else 0.0
    macro_rec = np.mean(recalls) if recalls else 0.0
    macro_f1 = np.mean(f1s) if f1s else 0.0

    # Weighted averages
    total_support = sum(supports)
    w_prec = sum(p * s for p, s in zip(precisions, supports)) / total_support if total_support else 0.0
    w_rec = sum(r * s for r, s in zip(recalls, supports)) / total_support if total_support else 0.0
    w_f1 = sum(f * s for f, s in zip(f1s, supports)) / total_support if total_support else 0.0

    logger.info("-" * 48)
    logger.info("%-10s  %6.3f  %6.3f  %6.3f  %7d", "macro", macro_prec, macro_rec, macro_f1, total_support)
    logger.info("%-10s  %6.3f  %6.3f  %6.3f  %7d", "weighted", w_prec, w_rec, w_f1, total_support)
    logger.info("")
    logger.info("Overall accuracy: %.2f%% (%d / %d)", overall_acc * 100, overall_correct, len(y_true_arr))

    metrics["macro_precision"] = float(macro_prec)
    metrics["macro_recall"] = float(macro_rec)
    metrics["macro_f1"] = float(macro_f1)
    metrics["weighted_precision"] = float(w_prec)
    metrics["weighted_recall"] = float(w_rec)
    metrics["weighted_f1"] = float(w_f1)

    return metrics


# ------------------------------------------------------------------
# Full pipeline evaluation (end-to-end)
# ------------------------------------------------------------------

def evaluate_pipeline(
    config_path: str,
    dataset_path: str,
    device: str | None = None,
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Run the full pipeline on val images and evaluate end-to-end.

    For each image, runs detect → track → classify and matches predicted
    TrafficLight outputs to ground-truth annotations using IoU.  Reports
    combined detection + classification accuracy.
    """
    from adas_perception.traffic_light.config import load_config
    from adas_perception.traffic_light.node import TrafficLightNode

    cfg = load_config(config_path)
    if device:
        cfg.detector.device = device
        cfg.classifier.device = device

    node = TrafficLightNode(cfg)

    # Load val annotations
    data_dir = Path(dataset_path).resolve()
    ann_path = data_dir / "annotations" / "instances_val.json"
    if not ann_path.is_file():
        logger.error("Val annotations not found: %s", ann_path)
        return {}

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    anns_by_img: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    image_dir = data_dir / "val"

    total_gt = 0
    total_detected = 0
    total_correct_state = 0
    total_predictions = 0

    state_to_idx = {"red": 0, "yellow": 1, "green": 2, "off": 3}

    for frame_id, (img_id, info) in enumerate(sorted(id_to_file.items())):
        img_path = image_dir / info
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        # Run pipeline
        lights = node.process_frame(image, frame_id)
        total_predictions += len(lights)

        gt_anns = anns_by_img.get(img_id, [])
        # Filter to annotated GT with known tags
        gt_boxes = []
        gt_classes = []
        for ann in gt_anns:
            tag = ann.get("attributes", {}).get("lisa_tag")
            if tag is None or tag not in _TAG_TO_CLASS:
                continue
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])
            gt_classes.append(_TAG_TO_CLASS[tag])
        total_gt += len(gt_boxes)

        if not gt_boxes or not lights:
            continue

        # Match predictions to GT by IoU
        gt_boxes_arr = np.array(gt_boxes, dtype=np.float32)
        matched_gt = set()

        for light in lights:
            pred_box = light.bbox.reshape(1, 4)
            # Compute IoU with all GT boxes
            ious = _compute_iou(pred_box, gt_boxes_arr).flatten()
            best_idx = int(ious.argmax())
            if ious[best_idx] >= iou_threshold and best_idx not in matched_gt:
                matched_gt.add(best_idx)
                total_detected += 1
                # Check state classification
                pred_cls = state_to_idx.get(light.state.value, -1)
                if pred_cls == gt_classes[best_idx]:
                    total_correct_state += 1

    metrics = {
        "total_gt": total_gt,
        "total_predictions": total_predictions,
        "total_detected": total_detected,
        "detection_recall": total_detected / total_gt if total_gt else 0.0,
        "detection_precision": total_detected / total_predictions if total_predictions else 0.0,
        "state_accuracy_on_matched": (
            total_correct_state / total_detected if total_detected else 0.0
        ),
        "end_to_end_accuracy": (
            total_correct_state / total_gt if total_gt else 0.0
        ),
    }

    logger.info("")
    logger.info("=" * 50)
    logger.info("END-TO-END PIPELINE RESULTS (IoU >= %.2f)", iou_threshold)
    logger.info("=" * 50)
    logger.info("Ground-truth objects:       %d", total_gt)
    logger.info("Predicted objects:          %d", total_predictions)
    logger.info("Matched (TP detections):    %d", total_detected)
    logger.info("Correct state on matched:   %d", total_correct_state)
    logger.info("Detection recall:           %.2f%%", metrics["detection_recall"] * 100)
    logger.info("Detection precision:        %.2f%%", metrics["detection_precision"] * 100)
    logger.info("State accuracy (matched):   %.2f%%", metrics["state_accuracy_on_matched"] * 100)
    logger.info("End-to-end accuracy:        %.2f%%", metrics["end_to_end_accuracy"] * 100)

    return metrics


def _compute_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of [x1,y1,x2,y2] boxes."""
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the traffic-light pipeline on a COCO validation set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/val_best.yaml",
        help="Pipeline config YAML (should point to best weights)",
    )
    parser.add_argument(
        "--dataset",
        default="data/coco_tl",
        help="Path to COCO-format dataset directory",
    )
    parser.add_argument("--device", default=None, help="cuda or cpu (auto from config if omitted)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for detector eval")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for end-to-end matching",
    )
    parser.add_argument(
        "--skip-detector",
        action="store_true",
        help="Skip COCO mAP detector evaluation",
    )
    parser.add_argument(
        "--skip-classifier",
        action="store_true",
        help="Skip standalone classifier evaluation",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="Skip end-to-end pipeline evaluation",
    )

    args = parser.parse_args(argv)

    results: dict[str, dict] = {}

    # --- 1. Detector mAP ---
    if not args.skip_detector:
        logger.info("=" * 60)
        logger.info("DETECTOR EVALUATION (COCO mAP)")
        logger.info("=" * 60)
        det_metrics = evaluate_detector(
            args.config, args.dataset, args.device, args.batch_size
        )
        results["detector"] = det_metrics
        logger.info("mAP: %.4f | mAP_50: %.4f | mAP_75: %.4f",
                     det_metrics["mAP"], det_metrics["mAP_50"], det_metrics["mAP_75"])

    # --- 2. Classifier accuracy ---
    if not args.skip_classifier:
        logger.info("")
        logger.info("=" * 60)
        logger.info("CLASSIFIER EVALUATION (per-class metrics)")
        logger.info("=" * 60)
        cls_metrics = evaluate_classifier(args.config, args.dataset, args.device)
        results["classifier"] = cls_metrics

    # --- 3. Full pipeline end-to-end ---
    if not args.skip_pipeline:
        logger.info("")
        logger.info("=" * 60)
        logger.info("PIPELINE EVALUATION (end-to-end)")
        logger.info("=" * 60)
        pipe_metrics = evaluate_pipeline(
            args.config, args.dataset, args.device, args.iou_threshold
        )
        results["pipeline"] = pipe_metrics

    # --- Summary ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    if "detector" in results:
        d = results["detector"]
        logger.info("Detector  — mAP: %.4f  mAP@50: %.4f  mAP@75: %.4f",
                     d["mAP"], d["mAP_50"], d["mAP_75"])
    if "classifier" in results:
        c = results["classifier"]
        logger.info("Classifier — accuracy: %.2f%%  macro-F1: %.3f  weighted-F1: %.3f",
                     c.get("accuracy", 0) * 100,
                     c.get("macro_f1", 0),
                     c.get("weighted_f1", 0))
    if "pipeline" in results:
        p = results["pipeline"]
        logger.info("Pipeline  — det-recall: %.2f%%  det-precision: %.2f%%  "
                     "state-acc: %.2f%%  e2e-acc: %.2f%%",
                     p.get("detection_recall", 0) * 100,
                     p.get("detection_precision", 0) * 100,
                     p.get("state_accuracy_on_matched", 0) * 100,
                     p.get("end_to_end_accuracy", 0) * 100)


if __name__ == "__main__":
    main()
