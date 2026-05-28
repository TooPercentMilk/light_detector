"""Grid-search detector confidence thresholds on the COCO val split.

The script loads the detector from ``configs/val_best.yaml`` by default,
runs YOLOX inference once at the lowest requested threshold, then filters the
cached detections across the threshold grid. It writes JSON, CSV, and a PNG
plot showing detector performance as the confidence threshold changes.

Usage::

    python scripts/grid_search_confidence_thresholds.py

    python scripts/grid_search_confidence_thresholds.py \
        --config configs/val_best.yaml \
        --dataset data/coco_tl \
        --threshold-min 0.01 \
        --threshold-max 0.95 \
        --threshold-step 0.05

    python scripts/grid_search_confidence_thresholds.py \
        --thresholds 0.05 0.1 0.15 0.2 0.25 0.3 \
        --device cpu
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import logging
import os
import tempfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-5s  %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("runs/eval")
DEFAULT_OPTIMIZE_METRIC = "f1"
DEFAULT_THRESHOLDS = [0.01] + [round(i * 0.05, 2) for i in range(1, 20)]


def _threshold_grid(
    explicit_thresholds: list[float] | None,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> list[float]:
    if explicit_thresholds:
        thresholds = explicit_thresholds
    else:
        if threshold_step <= 0:
            raise ValueError("--threshold-step must be greater than 0")
        if threshold_min > threshold_max:
            raise ValueError("--threshold-min must be <= --threshold-max")
        thresholds = [
            round(float(v), 6)
            for v in np.arange(threshold_min, threshold_max + threshold_step * 0.5, threshold_step)
            if float(v) <= threshold_max + 1e-9
        ]

    thresholds = sorted(set(round(float(t), 6) for t in thresholds))
    bad = [t for t in thresholds if t < 0 or t > 1]
    if bad:
        raise ValueError(f"Confidence thresholds must be within [0, 1], got: {bad}")
    if not thresholds:
        raise ValueError("At least one threshold is required")
    return thresholds


def _torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)


def _load_best_detector(config_path: str, device_override: str | None):
    from adas_perception.traffic_light.config import load_config
    from yolox.exp import get_exp

    cfg = load_config(config_path)
    device_name = device_override or cfg.detector.device
    if str(device_name).startswith("cuda") and not torch.cuda.is_available():
        logger.warning("Config requested CUDA but CUDA is not available; using CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    exp = get_exp(None, cfg.detector.exp_name)
    exp.num_classes = cfg.detector.num_classes
    exp.test_size = tuple(cfg.detector.input_size)
    model = exp.get_model()

    weights_path = Path(cfg.detector.model_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Detector weights not found: {weights_path}")

    ckpt = _torch_load(weights_path, device)
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
) -> tuple[list[dict[str, Any]], Any, list[int]]:
    """Run model inference once and return COCO-format detection records."""
    from adas_perception.traffic_light.detector.evaluator import _COCOValDataset, _collate_fn
    from yolox.utils import postprocess

    val_dataset = _COCOValDataset(
        data_dir=str(Path(dataset_path).resolve()),
        json_file="instances_val.json",
        name="val",
        img_size=input_size,
        positive_images_only=positive_images_only,
    )
    if len(val_dataset) == 0:
        raise RuntimeError("Validation dataset is empty after image filtering")

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
        "Running inference once at min threshold %.4f over %d images",
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

            scores = (output[:, 4] * output[:, 5]).numpy()
            cls = output[:, 6].numpy().astype(int)

            for j in range(len(bboxes_xywh)):
                cat_idx = int(cls[j])
                if cat_idx >= len(val_dataset.class_ids):
                    continue
                detections.append(
                    {
                        "image_id": int(img_id),
                        "category_id": int(val_dataset.class_ids[cat_idx]),
                        "bbox": [float(v) for v in bboxes_xywh[j].numpy().tolist()],
                        "score": float(scores[j]),
                    }
                )

        if batch_idx % 25 == 0:
            logger.info("Processed %d batches; cached %d detections", batch_idx, len(detections))

    logger.info("Cached %d detections before grid filtering", len(detections))
    return detections, val_dataset.coco, list(val_dataset.img_ids)


def _coco_ap_metrics(coco_gt: Any, detections: list[dict[str, Any]], image_ids: list[int]) -> dict[str, float]:
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
    redirect = io.StringIO()
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(detections, f)
        with contextlib.redirect_stdout(redirect):
            coco_dt = coco_gt.loadRes(tmp_path)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.params.imgIds = image_ids
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass

    stats = coco_eval.stats
    return {
        "mAP": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
    }


def _xywh_to_xyxy(box: list[float]) -> np.ndarray:
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = max(0.0, float((box[2] - box[0]) * (box[3] - box[1])))
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return inter / np.maximum(union, 1e-6)


def _ground_truth_by_image(coco_gt: Any, image_ids: list[int]) -> tuple[dict[int, list[dict[str, Any]]], int]:
    image_id_set = set(image_ids)
    valid_cat_ids = set(coco_gt.getCatIds())
    gt_by_img: dict[int, list[dict[str, Any]]] = defaultdict(list)
    total_gt = 0

    for ann in coco_gt.dataset.get("annotations", []):
        image_id = int(ann.get("image_id"))
        if image_id not in image_id_set:
            continue
        if ann.get("category_id") not in valid_cat_ids:
            continue
        bbox = ann.get("bbox", [])
        if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue

        gt_by_img[image_id].append(
            {
                "category_id": int(ann["category_id"]),
                "bbox": _xywh_to_xyxy([float(v) for v in bbox]),
            }
        )
        total_gt += 1

    return gt_by_img, total_gt


def _fixed_iou_metrics(
    coco_gt: Any,
    detections: list[dict[str, Any]],
    image_ids: list[int],
    iou_threshold: float,
) -> dict[str, float | int]:
    gt_by_img, total_gt = _ground_truth_by_image(coco_gt, image_ids)
    matched_gt: dict[int, set[int]] = defaultdict(set)
    true_positives = 0

    for det in sorted(detections, key=lambda d: d["score"], reverse=True):
        img_id = int(det["image_id"])
        cat_id = int(det["category_id"])
        gt_items = gt_by_img.get(img_id, [])
        candidate_indices = [
            idx
            for idx, gt in enumerate(gt_items)
            if idx not in matched_gt[img_id] and gt["category_id"] == cat_id
        ]
        if not candidate_indices:
            continue

        pred_box = _xywh_to_xyxy([float(v) for v in det["bbox"]])
        gt_boxes = np.stack([gt_items[idx]["bbox"] for idx in candidate_indices], axis=0)
        ious = _iou_one_to_many(pred_box, gt_boxes)
        best_local = int(ious.argmax())
        if float(ious[best_local]) >= iou_threshold:
            matched_gt[img_id].add(candidate_indices[best_local])
            true_positives += 1

    total_predictions = len(detections)
    false_positives = total_predictions - true_positives
    false_negatives = total_gt - true_positives
    precision = true_positives / total_predictions if total_predictions else 0.0
    recall = true_positives / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "total_predictions": int(total_predictions),
        "total_gt": int(total_gt),
    }


def _evaluate_thresholds(
    coco_gt: Any,
    detections: list[dict[str, Any]],
    image_ids: list[int],
    thresholds: list[float],
    iou_threshold: float,
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        filtered = [d for d in detections if d["score"] >= threshold]
        ap_metrics = _coco_ap_metrics(coco_gt, filtered, image_ids)
        fixed_metrics = _fixed_iou_metrics(coco_gt, filtered, image_ids, iou_threshold)
        row: dict[str, float | int] = {
            "confidence_threshold": float(threshold),
            **ap_metrics,
            **fixed_metrics,
        }
        rows.append(row)
        logger.info(
            "thr=%.3f | mAP=%.4f mAP50=%.4f | P=%.4f R=%.4f F1=%.4f | preds=%d",
            threshold,
            row["mAP"],
            row["mAP_50"],
            row["precision"],
            row["recall"],
            row["f1"],
            row["total_predictions"],
        )
    return rows


def _write_csv(rows: list[dict[str, float | int]], out_csv: Path) -> None:
    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote CSV -> %s", out_csv)


def _plot_results(
    rows: list[dict[str, float | int]],
    out_png: Path,
    optimize_metric: str,
    iou_threshold: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to write the graph") from exc

    thresholds = [float(row["confidence_threshold"]) for row in rows]
    best = max(rows, key=lambda row: float(row[optimize_metric]))
    best_threshold = float(best["confidence_threshold"])
    best_value = float(best[optimize_metric])

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ap_series = [("mAP", "mAP"), ("mAP@50", "mAP_50"), ("mAP@75", "mAP_75")]
    for label, key in ap_series:
        axes[0].plot(thresholds, [float(row[key]) for row in rows], marker="o", label=label)
    axes[0].set_ylabel("AP")
    axes[0].set_title("COCO AP vs. confidence threshold")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")

    operating_series = [("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")]
    for label, key in operating_series:
        axes[1].plot(thresholds, [float(row[key]) for row in rows], marker="o", label=label)
    axes[1].set_ylabel(f"IoU >= {iou_threshold:.2f}")
    axes[1].set_xlabel("confidence threshold")
    axes[1].set_title("Fixed-IoU operating metrics vs. confidence threshold")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best")

    for ax in axes:
        ax.axvline(best_threshold, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1.02)

    fig.suptitle(
        f"Confidence threshold grid search - best {optimize_metric} "
        f"{best_value:.4f} at {best_threshold:.3f}"
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    logger.info("Wrote plot -> %s", out_png)


def _print_table(rows: list[dict[str, float | int]], optimize_metric: str) -> None:
    logger.info("")
    logger.info("%10s %8s %8s %8s %8s %8s %8s %8s", "threshold", "mAP", "mAP50", "mAP75", "prec", "recall", "f1", "preds")
    logger.info("-" * 78)
    for row in rows:
        logger.info(
            "%10.3f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8d",
            row["confidence_threshold"],
            row["mAP"],
            row["mAP_50"],
            row["mAP_75"],
            row["precision"],
            row["recall"],
            row["f1"],
            row["total_predictions"],
        )

    best = max(rows, key=lambda row: float(row[optimize_metric]))
    logger.info("")
    logger.info(
        "Best %s: threshold=%.3f value=%.4f",
        optimize_metric,
        best["confidence_threshold"],
        best[optimize_metric],
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Grid-search detector confidence thresholds using the best model config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="configs/val_best.yaml", help="Config pointing to the best model")
    parser.add_argument("--dataset", default="data/coco_tl", help="COCO-format dataset root")
    parser.add_argument("--device", default=None, help="Override detector device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    parser.add_argument("--nms-threshold", type=float, default=None, help="Override config detector NMS threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU used for precision/recall/F1")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Explicit confidence thresholds. Default: 0.01, 0.05, 0.10, ..., 0.95",
    )
    parser.add_argument("--threshold-min", type=float, default=None, help="Custom grid minimum")
    parser.add_argument("--threshold-max", type=float, default=None, help="Custom grid maximum")
    parser.add_argument("--threshold-step", type=float, default=None, help="Custom grid step")
    parser.add_argument(
        "--positive-images-only",
        action="store_true",
        help="Evaluate only val images that contain at least one valid annotation",
    )
    parser.add_argument(
        "--optimize-metric",
        default=DEFAULT_OPTIMIZE_METRIC,
        choices=["f1", "precision", "recall", "mAP", "mAP_50", "mAP_75"],
        help="Metric used to mark the best threshold on the plot",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output prefix without extension. Default: runs/eval/conf_threshold_grid_<timestamp>",
    )
    args = parser.parse_args(argv)

    if (
        args.thresholds is None
        and args.threshold_min is None
        and args.threshold_max is None
        and args.threshold_step is None
    ):
        thresholds = DEFAULT_THRESHOLDS
    else:
        thresholds = _threshold_grid(
            args.thresholds,
            0.05 if args.threshold_min is None else args.threshold_min,
            0.95 if args.threshold_max is None else args.threshold_max,
            0.05 if args.threshold_step is None else args.threshold_step,
        )

    cfg, model, device = _load_best_detector(args.config, args.device)
    nms_threshold = args.nms_threshold if args.nms_threshold is not None else cfg.detector.nms_threshold
    input_size = tuple(int(v) for v in cfg.detector.input_size)

    detections, coco_gt, image_ids = _collect_detections(
        model=model,
        dataset_path=args.dataset,
        input_size=input_size,
        min_conf_threshold=min(thresholds),
        nms_threshold=nms_threshold,
        batch_size=args.batch_size,
        device=device,
        num_workers=args.num_workers,
        positive_images_only=args.positive_images_only,
    )

    rows = _evaluate_thresholds(
        coco_gt=coco_gt,
        detections=detections,
        image_ids=image_ids,
        thresholds=thresholds,
        iou_threshold=args.iou_threshold,
    )
    _print_table(rows, args.optimize_metric)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_prefix = Path(args.output_prefix) if args.output_prefix else RESULTS_DIR / f"conf_threshold_grid_{timestamp}"
    best = max(rows, key=lambda row: float(row[args.optimize_metric]))
    results = {
        "config": str(args.config),
        "dataset": str(args.dataset),
        "detector_weights": cfg.detector.model_path,
        "input_size": list(input_size),
        "nms_threshold": float(nms_threshold),
        "iou_threshold": float(args.iou_threshold),
        "positive_images_only": bool(args.positive_images_only),
        "num_images": len(image_ids),
        "num_cached_detections": len(detections),
        "optimize_metric": args.optimize_metric,
        "best": {
            "confidence_threshold": float(best["confidence_threshold"]),
            "value": float(best[args.optimize_metric]),
        },
        "thresholds": rows,
    }

    out_json = out_prefix.with_suffix(".json")
    out_csv = out_prefix.with_suffix(".csv")
    out_png = out_prefix.with_suffix(".png")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote JSON -> %s", out_json)
    _write_csv(rows, out_csv)
    _plot_results(rows, out_png, args.optimize_metric, args.iou_threshold)


if __name__ == "__main__":
    main()
