"""Analyze IoU severity for low-IoU false negatives.

This script reads a ``false_negatives.json`` file produced by
``scripts/log_false_negatives.py``, filters to rows whose miss reason is
``low_iou``, keeps only detections that pass the model confidence threshold,
calculates each kept detection's IoU from its stored bounding box, and reports
the proportion at or below each IoU threshold.

Example:

    python scripts/analyze_low_iou_false_negatives.py \
        runs/eval/false_negatives_2026-05-22_0019/false_negatives.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _as_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric {field_name}, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Expected finite {field_name}, got {value!r}")
    return result


def _as_box_xywh(value: Any, field_name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Expected {field_name} to be a 4-value box, got {value!r}")
    box = tuple(_as_float(part, field_name) for part in value)
    if box[2] < 0 or box[3] < 0:
        raise ValueError(f"Expected non-negative width/height for {field_name}, got {value!r}")
    return box


def _xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return (x, y, x + w, y + h)


def _box_area(box: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou_xywh(
    gt_box_xywh: tuple[float, float, float, float],
    candidate_box_xywh: tuple[float, float, float, float],
) -> float:
    gt = _xywh_to_xyxy(gt_box_xywh)
    candidate = _xywh_to_xyxy(candidate_box_xywh)
    x1 = max(gt[0], candidate[0])
    y1 = max(gt[1], candidate[1])
    x2 = min(gt[2], candidate[2])
    y2 = min(gt[3], candidate[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _box_area(gt) + _box_area(candidate) - intersection
    return intersection / max(union, 1e-6)


def _confidence_threshold(row: dict[str, Any], threshold_override: float | None) -> float:
    threshold = threshold_override if threshold_override is not None else row.get("confidence_threshold")
    if threshold is None:
        raise ValueError(
            "Rows must include confidence_threshold, or pass --confidence-threshold to the analyzer."
        )
    return _as_float(threshold, "confidence_threshold")


def _score_passes(value: Any, threshold: float, field_name: str) -> bool:
    if value is None:
        return False
    return _as_float(value, field_name) >= threshold


def _thresholded_detection_box(
    row: dict[str, Any],
    threshold_override: float | None,
) -> tuple[float, float, float, float] | None:
    threshold = _confidence_threshold(row, threshold_override)
    final_box = row.get("best_final_bbox_xywh")
    if final_box is not None and _score_passes(row.get("best_final_score"), threshold, "best_final_score"):
        return _as_box_xywh(final_box, "best_final_bbox_xywh")

    candidate_box = row.get("best_candidate_bbox_xywh")
    if candidate_box is not None and _score_passes(
        row.get("best_candidate_score"),
        threshold,
        "best_candidate_score",
    ):
        return _as_box_xywh(candidate_box, "best_candidate_bbox_xywh")

    return None


def _calculate_thresholded_detection_iou(
    row: dict[str, Any],
    threshold_override: float | None,
) -> float:
    gt_box = _as_box_xywh(row.get("gt_bbox_xywh"), "gt_bbox_xywh")
    detection_box = _thresholded_detection_box(row, threshold_override)
    if detection_box is None:
        raise ValueError("Cannot calculate IoU for a row without a confidence-thresholded detection box")
    return _iou_xywh(gt_box, detection_box)


def _load_false_negatives(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    rows = payload.get("false_negatives")
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a false_negatives list")
    return rows


def _low_iou_rows(
    rows: list[dict[str, Any]],
    confidence_threshold: float | None,
) -> tuple[list[dict[str, Any]], int]:
    selected = [row for row in rows if row.get("miss_reason") == "low_iou"]
    thresholded = [
        row
        for row in selected
        if _thresholded_detection_box(row, confidence_threshold) is not None
    ]
    missing_boxes = [
        row
        for row in thresholded
        if row.get("gt_bbox_xywh") is None
    ]
    if missing_boxes:
        sample = missing_boxes[0]
        raise ValueError(
            "Found low_iou rows without GT boxes; "
            f"sample image_id={sample.get('image_id')!r} ann_id={sample.get('ann_id')!r}"
        )
    return thresholded, len(selected) - len(thresholded)


def _threshold_rows(values: list[float], thresholds: list[float]) -> list[dict[str, float | int]]:
    total = len(values)
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        count = sum(1 for value in values if value <= threshold)
        rows.append(
            {
                "threshold": threshold,
                "count_at_or_below": count,
                "total": total,
                "proportion": _safe_rate(count, total),
            }
        )
    return rows


def analyze(
    rows: list[dict[str, Any]],
    thresholds: list[float],
    confidence_threshold: float | None = None,
) -> dict[str, Any]:
    low_rows, ignored_below_confidence = _low_iou_rows(rows, confidence_threshold)
    object_ious = [_calculate_thresholded_detection_iou(row, confidence_threshold) for row in low_rows]

    image_to_ious: dict[str, list[float]] = defaultdict(list)
    for row, iou in zip(low_rows, object_ious):
        image_id = row.get("image_id")
        image_key = str(image_id if image_id is not None else row.get("file_name", ""))
        image_to_ious[image_key].append(iou)

    # An image is counted at a threshold if any analyzed low-IoU false negative
    # in that image has calculated thresholded-detection IoU <= threshold.
    image_min_ious = [min(ious) for ious in image_to_ious.values()]

    return {
        "counts": {
            "all_false_negatives": len(rows),
            "low_iou_false_negatives_analyzed": len(low_rows),
            "low_iou_ignored_below_confidence": ignored_below_confidence,
            "low_iou_unique_images_analyzed": len(image_to_ious),
        },
        "confidence_filter": (
            f"detection score >= {confidence_threshold}"
            if confidence_threshold is not None
            else "detection score >= row confidence_threshold"
        ),
        "iou_source": (
            "calculated from gt_bbox_xywh and best_final_bbox_xywh when available; "
            "falls back to best_candidate_bbox_xywh only when that candidate passes confidence"
        ),
        "object_level": _threshold_rows(object_ious, thresholds),
        "image_level_min_iou": _threshold_rows(image_min_ious, thresholds),
    }


def _format_table(title: str, rows: list[dict[str, float | int]]) -> list[str]:
    lines = [
        title,
        f"{'iou <= ':<10} {'count':>8} {'total':>8} {'proportion':>12}",
        "-" * 42,
    ]
    for row in rows:
        lines.append(
            f"{float(row['threshold']):<10.1f} "
            f"{int(row['count_at_or_below']):>8} "
            f"{int(row['total']):>8} "
            f"{float(row['proportion']):>11.4f}"
        )
    return lines


def print_report(analysis: dict[str, Any], source_path: Path) -> None:
    counts = analysis["counts"]
    lines = [
        "LOW-IOU FALSE NEGATIVE THRESHOLD ANALYSIS",
        "=" * 47,
        f"Source: {source_path}",
        f"All false negatives: {counts['all_false_negatives']}",
        f"Low-IoU false negatives analyzed: {counts['low_iou_false_negatives_analyzed']}",
        f"Low-IoU ignored below confidence: {counts['low_iou_ignored_below_confidence']}",
        f"Low-IoU unique images analyzed: {counts['low_iou_unique_images_analyzed']}",
        f"Confidence filter: {analysis['confidence_filter']}",
        f"IoU source: {analysis['iou_source']}",
        "",
        *_format_table("Object-level thresholded low_iou false negatives", analysis["object_level"]),
        "",
        *_format_table("Image-level, minimum thresholded low_iou per image", analysis["image_level_min_iou"]),
    ]
    print("\n".join(lines))


def write_csv(analysis: dict[str, Any], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["level", "threshold", "count_at_or_below", "total", "proportion"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for level, rows in [
            ("object_level", analysis["object_level"]),
            ("image_level_min_iou", analysis["image_level_min_iou"]),
        ]:
            for row in rows:
                writer.writerow({"level": level, **row})


def write_json(analysis: dict[str, Any], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Report low_iou false-negative proportions at inclusive IoU thresholds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "false_negatives_json",
        type=Path,
        help="Path to false_negatives.json from scripts/log_false_negatives.py.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="Inclusive IoU thresholds to evaluate.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Override confidence threshold. Default uses each row's confidence_threshold.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--output-csv", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args(argv)

    if any(threshold < 0 or threshold > 1 for threshold in args.thresholds):
        parser.error("--thresholds must all be in [0, 1]")
    if args.confidence_threshold is not None and not 0 <= args.confidence_threshold <= 1:
        parser.error("--confidence-threshold must be in [0, 1]")
    thresholds = sorted(float(threshold) for threshold in args.thresholds)

    rows = _load_false_negatives(args.false_negatives_json)
    result = analyze(rows, thresholds, args.confidence_threshold)
    print_report(result, args.false_negatives_json)

    if args.output_json is not None:
        write_json(result, args.output_json)
    if args.output_csv is not None:
        write_csv(result, args.output_csv)


if __name__ == "__main__":
    main()
