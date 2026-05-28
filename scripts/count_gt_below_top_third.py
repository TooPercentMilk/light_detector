"""Count COCO ground-truth traffic lights below the top third of images.

By default, "below" means the annotation box has no overlap with the top
third of its image: bbox_y >= ceil(image_height / 3). This matches the labels
that would be dropped by top-third-only detector training.

Usage:

    python scripts/count_gt_below_top_third.py --dataset data/coco_tl
    python scripts/count_gt_below_top_third.py --dataset data/coco_tl --splits train val
    python scripts/count_gt_below_top_third.py --annotations data/coco_tl/annotations/instances_val.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SPLITS = ("train", "val", "test")


def _proportion(count: int, total: int) -> float:
    return count / total if total else 0.0


def _fmt_prop(count: int, total: int) -> str:
    return f"{count:,} / {total:,} ({_proportion(count, total) * 100:.2f}%)"


def _top_third_height(image_height: int) -> int:
    return max(1, int(math.ceil(image_height / 3.0)))


def _category_label(category: dict[str, Any]) -> str:
    name = str(category.get("name", ""))
    category_id = str(category.get("id", ""))
    return f"{category_id}:{name}"


def _resolve_category_ids(coco: dict[str, Any], filters: list[str]) -> set[int] | None:
    if not filters:
        return None

    normalized_filters = {item.strip().lower() for item in filters}
    resolved: set[int] = set()
    for category in coco.get("categories", []):
        category_id = int(category["id"])
        names = {
            str(category_id).lower(),
            str(category.get("name", "")).lower(),
            str(category.get("supercategory", "")).lower(),
        }
        if names & normalized_filters:
            resolved.add(category_id)

    missing = sorted(normalized_filters - {
        value
        for category in coco.get("categories", [])
        for value in {
            str(category.get("id", "")).lower(),
            str(category.get("name", "")).lower(),
            str(category.get("supercategory", "")).lower(),
        }
    })
    if missing:
        raise ValueError(f"Unknown category filter(s): {', '.join(missing)}")
    return resolved


def _annotation_files(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if args.annotations:
        path = Path(args.annotations)
        return [(path.stem.replace("instances_", ""), path)]

    dataset = Path(args.dataset)
    return [
        (split, dataset / "annotations" / f"instances_{split}.json")
        for split in args.splits
    ]


def analyze_file(path: Path, category_filters: list[str]) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    category_ids = _resolve_category_ids(coco, category_filters)
    image_by_id = {int(img["id"]): img for img in coco.get("images", [])}
    categories_by_id = {
        int(cat["id"]): cat for cat in coco.get("categories", [])
    }

    counts = {
        "total": 0,
        "fully_below": 0,
        "center_below": 0,
        "any_part_below": 0,
        "crosses_boundary": 0,
        "fully_in_top_third": 0,
        "invalid_bbox": 0,
        "missing_image": 0,
        "filtered_category": 0,
    }
    by_category: dict[int, dict[str, int]] = {}

    for ann in coco.get("annotations", []):
        category_id = int(ann.get("category_id", -1))
        if category_ids is not None and category_id not in category_ids:
            counts["filtered_category"] += 1
            continue

        bbox = ann.get("bbox", [])
        if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            counts["invalid_bbox"] += 1
            continue

        image = image_by_id.get(int(ann.get("image_id", -1)))
        if image is None:
            counts["missing_image"] += 1
            continue

        image_height = int(image["height"])
        boundary_y = _top_third_height(image_height)
        _, y, _, h = [float(value) for value in bbox]
        y2 = y + h
        center_y = y + h / 2.0

        counts["total"] += 1
        cat_counts = by_category.setdefault(
            category_id,
            {
                "total": 0,
                "fully_below": 0,
                "center_below": 0,
                "any_part_below": 0,
                "crosses_boundary": 0,
                "fully_in_top_third": 0,
            },
        )
        cat_counts["total"] += 1

        if y >= boundary_y:
            counts["fully_below"] += 1
            cat_counts["fully_below"] += 1
        if center_y >= boundary_y:
            counts["center_below"] += 1
            cat_counts["center_below"] += 1
        if y2 > boundary_y:
            counts["any_part_below"] += 1
            cat_counts["any_part_below"] += 1
        if y < boundary_y < y2:
            counts["crosses_boundary"] += 1
            cat_counts["crosses_boundary"] += 1
        if y2 <= boundary_y:
            counts["fully_in_top_third"] += 1
            cat_counts["fully_in_top_third"] += 1

    return {
        "path": str(path),
        "categories": {
            category_id: _category_label(category)
            for category_id, category in categories_by_id.items()
            if category_ids is None or category_id in category_ids
        },
        "counts": counts,
        "by_category": by_category,
    }


def print_report(split: str, result: dict[str, Any]) -> None:
    counts = result["counts"]
    total = counts["total"]
    print(f"=== {split} ===")
    print(f"annotations: {result['path']}")
    print(f"categories: {', '.join(result['categories'].values()) or 'none'}")
    print(f"total valid GT traffic lights: {_fmt_prop(total, total)}")
    print(f"fully below top third:       {_fmt_prop(counts['fully_below'], total)}")
    print(f"center below top third:      {_fmt_prop(counts['center_below'], total)}")
    print(f"any part below top third:    {_fmt_prop(counts['any_part_below'], total)}")
    print(f"crosses boundary:            {_fmt_prop(counts['crosses_boundary'], total)}")
    print(f"fully in top third:          {_fmt_prop(counts['fully_in_top_third'], total)}")

    if counts["invalid_bbox"] or counts["missing_image"] or counts["filtered_category"]:
        print(
            "skipped: "
            f"invalid_bbox={counts['invalid_bbox']:,}, "
            f"missing_image={counts['missing_image']:,}, "
            f"filtered_category={counts['filtered_category']:,}"
        )
    print()


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_counts: dict[str, int] = {}
    for result in results:
        for key, value in result["counts"].items():
            total_counts[key] = total_counts.get(key, 0) + int(value)
    return {"path": "aggregate", "categories": {}, "counts": total_counts}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Count ground-truth boxes below the top third in COCO annotations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=False)
    source.add_argument(
        "--dataset",
        default="data/coco_tl",
        help="COCO dataset root containing annotations/instances_<split>.json",
    )
    source.add_argument(
        "--annotations",
        help="Path to a single COCO annotation JSON file",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Dataset splits to analyze when using --dataset",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Optional category name, supercategory, or id to include; repeat for multiple",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the raw counts as JSON",
    )
    args = parser.parse_args(argv)

    reports: list[tuple[str, dict[str, Any]]] = []
    for split, path in _annotation_files(args):
        if not path.is_file():
            print(f"Skipping {split}: annotation file not found: {path}")
            continue
        result = analyze_file(path, args.category)
        reports.append((split, result))
        print_report(split, result)

    if len(reports) > 1:
        aggregate_result = aggregate([result for _, result in reports])
        print_report("aggregate", aggregate_result)

    if args.json_out is not None:
        payload = {
            split: result
            for split, result in reports
        }
        if len(reports) > 1:
            payload["aggregate"] = aggregate([result for _, result in reports])
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON report to {args.json_out}")


if __name__ == "__main__":
    main()
