"""Convert the ATLAS traffic light dataset to COCO format for YOLOX.

Expected ATLAS layout::

    data/ATLAS/
      ATLAS_classes.yaml
      train/front_medium/images/*.jpg
      train/front_medium/labels/*.txt
      ...
      test/front_wide/images/*.jpg
      test/front_wide/labels/*.txt

Output layout used by this project::

    data/coco_atlas/
      train/*.jpg
      val/*.jpg
      test/*.jpg
      annotations/instances_train.json
      annotations/instances_val.json
      annotations/instances_test.json

ATLAS label files are YOLO text files:
``class_id x_center y_center width height`` with normalized coordinates.
The COCO JSON written here stores boxes as pixel ``[x, y, width, height]``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SOURCE_SPLITS = ("train", "test")
STATE4_CATEGORIES = ("off", "red", "yellow", "green")
STATE4_CATEGORY_IDS = {name: i + 1 for i, name in enumerate(STATE4_CATEGORIES)}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SourceImage:
    source_split: str
    camera: str
    image_path: Path
    label_path: Path


def load_atlas_classes(path: Path) -> dict[int, str]:
    """Load ``ATLAS_classes.yaml`` as ``{class_id: class_name}``."""
    with open(path, "r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    names = payload.get("names")
    if not isinstance(names, dict):
        raise ValueError(f"Expected a 'names' mapping in {path}")

    classes: dict[int, str] = {}
    for raw_id, raw_name in names.items():
        class_id = int(raw_id)
        # PyYAML's YAML 1.1 resolver parses the unquoted ATLAS label "off"
        # as boolean False. Restore the intended class name.
        class_name = "off" if raw_name is False else str(raw_name)
        classes[class_id] = class_name

    if not classes:
        raise ValueError(f"No classes found in {path}")
    expected = set(range(max(classes) + 1))
    missing = sorted(expected - set(classes))
    if missing:
        raise ValueError(f"Missing class IDs in {path}: {missing}")
    return dict(sorted(classes.items()))


def discover_images(
    atlas_root: Path,
    cameras: set[str] | None = None,
    max_images_per_camera: int | None = None,
) -> list[SourceImage]:
    """Discover ATLAS image/label pairs under train/test camera folders."""
    samples: list[SourceImage] = []
    missing_labels = 0

    for source_split in SOURCE_SPLITS:
        split_dir = atlas_root / source_split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"ATLAS split directory not found: {split_dir}")

        for camera_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if cameras is not None and camera_dir.name not in cameras:
                continue
            image_dir = camera_dir / "images"
            label_dir = camera_dir / "labels"
            if not image_dir.is_dir():
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            if not label_dir.is_dir():
                raise FileNotFoundError(f"Label directory not found: {label_dir}")

            images = sorted(
                p for p in image_dir.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
            if max_images_per_camera is not None:
                images = images[:max_images_per_camera]

            camera_count = 0
            for image_path in images:
                label_path = label_dir / f"{image_path.stem}.txt"
                if not label_path.is_file():
                    missing_labels += 1
                    continue
                samples.append(
                    SourceImage(
                        source_split=source_split,
                        camera=camera_dir.name,
                        image_path=image_path,
                        label_path=label_path,
                    )
                )
                camera_count += 1

            logger.info(
                "%s/%s: %d paired images",
                source_split,
                camera_dir.name,
                camera_count,
            )

    if missing_labels:
        logger.warning("Skipped %d image(s) with missing label files", missing_labels)
    return samples


def assign_output_splits(
    samples: list[SourceImage],
    val_fraction: float,
    seed: int,
) -> dict[str, list[SourceImage]]:
    """Map source samples to output train/val/test splits.

    ATLAS ships train/test only. For this project's validation workflow, a
    deterministic per-camera subset of ATLAS train can be carved out as val.
    """
    train_by_camera: dict[str, list[SourceImage]] = defaultdict(list)
    test_samples: list[SourceImage] = []
    for sample in samples:
        if sample.source_split == "train":
            train_by_camera[sample.camera].append(sample)
        elif sample.source_split == "test":
            test_samples.append(sample)

    rng = random.Random(seed)
    output: dict[str, list[SourceImage]] = {"train": [], "test": sorted(test_samples, key=_sample_sort_key)}
    if val_fraction > 0:
        output["val"] = []

    for camera, camera_samples in sorted(train_by_camera.items()):
        camera_samples = sorted(camera_samples, key=_sample_sort_key)
        if val_fraction <= 0 or len(camera_samples) < 2:
            output["train"].extend(camera_samples)
            continue

        shuffled = camera_samples[:]
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_fraction))
        val_count = max(1, min(len(shuffled) - 1, val_count))
        val_set = set(shuffled[:val_count])

        train_part = [sample for sample in camera_samples if sample not in val_set]
        val_part = [sample for sample in camera_samples if sample in val_set]
        output["train"].extend(train_part)
        output["val"].extend(val_part)
        logger.info(
            "train/%s split into %d train and %d val images",
            camera,
            len(train_part),
            len(val_part),
        )

    for split in output:
        output[split] = sorted(output[split], key=_sample_sort_key)
    return output


def _sample_sort_key(sample: SourceImage) -> tuple[str, str, str]:
    return sample.source_split, sample.camera, sample.image_path.name


def categories_for_mode(class_mode: str, atlas_classes: dict[int, str]) -> list[dict[str, Any]]:
    if class_mode == "single":
        return [{"id": 1, "name": "traffic_light", "supercategory": "none"}]
    if class_mode == "state4":
        return [
            {
                "id": STATE4_CATEGORY_IDS[class_name],
                "name": class_name,
                "supercategory": "traffic_light_state",
            }
            for class_name in STATE4_CATEGORIES
        ]
    if class_mode == "atlas":
        return [
            {
                "id": class_id + 1,
                "name": class_name,
                "supercategory": "traffic_light",
            }
            for class_id, class_name in atlas_classes.items()
        ]
    raise ValueError(f"Unsupported class mode: {class_mode}")


def atlas_state(class_name: str) -> str:
    if class_name == "off":
        return "off"
    if "red_yellow" in class_name:
        return "red_yellow"
    if "green" in class_name:
        return "green"
    if "yellow" in class_name:
        return "yellow"
    if "red" in class_name:
        return "red"
    return "unknown"


def classifier_state(class_name: str) -> str:
    state = atlas_state(class_name)
    if state == "red_yellow":
        return "yellow"
    return state


def category_id_for_mode(class_mode: str, atlas_class_id: int, class_name: str) -> int | None:
    if class_mode == "single":
        return 1
    if class_mode == "state4":
        return STATE4_CATEGORY_IDS.get(classifier_state(class_name))
    if class_mode == "atlas":
        return atlas_class_id + 1
    raise ValueError(f"Unsupported class mode: {class_mode}")


def lisa_compatible_tag(class_name: str) -> str:
    """Return a best-effort tag consumed by the existing classifier trainer."""
    state = classifier_state(class_name)
    if state == "green":
        if "left" in class_name:
            return "goLeft"
        if "straight" in class_name:
            return "goForward"
        return "go"
    if state == "red":
        if "left" in class_name:
            return "stopLeft"
        return "stop"
    if state == "yellow":
        if "left" in class_name:
            return "warningLeft"
        return "warning"
    if state == "off":
        return "off"
    return "unknown"


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def parse_label_file(
    sample: SourceImage,
    image_id: int,
    image_width: int,
    image_height: int,
    atlas_classes: dict[int, str],
    class_mode: str,
    ann_id_start: int,
) -> tuple[list[dict[str, Any]], Counter[str], int]:
    annotations: list[dict[str, Any]] = []
    skipped = Counter()
    ann_id = ann_id_start

    with open(sample.label_path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                skipped["bad_column_count"] += 1
                logger.warning("Bad label row %s:%d: %s", sample.label_path, line_no, line)
                continue

            try:
                atlas_class_id = int(parts[0])
                x_center, y_center, box_width, box_height = (float(v) for v in parts[1:])
            except ValueError:
                skipped["bad_number"] += 1
                logger.warning("Bad numeric label %s:%d: %s", sample.label_path, line_no, line)
                continue

            values = (x_center, y_center, box_width, box_height)
            if not all(math.isfinite(v) for v in values):
                skipped["non_finite"] += 1
                continue
            if atlas_class_id not in atlas_classes:
                skipped["unknown_class"] += 1
                continue
            if box_width <= 0 or box_height <= 0:
                skipped["non_positive_box"] += 1
                continue

            x1 = (x_center - box_width / 2.0) * image_width
            y1 = (y_center - box_height / 2.0) * image_height
            x2 = (x_center + box_width / 2.0) * image_width
            y2 = (y_center + box_height / 2.0) * image_height

            x1 = min(max(x1, 0.0), float(image_width))
            y1 = min(max(y1, 0.0), float(image_height))
            x2 = min(max(x2, 0.0), float(image_width))
            y2 = min(max(y2, 0.0), float(image_height))
            width = x2 - x1
            height = y2 - y1
            if width <= 0 or height <= 0:
                skipped["clipped_empty_box"] += 1
                continue

            class_name = atlas_classes[atlas_class_id]
            category_id = category_id_for_mode(class_mode, atlas_class_id, class_name)
            if category_id is None:
                skipped["unknown_state"] += 1
                continue
            ann_id += 1
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(width, 2),
                        round(height, 2),
                    ],
                    "area": round(width * height, 2),
                    "iscrowd": 0,
                    "attributes": {
                        "atlas_class_id": atlas_class_id,
                        "atlas_label": class_name,
                        "atlas_state": atlas_state(class_name),
                        "classifier_state": classifier_state(class_name),
                        "lisa_tag": lisa_compatible_tag(class_name),
                        "source_split": sample.source_split,
                        "source_camera": sample.camera,
                        "source_label": sample.label_path.name,
                    },
                }
            )

    return annotations, skipped, ann_id


def copy_or_link_image(src: Path, dst: Path, use_hardlinks: bool) -> None:
    if dst.exists():
        return
    if use_hardlinks:
        try:
            os.link(src.resolve(), dst)
            return
        except OSError as exc:
            logger.warning("Hard link failed for %s (%s); copying instead", src, exc)
    shutil.copy2(src, dst)


def output_file_name(sample: SourceImage, used_names: set[str]) -> str:
    name = sample.image_path.name
    if name not in used_names:
        return name

    prefixed = f"{sample.camera}__{name}"
    if prefixed not in used_names:
        return prefixed

    index = 2
    stem = sample.image_path.stem
    suffix = sample.image_path.suffix
    while True:
        candidate = f"{sample.camera}__{stem}_{index}{suffix}"
        if candidate not in used_names:
            return candidate
        index += 1


def convert_split(
    split: str,
    samples: list[SourceImage],
    output_dir: Path,
    atlas_classes: dict[int, str],
    class_mode: str,
    categories: list[dict[str, Any]],
    use_hardlinks: bool,
    skip_empty: bool,
) -> dict[str, Any]:
    image_dir = output_dir / split
    image_dir.mkdir(parents=True, exist_ok=True)

    coco_images: list[dict[str, Any]] = []
    coco_annotations: list[dict[str, Any]] = []
    used_names: set[str] = set()
    skipped = Counter()
    class_counts = Counter()
    state_counts = Counter()
    negative_images = 0
    image_id = 0
    ann_id = 0

    for sample in samples:
        width, height = read_image_size(sample.image_path)
        next_image_id = image_id + 1
        annotations, skipped_for_file, ann_id = parse_label_file(
            sample=sample,
            image_id=next_image_id,
            image_width=width,
            image_height=height,
            atlas_classes=atlas_classes,
            class_mode=class_mode,
            ann_id_start=ann_id,
        )
        skipped.update(skipped_for_file)

        if skip_empty and not annotations:
            continue

        file_name = output_file_name(sample, used_names)
        used_names.add(file_name)
        dst = image_dir / file_name
        copy_or_link_image(sample.image_path, dst, use_hardlinks)

        image_id = next_image_id
        coco_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
                "source": {
                    "dataset": "ATLAS",
                    "split": sample.source_split,
                    "camera": sample.camera,
                    "file_name": sample.image_path.name,
                },
            }
        )
        coco_annotations.extend(annotations)

        if not annotations:
            negative_images += 1
        for annotation in annotations:
            attrs = annotation["attributes"]
            class_counts[attrs["atlas_label"]] += 1
            state_counts[attrs["classifier_state"]] += 1

    payload = {
        "info": {
            "description": "ATLAS traffic light dataset converted to COCO",
            "source": "https://url.fzi.de/ATLAS",
            "class_mode": class_mode,
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }

    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    json_path = ann_dir / f"instances_{split}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    logger.info(
        "%s: %d images, %d annotations, %d negative images -> %s",
        split,
        len(coco_images),
        len(coco_annotations),
        negative_images,
        json_path,
    )
    if skipped:
        logger.warning("%s skipped labels: %s", split, dict(skipped))

    return {
        "split": split,
        "images": len(coco_images),
        "annotations": len(coco_annotations),
        "negative_images": negative_images,
        "json_path": str(json_path),
        "skipped": dict(skipped),
        "atlas_class_counts": dict(sorted(class_counts.items())),
        "classifier_state_counts": dict(sorted(state_counts.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ATLAS YOLO labels into this project's COCO dataset layout.",
    )
    parser.add_argument(
        "--atlas-root",
        type=Path,
        default=Path("data/ATLAS"),
        help="Path to the ATLAS dataset root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/coco_atlas"),
        help="Output COCO dataset directory",
    )
    parser.add_argument(
        "--class-mode",
        choices=["single", "state4", "atlas"],
        default="single",
        help=(
            "single collapses all ATLAS classes into category_id=1 traffic_light "
            "(matches --det-num-classes 1). state4 condenses ATLAS labels into "
            "off/red/yellow/green categories. atlas keeps all 25 ATLAS classes. "
            "Training infers the detector class count from the converted COCO categories."
        ),
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Fraction of ATLAS train images to reserve as val; set 0 to skip val",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for train/val split")
    parser.add_argument(
        "--hardlink",
        "--link",
        action="store_true",
        help="Hard-link images into the output tree instead of copying when possible",
    )
    parser.add_argument(
        "--skip-empty",
        action="store_true",
        help="Exclude images whose label file has no valid annotations",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        help=(
            "Camera folder names to include, e.g. front_medium. "
            "Defaults to all ATLAS cameras."
        ),
    )
    parser.add_argument(
        "--max-images-per-camera",
        type=int,
        default=None,
        help="Debug/smoke-test limit applied to each source split/camera",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.val_fraction < 1.0:
        raise ValueError("--val-fraction must be in the range [0, 1)")
    if args.max_images_per_camera is not None and args.max_images_per_camera <= 0:
        raise ValueError("--max-images-per-camera must be positive")
    cameras = set(args.cameras) if args.cameras else None

    atlas_root = args.atlas_root.resolve()
    output_dir = args.output_dir.resolve()
    classes_path = atlas_root / "ATLAS_classes.yaml"

    if not atlas_root.is_dir():
        raise FileNotFoundError(f"ATLAS root not found: {atlas_root}")
    atlas_classes = load_atlas_classes(classes_path)
    categories = categories_for_mode(args.class_mode, atlas_classes)

    logger.info("ATLAS root : %s", atlas_root)
    logger.info("Output dir : %s", output_dir)
    logger.info("Class mode : %s", args.class_mode)
    logger.info("Image mode : %s", "hardlink" if args.hardlink else "copy")
    logger.info("Cameras    : %s", ", ".join(sorted(cameras)) if cameras else "all")

    samples = discover_images(atlas_root, cameras, args.max_images_per_camera)
    if not samples:
        raise RuntimeError(f"No ATLAS image/label pairs found under {atlas_root}")

    split_samples = assign_output_splits(samples, args.val_fraction, args.seed)
    summaries = []
    for split in ("train", "val", "test"):
        samples_for_split = split_samples.get(split)
        if not samples_for_split:
            continue
        summaries.append(
            convert_split(
                split=split,
                samples=samples_for_split,
                output_dir=output_dir,
                atlas_classes=atlas_classes,
                class_mode=args.class_mode,
                categories=categories,
                use_hardlinks=args.hardlink,
                skip_empty=args.skip_empty,
            )
        )

    summary_path = output_dir / "conversion_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "atlas_root": str(atlas_root),
                "output_dir": str(output_dir),
                "class_mode": args.class_mode,
                "val_fraction": args.val_fraction,
                "seed": args.seed,
                "hardlink": args.hardlink,
                "skip_empty": args.skip_empty,
                "splits": summaries,
            },
            f,
            indent=2,
        )
    logger.info("Summary -> %s", summary_path)


if __name__ == "__main__":
    main()
