"""Convert LISA Traffic Light Dataset to COCO format for YOLOX training.

Usage:
    python scripts/convert_lisa_to_coco.py --lisa-root data/LISA --output-dir data/coco_tl
    python scripts/convert_lisa_to_coco.py --lisa-root data/LISA --output-dir data/coco_tl --symlink
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VALID_TAGS = {"go", "goLeft", "goForward", "stop", "stopLeft", "warning", "warningLeft"}

# --------------------------------------------------------------------------- #
#  Split configuration                                                        #
# --------------------------------------------------------------------------- #


def build_split_config(lisa_root: Path) -> dict[str, list[dict]]:
    """Return a mapping of split name -> list of source descriptors.

    Each descriptor is a dict with keys:
        annotation_csv : Path to frameAnnotationsBOX.csv
        image_dir      : Path to the frames/ directory on disk
    """
    ann = lisa_root / "Annotations" / "Annotations"
    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}

    # --- Training: dayClip1-13 + nightClip1-5 ---
    for i in range(1, 14):
        clip = f"dayClip{i}"
        splits["train"].append(
            {
                "annotation_csv": ann / "dayTrain" / clip / "frameAnnotationsBOX.csv",
                "image_dir": lisa_root / "dayTrain" / "dayTrain" / clip / "frames",
            }
        )
    for i in range(1, 6):
        clip = f"nightClip{i}"
        splits["train"].append(
            {
                "annotation_csv": ann / "nightTrain" / clip / "frameAnnotationsBOX.csv",
                "image_dir": lisa_root / "nightTrain" / "nightTrain" / clip / "frames",
            }
        )

    # --- Validation: daySequence1 + nightSequence1 ---
    splits["val"].append(
        {
            "annotation_csv": ann / "daySequence1" / "frameAnnotationsBOX.csv",
            "image_dir": lisa_root / "daySequence1" / "daySequence1" / "frames",
        }
    )
    splits["val"].append(
        {
            "annotation_csv": ann / "nightSequence1" / "frameAnnotationsBOX.csv",
            "image_dir": lisa_root / "nightSequence1" / "nightSequence1" / "frames",
        }
    )

    # --- Test: daySequence2 + nightSequence2 ---
    splits["test"].append(
        {
            "annotation_csv": ann / "daySequence2" / "frameAnnotationsBOX.csv",
            "image_dir": lisa_root / "daySequence2" / "daySequence2" / "frames",
        }
    )
    splits["test"].append(
        {
            "annotation_csv": ann / "nightSequence2" / "frameAnnotationsBOX.csv",
            "image_dir": lisa_root / "nightSequence2" / "nightSequence2" / "frames",
        }
    )

    return splits


# --------------------------------------------------------------------------- #
#  CSV parsing                                                                #
# --------------------------------------------------------------------------- #


def parse_annotation_csv(csv_path: Path) -> dict[str, list[dict]]:
    """Parse a LISA frameAnnotationsBOX.csv and group annotations by image filename.

    Returns a dict mapping image basename (e.g. "dayClip1--00000.jpg") to a
    list of annotation dicts with keys: tag, x1, y1, x2, y2.
    """
    annotations: dict[str, list[dict]] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)  # skip header
        if "Filename" not in header[0]:
            logger.warning("Unexpected header in %s: %s", csv_path, header)

        for row in reader:
            if len(row) < 6:
                continue
            # Filename column contains a relative path like "dayTraining/dayClip1--00000.jpg"
            # We only need the basename.
            img_basename = Path(row[0]).name
            tag = row[1].strip()
            if tag not in VALID_TAGS:
                logger.warning("Unknown tag '%s' in %s, skipping", tag, csv_path)
                continue
            try:
                x1, y1, x2, y2 = float(row[2]), float(row[3]), float(row[4]), float(row[5])
            except ValueError:
                logger.warning("Bad bbox coords in %s: %s", csv_path, row[2:6])
                continue

            annotations.setdefault(img_basename, []).append(
                {"tag": tag, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            )
    return annotations


# --------------------------------------------------------------------------- #
#  Conversion                                                                 #
# --------------------------------------------------------------------------- #


def convert_split(
    split_name: str,
    sources: list[dict],
    output_dir: Path,
    use_symlink: bool,
    ann_id_start: int,
) -> int:
    """Convert one split (train/val/test) to COCO format.

    Returns the next available annotation ID (for global uniqueness).
    """
    images_out_dir = output_dir / split_name
    images_out_dir.mkdir(parents=True, exist_ok=True)

    coco_images: list[dict] = []
    coco_annotations: list[dict] = []
    image_id = 0
    ann_id = ann_id_start
    seen_filenames: set[str] = set()

    for source in sources:
        csv_path: Path = source["annotation_csv"]
        image_dir: Path = source["image_dir"]

        if not csv_path.exists():
            logger.error("Annotation CSV not found: %s", csv_path)
            continue
        if not image_dir.exists():
            logger.error("Image directory not found: %s", image_dir)
            continue

        # Parse annotations for this source
        annotations_by_image = parse_annotation_csv(csv_path)

        # Discover ALL images in the source directory (including unannotated)
        all_images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

        if not all_images:
            logger.warning("No images found in %s", image_dir)
            continue

        # Read resolution once from the first image (all frames in a clip share dimensions)
        with Image.open(all_images[0]) as im:
            width, height = im.size

        logger.info(
            "  %s: %d images (%dx%d), %d annotated frames, %d annotations",
            image_dir.name if image_dir.name != "frames" else image_dir.parent.name,
            len(all_images),
            width,
            height,
            len(annotations_by_image),
            sum(len(v) for v in annotations_by_image.values()),
        )

        for img_path in all_images:
            fname = img_path.name
            if fname in seen_filenames:
                logger.warning("Duplicate filename %s from %s, skipping", fname, image_dir)
                continue
            seen_filenames.add(fname)

            # Link or copy image to output directory
            dst = images_out_dir / fname
            if not dst.exists():
                if use_symlink:
                    # Use hard links (works without admin on Windows; same filesystem)
                    os.link(img_path.resolve(), dst)
                else:
                    shutil.copy2(img_path, dst)

            image_id += 1
            coco_images.append(
                {
                    "id": image_id,
                    "file_name": fname,
                    "width": width,
                    "height": height,
                }
            )

            # Add annotations for this image (may be empty for negative examples)
            for ann in annotations_by_image.get(fname, []):
                x1, y1, x2, y2 = ann["x1"], ann["y1"], ann["x2"], ann["y2"]
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    logger.warning("Invalid bbox (w=%s, h=%s) in %s, skipping", w, h, fname)
                    continue

                ann_id += 1
                coco_annotations.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                        "area": round(w * h, 2),
                        "iscrowd": 0,
                        "attributes": {"lisa_tag": ann["tag"]},
                    }
                )

    # Write COCO JSON
    coco_json = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": 1, "name": "traffic_light", "supercategory": "none"}],
    }

    ann_dir = output_dir / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    json_path = ann_dir / f"instances_{split_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(coco_json, f)

    logger.info(
        "  => %s: %d images, %d annotations -> %s",
        split_name,
        len(coco_images),
        len(coco_annotations),
        json_path,
    )
    return ann_id


# --------------------------------------------------------------------------- #
#  CLI                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LISA traffic light dataset to COCO format.")
    parser.add_argument("--lisa-root", type=Path, required=True, help="Path to LISA dataset root directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for COCO dataset")
    parser.add_argument("--symlink", action="store_true", help="Use hard links instead of copying images (saves disk space)")
    args = parser.parse_args()

    lisa_root: Path = args.lisa_root.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not lisa_root.exists():
        logger.error("LISA root not found: %s", lisa_root)
        return

    logger.info("LISA root : %s", lisa_root)
    logger.info("Output dir: %s", output_dir)
    logger.info("Mode      : %s", "hardlink" if args.symlink else "copy")

    split_config = build_split_config(lisa_root)
    ann_id = 0

    for split_name in ("train", "val", "test"):
        logger.info("Processing split: %s", split_name)
        ann_id = convert_split(split_name, split_config[split_name], output_dir, args.symlink, ann_id)

    logger.info("Done. Total annotations: %d", ann_id)


if __name__ == "__main__":
    main()
