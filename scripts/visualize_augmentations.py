"""Visualize detector data augmentations on sample images.

Draws bounding boxes on augmented images so you can visually verify that
mosaic, scale jitter, crop/translate, HSV jitter, and flip work correctly.

Usage::

    # Show 8 augmented samples (default)
    python scripts/visualize_augmentations.py --dataset data/coco_tl

    # Save to disk instead of displaying
    python scripts/visualize_augmentations.py --dataset data/coco_tl \
        --output-dir runs/aug_preview --num-samples 16

    # Disable specific augmentations
    python scripts/visualize_augmentations.py --dataset data/coco_tl \
        --no-mosaic --no-scale-jitter
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from adas_perception.traffic_light.detector.trainer import _COCODetectionDataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BOX_COLOR = (0, 255, 0)  # green BGR
BOX_THICKNESS = 2


def draw_bboxes(image: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Draw bounding boxes from YOLOX label format ``[cls, cx, cy, w, h]``."""
    vis = image.copy()
    for row in labels:
        cls, cx, cy, w, h = row
        if w <= 0 or h <= 0:
            continue
        x1 = int(cx - w / 2)
        y1 = int(cy - h / 2)
        x2 = int(cx + w / 2)
        y2 = int(cy + h / 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, BOX_THICKNESS)
        cv2.putText(
            vis, f"{int(cls)}", (x1, max(y1 - 4, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, BOX_COLOR, 1,
        )
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize detector augmentations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="COCO-format dataset dir")
    parser.add_argument("--split", default="train")
    parser.add_argument("--ann-file", default="instances_train.json")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Save images here instead of displaying")
    parser.add_argument("--input-size", nargs=2, type=int, default=[1280, 1280],
                        metavar=("H", "W"))
    parser.add_argument(
        "--top-half-only",
        "--top-40-only",
        "--top-third-only",
        dest="top_crop_only",
        action="store_true",
        help="Visualize samples using only the top half of each source image",
    )
    parser.add_argument(
        "--top-crop-fraction",
        type=float,
        default=0.5,
        help="Image-height fraction kept when --top-half-only is enabled",
    )

    # Augmentation toggles
    parser.add_argument("--no-mosaic", action="store_true")
    parser.add_argument("--no-scale-jitter", action="store_true")
    parser.add_argument("--no-hsv", action="store_true")
    parser.add_argument("--no-flip", action="store_true")

    # Augmentation parameters
    parser.add_argument("--mosaic-prob", type=float, default=1.0)
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=1.5)
    parser.add_argument("--hsv-hue", type=float, default=0.015)
    parser.add_argument("--hsv-sat", type=float, default=0.7)
    parser.add_argument("--hsv-val", type=float, default=0.4)
    parser.add_argument("--flip-prob", type=float, default=0.5)

    args = parser.parse_args()
    if not 0.0 < args.top_crop_fraction <= 1.0:
        parser.error("--top-crop-fraction must be in the range (0, 1]")

    dataset = _COCODetectionDataset(
        data_dir=args.dataset,
        json_file=args.ann_file,
        split=args.split,
        input_size=tuple(args.input_size),
        augment=True,
        mosaic_prob=0.0 if args.no_mosaic else args.mosaic_prob,
        scale_jitter_range=(1.0, 1.0) if args.no_scale_jitter else (args.scale_min, args.scale_max),
        hsv_hue=0.0 if args.no_hsv else args.hsv_hue,
        hsv_sat=0.0 if args.no_hsv else args.hsv_sat,
        hsv_val=0.0 if args.no_hsv else args.hsv_val,
        flip_prob=0.0 if args.no_flip else args.flip_prob,
        top_crop_only=args.top_crop_only,
        top_crop_fraction=args.top_crop_fraction,
    )

    logger.info("Dataset: %d images", len(dataset))

    out_dir = None
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))

    for i, idx in enumerate(indices):
        img_t, labels = dataset[idx]

        # CHW RGB float32 [0,255] -> HWC BGR uint8
        img_np = img_t.numpy().transpose(1, 2, 0).astype(np.uint8)
        img_bgr = img_np[:, :, ::-1].copy()

        vis = draw_bboxes(img_bgr, labels.numpy())

        if out_dir:
            path = out_dir / f"aug_{i:03d}.jpg"
            cv2.imwrite(str(path), vis)
            logger.info("Saved %s", path)
        else:
            cv2.imshow(f"Augmented sample {i}", vis)

    if not out_dir:
        logger.info("Press any key to close windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        logger.info("Done — %d samples saved to %s", len(indices), out_dir)


if __name__ == "__main__":
    main()
