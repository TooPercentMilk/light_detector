"""Visualize the traffic-light pipeline on validation images.

Runs the full pipeline (detect → track → classify) on a set of images
and saves annotated frames with bounding boxes and state labels.

Usage::

    python scripts/visualize.py --config configs/val_best.yaml \
        --image-dir data/coco_tl/val --output-dir runs/viz \
        --clip daySequence1 --max-frames 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.adas_perception.traffic_light.config import (
    DEFAULT_DETECTOR_IMAGE_SIZE,
    apply_detector_input_size,
    detector_input_size_from_args,
    load_config,
)
from src.adas_perception.traffic_light.node import TrafficLightNode
from src.adas_perception.traffic_light.preprocess import top_fraction_height
from src.adas_perception.traffic_light.viz.overlays import draw_traffic_lights

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


def _draw_crop_boundary(frame, fraction: float) -> None:
    y = top_fraction_height(frame.shape[0], fraction)
    if 0 < y < frame.shape[0]:
        cv2.line(frame, (0, y), (frame.shape[1] - 1, y), (0, 255, 0), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize traffic-light pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/val_best.yaml",
        help="Path to pipeline config YAML",
    )
    parser.add_argument(
        "--image-dir", type=str, default="data/coco_tl/val",
        help="Directory containing input images",
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/viz",
        help="Directory to save annotated images",
    )
    parser.add_argument(
        "--clip", type=str, default=None,
        help="Filter images to a specific clip prefix (e.g. 'daySequence1')",
    )
    parser.add_argument(
        "--max-frames", type=int, default=200,
        help="Maximum number of frames to process (default: 200)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override device (e.g. 'cpu' or 'cuda')",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_DETECTOR_IMAGE_SIZE,
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
    parser.add_argument(
        "--top-half-only",
        "--top-40-only",
        "--top-third-only",
        dest="top_crop_only",
        action="store_true",
        help="Run detector inference on only the configured top crop of each frame",
    )
    parser.add_argument(
        "--top-crop-fraction",
        type=float,
        default=None,
        help="Image-height fraction kept when top-crop inference is enabled",
    )
    parser.add_argument(
        "--draw-crop-boundary",
        action="store_true",
        help="Draw the configured top-crop cutoff line on annotated frames",
    )
    args = parser.parse_args()
    if args.top_crop_fraction is not None and not 0.0 < args.top_crop_fraction <= 1.0:
        parser.error("--top-crop-fraction must be in the range (0, 1]")

    # Load config and optionally override device
    config = load_config(args.config)
    try:
        input_size = detector_input_size_from_args(args.image_size, args.input_size)
    except ValueError as exc:
        parser.error(str(exc))
    apply_detector_input_size(config, input_size)
    if args.top_crop_fraction is not None:
        config.preprocess.top_crop_fraction = args.top_crop_fraction
    if args.top_crop_only:
        config.preprocess.top_crop_only = True
        config.preprocess.top_third_only = False
    if args.device:
        config.detector.device = args.device
        config.classifier.device = args.device
    logger.info(
        "Top crop: enabled=%s | fraction=%.3f",
        config.preprocess.top_crop_only or config.preprocess.top_third_only,
        config.preprocess.top_crop_fraction,
    )

    # Build pipeline
    node = TrafficLightNode(config)

    # Gather and filter images
    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        logger.error("Image directory not found: %s", image_dir)
        sys.exit(1)

    files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if args.clip:
        files = [f for f in files if f.name.startswith(args.clip)]

    if not files:
        logger.error("No images found (clip=%s)", args.clip)
        sys.exit(1)

    files = files[: args.max_frames]
    logger.info("Processing %d frames (clip=%s)", len(files), args.clip or "all")

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run pipeline and save annotated frames
    total_detections = 0
    for frame_id, filepath in enumerate(files):
        image = cv2.imread(str(filepath))
        if image is None:
            logger.warning("Could not read %s, skipping", filepath.name)
            continue

        results = node.process_frame(image, frame_id)
        total_detections += len(results)

        annotated = draw_traffic_lights(image, results)
        if args.draw_crop_boundary:
            _draw_crop_boundary(annotated, config.preprocess.top_crop_fraction)

        out_path = output_dir / filepath.name
        cv2.imwrite(str(out_path), annotated)

        if (frame_id + 1) % 50 == 0:
            logger.info(
                "  [%d/%d] %d detections so far",
                frame_id + 1, len(files), total_detections,
            )

    logger.info(
        "Done. %d frames processed, %d total detections. Output: %s",
        len(files), total_detections, output_dir,
    )


if __name__ == "__main__":
    main()
