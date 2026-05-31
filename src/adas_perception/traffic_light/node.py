"""Traffic-light detection pipeline orchestrator.

Usage::

    python -m adas_perception.traffic_light.node \
        --config configs/default.yaml \
        --image-dir data/frames/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

from .config import (
    DEFAULT_DETECTOR_IMAGE_SIZE,
    PipelineConfig,
    apply_detector_input_size,
    detector_input_size_from_args,
    load_config,
)
from .detector import build_detector
from .fusion.map_gate import MapGate
from .postprocess import postprocess
from .preprocess import Preprocessor, top_fraction_height
from .schemas import LightState, TrafficLight
from .state.classifier import StateClassifier
from .state.roi_refiner import RoiRefiner
from .state.temporal_smoother import TemporalSmoother
from .tracker import build_tracker
from .viz.overlays import draw_traffic_lights

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}


class TrafficLightNode:
    """End-to-end pipeline: load frame → detect → track → classify → output."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        # ---- build swappable components via registries ----
        self.detector = build_detector(config.detector)
        self.tracker = build_tracker(config.tracker)

        # ---- fixed components (extend with ABCs if needed) ----
        self.preprocessor = Preprocessor(config.preprocess)
        self.classifier = StateClassifier(config.classifier)
        self.classifier.load_model(
            config.classifier.model_path, config.classifier.device
        )
        self.smoother = TemporalSmoother(config.temporal_smoother)
        self.roi_refiner = RoiRefiner()
        self.map_gate = MapGate()

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def process_frame(
        self,
        image: np.ndarray,
        frame_id: int,
    ) -> List[TrafficLight]:
        """Run the full pipeline on a single BGR image.

        Returns a list of :class:`TrafficLight` results for this frame.
        """
        # 1. Pre-process
        tensor, scale = self.preprocessor(image)

        # 2. Detect
        raw_detections = self.detector.predict(tensor)

        # 3. Post-process (NMS + confidence filter)
        detections = postprocess(
            raw_detections,
            nms_threshold=self.config.detector.nms_threshold,
            confidence_threshold=self.config.detector.confidence_threshold,
        )

        # 3b. Rescale boxes from letterbox space → original image coords
        top_crop_enabled = (
            self.config.preprocess.top_crop_only
            or self.config.preprocess.top_third_only
        )
        max_y = (
            top_fraction_height(
                image.shape[0],
                self.config.preprocess.top_crop_fraction,
            )
            if top_crop_enabled
            else image.shape[0]
        )
        clipped_detections = []
        for det in detections:
            det.bbox = det.bbox / scale
            det.bbox[0::2] = np.clip(det.bbox[0::2], 0, image.shape[1])
            det.bbox[1::2] = np.clip(det.bbox[1::2], 0, max_y)
            if det.bbox[2] > det.bbox[0] and det.bbox[3] > det.bbox[1]:
                clipped_detections.append(det)
        detections = clipped_detections

        # 4. Track
        tracked = self.tracker.update(detections, frame_id, (max_y, image.shape[1]))
        clipped_tracked = []
        for obj in tracked:
            obj.bbox[0::2] = np.clip(obj.bbox[0::2], 0, image.shape[1])
            obj.bbox[1::2] = np.clip(obj.bbox[1::2], 0, max_y)
            if obj.bbox[2] > obj.bbox[0] and obj.bbox[3] > obj.bbox[1]:
                clipped_tracked.append(obj)
        tracked = clipped_tracked

        # 5. Classify state per tracked object
        results: List[TrafficLight] = []
        for obj in tracked:
            roi = self.roi_refiner.refine(image, obj.bbox)
            raw_state, state_conf = self.classifier.classify(roi)
            smoothed_state = self.smoother.update(obj.track_id, raw_state)

            results.append(
                TrafficLight(
                    track_id=obj.track_id,
                    bbox=obj.bbox,
                    state=smoothed_state,
                    detection_confidence=obj.confidence,
                    state_confidence=state_conf,
                )
            )

        # 6. Optional map-based gating
        results = self.map_gate.filter(results)

        return results

    # ------------------------------------------------------------------
    # Batch runner
    # ------------------------------------------------------------------

    def run(self, image_dir: str, visualize: bool = False) -> None:
        """Iterate over sorted image files in *image_dir* and process each."""
        path = Path(image_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        files = sorted(
            p for p in path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        logger.info("Found %d images in %s", len(files), image_dir)

        for frame_id, filepath in enumerate(files):
            image = cv2.imread(str(filepath))
            if image is None:
                logger.warning("Could not read %s — skipping", filepath)
                continue

            lights = self.process_frame(image, frame_id)
            logger.info(
                "Frame %04d  |  %d lights detected", frame_id, len(lights)
            )

            if visualize:
                vis = draw_traffic_lights(image, lights)
                cv2.imshow("Traffic Lights", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if visualize:
            cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run traffic light detection on a directory of images."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to pipeline YAML config.",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Directory containing input image frames.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show annotated frames in a window.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_DETECTOR_IMAGE_SIZE,
        help="Square detector/preprocessor image size.",
    )
    parser.add_argument(
        "--input-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Detector/preprocessor input size; overrides --image-size.",
    )
    parser.add_argument(
        "--top-half-only",
        "--top-40-only",
        "--top-third-only",
        dest="top_crop_only",
        action="store_true",
        help="Run detector inference on only the top half of each frame.",
    )
    parser.add_argument(
        "--top-crop-fraction",
        type=float,
        default=0.5,
        help="Image-height fraction kept when --top-half-only is enabled.",
    )
    args = parser.parse_args(argv)
    if not 0.0 < args.top_crop_fraction <= 1.0:
        parser.error("--top-crop-fraction must be in the range (0, 1]")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = load_config(args.config)
    try:
        input_size = detector_input_size_from_args(args.image_size, args.input_size)
    except ValueError as exc:
        parser.error(str(exc))
    apply_detector_input_size(config, input_size)
    if args.top_crop_only:
        config.preprocess.top_crop_only = True
        config.preprocess.top_crop_fraction = args.top_crop_fraction
    node = TrafficLightNode(config)
    node.run(args.image_dir, visualize=args.visualize)


if __name__ == "__main__":
    main()
