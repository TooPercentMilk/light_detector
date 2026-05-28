from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import yaml


DEFAULT_DETECTOR_IMAGE_SIZE = 960


@dataclass
class DetectorConfig:
    type: str = "yolox"
    model_path: str = "weights/yolox_tl.pth"
    device: str = "cuda"
    input_size: List[int] = field(
        default_factory=lambda: [
            DEFAULT_DETECTOR_IMAGE_SIZE,
            DEFAULT_DETECTOR_IMAGE_SIZE,
        ]
    )
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    num_classes: int = 1
    exp_name: str = "yolox-m"


@dataclass
class TrackerConfig:
    type: str = "bytetrack"
    track_thresh: float = 0.5
    track_buffer: int = 30
    match_thresh: float = 0.8
    frame_rate: int = 30


@dataclass
class ClassifierConfig:
    type: str = "cnn"
    model_path: str = "weights/tl_state_classifier.pth"
    device: str = "cuda"
    input_size: List[int] = field(default_factory=lambda: [32, 64])
    classes: List[str] = field(
        default_factory=lambda: ["red", "yellow", "green", "off"]
    )


@dataclass
class TemporalSmootherConfig:
    window_size: int = 5
    min_consensus: int = 3


@dataclass
class PreprocessConfig:
    input_size: List[int] = field(
        default_factory=lambda: [
            DEFAULT_DETECTOR_IMAGE_SIZE,
            DEFAULT_DETECTOR_IMAGE_SIZE,
        ]
    )
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    swap_rb: bool = True
    top_crop_only: bool = False
    top_crop_fraction: float = 0.5
    top_third_only: bool = False


@dataclass
class PipelineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    temporal_smoother: TemporalSmootherConfig = field(
        default_factory=TemporalSmootherConfig
    )
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


def normalize_detector_input_size(input_size: Sequence[int]) -> tuple[int, int]:
    """Validate and normalize detector/preprocessor input size as ``(H, W)``."""
    if len(input_size) != 2:
        raise ValueError("input_size must contain exactly two values: H W")
    h, w = (int(input_size[0]), int(input_size[1]))
    if h <= 0 or w <= 0:
        raise ValueError("input_size values must be positive")
    return h, w


def detector_input_size_from_args(
    image_size: int | None = DEFAULT_DETECTOR_IMAGE_SIZE,
    input_size: Sequence[int] | None = None,
) -> tuple[int, int]:
    """Resolve CLI image-size arguments to a detector ``(H, W)`` tuple."""
    if input_size is not None:
        return normalize_detector_input_size(input_size)
    size = DEFAULT_DETECTOR_IMAGE_SIZE if image_size is None else int(image_size)
    if size <= 0:
        raise ValueError("image_size must be positive")
    return size, size


def apply_detector_input_size(
    config: PipelineConfig,
    input_size: Sequence[int],
) -> tuple[int, int]:
    """Apply one detector input size to every detector pre-processing path."""
    h, w = normalize_detector_input_size(input_size)
    config.detector.input_size = [h, w]
    config.preprocess.input_size = [h, w]
    return h, w


def load_config(path: str | Path) -> PipelineConfig:
    """Load a YAML config file and return a typed PipelineConfig."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    return PipelineConfig(
        detector=DetectorConfig(**raw.get("detector", {})),
        tracker=TrackerConfig(**raw.get("tracker", {})),
        classifier=ClassifierConfig(**raw.get("classifier", {})),
        temporal_smoother=TemporalSmootherConfig(
            **raw.get("temporal_smoother", {})
        ),
        preprocess=PreprocessConfig(**raw.get("preprocess", {})),
    )
