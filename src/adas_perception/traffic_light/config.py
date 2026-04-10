from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class DetectorConfig:
    type: str = "yolox"
    model_path: str = "weights/yolox_tl.pth"
    device: str = "cuda"
    input_size: List[int] = field(default_factory=lambda: [640, 640])
    confidence_threshold: float = 0.25
    nms_threshold: float = 0.45
    num_classes: int = 1


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
    input_size: List[int] = field(default_factory=lambda: [640, 640])
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    swap_rb: bool = True


@dataclass
class PipelineConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    temporal_smoother: TemporalSmootherConfig = field(
        default_factory=TemporalSmootherConfig
    )
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)


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
