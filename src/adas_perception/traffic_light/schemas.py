from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class LightState(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"
    OFF = "off"
    UNKNOWN = "unknown"


@dataclass
class Detection:
    """A single object detection result."""

    bbox: np.ndarray  # shape (4,) — x1, y1, x2, y2
    confidence: float
    class_id: int


@dataclass
class TrackedObject:
    """A detection associated with a persistent track."""

    track_id: int
    bbox: np.ndarray  # shape (4,) — x1, y1, x2, y2
    confidence: float
    class_id: int
    age: int = 0


@dataclass
class TrafficLight:
    """Final per-frame output for a tracked traffic light."""

    track_id: int
    bbox: np.ndarray  # shape (4,) — x1, y1, x2, y2
    state: LightState
    detection_confidence: float
    state_confidence: float
