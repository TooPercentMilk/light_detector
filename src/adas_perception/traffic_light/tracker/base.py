from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ..config import TrackerConfig
    from ..schemas import Detection, TrackedObject


class BaseTracker(ABC):
    """Abstract interface that every tracker implementation must satisfy.

    To add a new tracker:
      1. Subclass ``BaseTracker`` and implement all abstract methods.
      2. Register the subclass in ``tracker/__init__.py`` via TRACKER_REGISTRY.
    """

    def __init__(self, config: TrackerConfig) -> None:
        self.config = config

    @abstractmethod
    def update(
        self,
        detections: List[Detection],
        frame_id: int,
        image_shape: tuple[int, int] | None = None,
    ) -> List[TrackedObject]:
        """Associate *detections* with existing tracks and return updated list."""

    @abstractmethod
    def reset(self) -> None:
        """Clear all internal track state."""
