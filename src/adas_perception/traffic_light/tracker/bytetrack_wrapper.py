from __future__ import annotations

import logging
from typing import List

from ..config import TrackerConfig
from ..schemas import Detection, TrackedObject
from .base import BaseTracker

logger = logging.getLogger(__name__)


class ByteTrackWrapper(BaseTracker):
    """ByteTrack multi-object tracker."""

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._tracks: list = []
        self._frame_id: int = 0
        # TODO: initialise internal ByteTrack state (Kalman filters, etc.)

    def update(
        self, detections: List[Detection], frame_id: int
    ) -> List[TrackedObject]:
        # TODO: two-stage association (high-score + low-score), Kalman predict/update
        self._frame_id = frame_id
        return []

    def reset(self) -> None:
        self._tracks.clear()
        self._frame_id = 0
