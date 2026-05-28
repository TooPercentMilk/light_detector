from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

from ..config import TrackerConfig
from ..schemas import Detection, TrackedObject
from .base import BaseTracker

logger = logging.getLogger(__name__)

# BYTETracker is available two ways (in priority order):
#   1. pip-installed YOLOX (preferred) — includes yolox.tracker.byte_tracker
#   2. external/ByteTrack cloned via setup_external.py (fallback)
#
# We always try the pip-installed yolox first.  The external/ clone is only
# added to sys.path as a fallback, and is appended (not prepended) so it
# can never shadow a properly installed yolox package.
_bytetrack_available: bool | None = None
_EXTERNAL_DIR = Path(__file__).resolve().parents[4] / "external" / "ByteTrack"


def _ensure_bytetrack_on_path() -> None:
    global _bytetrack_available
    if _bytetrack_available is not None:
        return

    # Attempt 1: pip-installed yolox already on sys.path
    try:
        from yolox.tracker.byte_tracker import BYTETracker  # noqa: F401

        _bytetrack_available = True
        return
    except ImportError:
        pass

    # Attempt 2: external/ByteTrack clone (appended to avoid shadowing yolox pkg)
    if _EXTERNAL_DIR.is_dir():
        path_str = str(_EXTERNAL_DIR)
        if path_str not in sys.path:
            sys.path.append(path_str)
        try:
            from yolox.tracker.byte_tracker import BYTETracker  # noqa: F401

            _bytetrack_available = True
            return
        except ImportError:
            pass

    _bytetrack_available = False
    logger.warning(
        "BYTETracker not found. Either install YOLOX "
        "(`pip install -e '.[models]'`) or run `python setup_external.py`."
    )


class _TrackerArgs:
    """Mimics the argparse namespace that BYTETracker expects."""

    def __init__(self, config: TrackerConfig) -> None:
        self.track_thresh = config.track_thresh
        self.track_buffer = config.track_buffer
        self.match_thresh = config.match_thresh
        self.mot20 = False


class ByteTrackWrapper(BaseTracker):
    """ByteTrack multi-object tracker.

    Requires ``external/ByteTrack`` to be cloned
    (``python setup_external.py``).
    """

    def __init__(self, config: TrackerConfig) -> None:
        super().__init__(config)
        self._tracker = None
        self._frame_id: int = 0

        _ensure_bytetrack_on_path()
        if _bytetrack_available:
            from yolox.tracker.byte_tracker import BYTETracker

            args = _TrackerArgs(config)
            self._tracker = BYTETracker(args, frame_rate=config.frame_rate)

    def update(
        self,
        detections: List[Detection],
        frame_id: int,
        image_shape: tuple[int, int] | None = None,
    ) -> List[TrackedObject]:
        self._frame_id = frame_id

        if self._tracker is None:
            return []

        # BYTETracker expects an (N, 5) numpy array: x1, y1, x2, y2, score
        if detections:
            det_array = np.array(
                [[*d.bbox, d.confidence] for d in detections], dtype=np.float32
            )
        else:
            det_array = np.empty((0, 5), dtype=np.float32)

        if image_shape is not None:
            img_h, img_w = int(image_shape[0]), int(image_shape[1])
        else:
            img_h, img_w = self.config.track_buffer, self.config.track_buffer

        # BYTETracker.update signature: (output_results, img_info, img_size)
        online_targets = self._tracker.update(
            det_array,
            [img_h, img_w],
            [img_h, img_w],
        )

        tracked: list[TrackedObject] = []
        for t in online_targets:
            tlwh = t.tlwh
            x1, y1, w, h = tlwh
            tracked.append(
                TrackedObject(
                    track_id=int(t.track_id),
                    bbox=np.array(
                        [x1, y1, x1 + w, y1 + h], dtype=np.float32
                    ),
                    confidence=float(t.score),
                    class_id=0,
                    age=int(t.tracklet_len),
                )
            )
        return tracked

    def reset(self) -> None:
        self._frame_id = 0
        if _bytetrack_available:
            from yolox.tracker.byte_tracker import BYTETracker

            args = _TrackerArgs(self.config)
            self._tracker = BYTETracker(args, frame_rate=self.config.frame_rate)
