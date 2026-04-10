from __future__ import annotations

from typing import List

from .schemas import Detection


def postprocess(
    detections: List[Detection],
    nms_threshold: float = 0.45,
    confidence_threshold: float = 0.25,
) -> List[Detection]:
    """Filter and de-duplicate raw detector outputs.

    Steps (to be filled in):
      1. Discard detections below *confidence_threshold*.
      2. Apply class-agnostic NMS with *nms_threshold*.
      3. Return surviving detections.
    """
    # TODO: implement NMS + confidence filtering
    return detections
