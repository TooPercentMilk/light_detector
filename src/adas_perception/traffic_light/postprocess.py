from __future__ import annotations

from typing import List

import numpy as np
import torch
import torchvision

from .schemas import Detection


def postprocess(
    detections: List[Detection],
    nms_threshold: float = 0.45,
    confidence_threshold: float = 0.25,
) -> List[Detection]:
    """Filter and de-duplicate raw detector outputs.

    Steps:
      1. Discard detections below *confidence_threshold*.
      2. Apply class-agnostic NMS with *nms_threshold*.
      3. Return surviving detections.
    """
    if not detections:
        return []

    # 1. Confidence filter
    filtered = [d for d in detections if d.confidence >= confidence_threshold]
    if not filtered:
        return []

    # 2. NMS
    boxes = torch.tensor(
        np.array([d.bbox for d in filtered], dtype=np.float32)
    )
    scores = torch.tensor(
        np.array([d.confidence for d in filtered], dtype=np.float32)
    )

    keep = torchvision.ops.nms(boxes, scores, nms_threshold)

    return [filtered[i] for i in keep.tolist()]
