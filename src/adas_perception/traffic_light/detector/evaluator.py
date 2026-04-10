from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def evaluate(
    model: Any,
    dataset_path: str,
    iou_thresholds: list[float] | None = None,
) -> Dict[str, float]:
    """Evaluate a detector on a COCO-format dataset.

    Returns
    -------
    Dict with metrics such as ``mAP``, ``mAP_50``, ``mAP_75``.
    """
    # TODO: run inference over val set, compute COCO metrics
    raise NotImplementedError("evaluate not yet implemented")
