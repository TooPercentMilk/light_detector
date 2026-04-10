from __future__ import annotations

import logging
from typing import List

import torch

from ..config import DetectorConfig
from ..schemas import Detection
from .base import BaseDetector

logger = logging.getLogger(__name__)


class YoloxWrapper(BaseDetector):
    """YOLOX-based traffic light detector (PyTorch backend)."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        self.model = None

    def load_model(self, model_path: str, device: str) -> None:
        # TODO: build YOLOX model, load checkpoint, move to device
        logger.info(
            "YoloxWrapper.load_model — model_path=%s device=%s (stub)",
            model_path,
            device,
        )

    def predict(self, image: torch.Tensor) -> List[Detection]:
        # TODO: run forward pass, decode predictions
        return []
