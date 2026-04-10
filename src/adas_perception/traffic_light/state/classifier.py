from __future__ import annotations

import logging

import numpy as np

from ..config import ClassifierConfig
from ..schemas import LightState
from .base import BaseClassifier

logger = logging.getLogger(__name__)


class StateClassifier(BaseClassifier):
    """CNN-based traffic light state classifier."""

    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__(config)
        self.model = None

    def load_model(self, model_path: str, device: str) -> None:
        # TODO: build classification model, load weights, move to device
        logger.info(
            "StateClassifier.load_model — model_path=%s device=%s (stub)",
            model_path,
            device,
        )

    def classify(self, roi: np.ndarray) -> LightState:
        # TODO: preprocess ROI, run forward pass, argmax → LightState
        return LightState.UNKNOWN
