from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..config import ClassifierConfig
    from ..schemas import LightState


class BaseClassifier(ABC):
    """Abstract interface for traffic light state classifiers.

    To add a new classifier (e.g. rule-based HSV):
      1. Subclass ``BaseClassifier`` and implement :meth:`classify`.
      2. Update ``state/__init__.py`` to expose the new implementation.
    """

    def __init__(self, config: ClassifierConfig) -> None:
        self.config = config

    @abstractmethod
    def load_model(self, model_path: str, device: str) -> None:
        """Load classifier weights."""

    @abstractmethod
    def classify(self, roi: np.ndarray) -> LightState:
        """Classify the traffic light state from a cropped ROI image.

        Parameters
        ----------
        roi:
            HWC uint8 BGR crop of the traffic light.

        Returns
        -------
        The predicted state.
        """
