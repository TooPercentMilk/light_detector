from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import torch

if TYPE_CHECKING:
    from ..config import DetectorConfig
    from ..schemas import Detection


class BaseDetector(ABC):
    """Abstract interface that every detector implementation must satisfy.

    To add a new detector:
      1. Subclass ``BaseDetector`` and implement all abstract methods.
      2. Register the subclass in ``detector/__init__.py`` via DETECTOR_REGISTRY.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config

    @abstractmethod
    def load_model(self, model_path: str, device: str) -> None:
        """Load model weights from *model_path* onto *device*."""

    @abstractmethod
    def predict(self, image: torch.Tensor) -> List[Detection]:
        """Run inference on a pre-processed image tensor.

        Parameters
        ----------
        image:
            CHW float tensor, already normalized.

        Returns
        -------
        List of raw detections (before post-processing).
        """
