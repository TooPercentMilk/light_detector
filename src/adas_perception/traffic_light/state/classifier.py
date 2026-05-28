from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models

from ..config import ClassifierConfig
from ..schemas import LightState
from .base import BaseClassifier

logger = logging.getLogger(__name__)

# Map class index → LightState (must match config.classes order)
_INDEX_TO_STATE = {
    0: LightState.RED,
    1: LightState.YELLOW,
    2: LightState.GREEN,
    3: LightState.OFF,
}


class StateClassifier(BaseClassifier):
    """CNN-based traffic light state classifier using MobileNetV3-Small."""

    def __init__(self, config: ClassifierConfig) -> None:
        super().__init__(config)
        self.model: torch.nn.Module | None = None
        self.device: torch.device = torch.device("cpu")
        self.num_classes = len(config.classes)
        # config.input_size is [W, H]
        self.input_w, self.input_h = config.input_size

    def _build_model(self) -> torch.nn.Module:
        """Build a MobileNetV3-Small and replace the classifier head."""
        model = models.mobilenet_v3_small(weights=None)
        # Replace final classifier: default is Linear(1024, 1000)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, self.num_classes)
        return model

    def load_model(self, model_path: str, device: str) -> None:
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.model = self._build_model()

        weights_path = Path(model_path)
        if weights_path.is_file():
            try:
                ckpt = torch.load(weights_path, map_location=self.device, weights_only=True)
            except TypeError:
                ckpt = torch.load(weights_path, map_location=self.device)
            state_dict = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
            self.model.load_state_dict(state_dict)
            logger.info("StateClassifier: loaded weights from %s", model_path)
        else:
            logger.warning(
                "StateClassifier: weights not found at %s — using random init",
                model_path,
            )

        self.model.to(self.device)
        self.model.eval()

    def classify(self, roi: np.ndarray) -> tuple[LightState, float]:
        if self.model is None:
            return LightState.UNKNOWN, 0.0

        # Resize ROI to expected (H, W)
        resized = cv2.resize(roi, (self.input_w, self.input_h))
        # BGR → RGB, HWC → CHW, uint8 → float32 [0,1]
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        # ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        # Add batch dim and move to device
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            confidence = float(probs.max(dim=1).values.item())
            class_idx = int(probs.argmax(dim=1).item())

        return _INDEX_TO_STATE.get(class_idx, LightState.UNKNOWN), confidence
