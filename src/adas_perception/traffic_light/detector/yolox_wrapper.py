from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np
import torch

from ..config import DetectorConfig
from ..schemas import Detection
from .base import BaseDetector

logger = logging.getLogger(__name__)

# YOLOX is installed via `pip install -e ".[models]"`.
# These imports are deferred to load_model() so the wrapper module can still be
# imported even if yolox is not yet installed (e.g. during packaging / tests).
_yolox_available: bool | None = None


def _check_yolox() -> None:
    global _yolox_available
    if _yolox_available is None:
        try:
            import yolox  # noqa: F401

            _yolox_available = True
        except ImportError:
            _yolox_available = False


class YoloxWrapper(BaseDetector):
    """YOLOX-based traffic light detector (PyTorch backend).

    Expects the ``yolox`` package to be installed
    (``pip install -e ".[models]"``).
    """

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        self.model: torch.nn.Module | None = None
        self.device: str = config.device

    def load_model(self, model_path: str, device: str) -> None:
        _check_yolox()
        if not _yolox_available:
            logger.warning(
                "yolox package not installed — running in stub mode. "
                "Install with:  pip install -e '.[models]'"
            )
            return

        from yolox.exp import get_exp

        # Build the YOLOX-s experiment as default; override via config later.
        exp = get_exp(None, "yolox-s")
        exp.num_classes = self.config.num_classes
        exp.test_size = tuple(self.config.input_size)

        self.model = exp.get_model()

        if Path(model_path).is_file():
            ckpt = torch.load(model_path, map_location=device, weights_only=True)
            if "model" in ckpt:
                self.model.load_state_dict(ckpt["model"])
            else:
                self.model.load_state_dict(ckpt)
            logger.info("Loaded YOLOX weights from %s", model_path)
        else:
            logger.warning(
                "Weight file %s not found — using random init", model_path
            )

        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict(self, image: torch.Tensor) -> List[Detection]:
        if self.model is None:
            return []

        with torch.no_grad():
            # YOLOX expects NCHW float tensor
            if image.dim() == 3:
                image = image.unsqueeze(0)
            outputs = self.model(image.to(self.device))

        return self._decode(outputs)

    # ------------------------------------------------------------------

    @staticmethod
    def _decode(outputs: torch.Tensor) -> List[Detection]:
        """Convert raw YOLOX output tensor to a list of Detection objects.

        YOLOX outputs shape: (batch, num_anchors, 5 + num_classes)
        Columns: cx, cy, w, h, obj_conf, cls_conf...
        """
        if outputs is None:
            return []

        # Take first batch element
        out = outputs[0]
        if out is None or len(out) == 0:
            return []

        out = out.cpu().numpy()
        detections: list[Detection] = []
        for row in out:
            cx, cy, w, h = row[:4]
            obj_conf = row[4]
            cls_scores = row[5:]
            cls_id = int(np.argmax(cls_scores))
            confidence = float(obj_conf * cls_scores[cls_id])

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            detections.append(
                Detection(
                    bbox=np.array([x1, y1, x2, y2], dtype=np.float32),
                    confidence=confidence,
                    class_id=cls_id,
                )
            )
        return detections
