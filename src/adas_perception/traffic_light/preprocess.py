from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from .config import PreprocessConfig


class Preprocessor:
    """Resize, normalize, and convert an image to a model-ready tensor.

    Uses YOLOX-compatible preprocessing: letterbox resize that preserves
    aspect ratio (padding with value 114) and outputs float32 in [0, 255].
    """

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config
        self.input_h, self.input_w = config.input_size
        self.swap_rb = config.swap_rb

    def __call__(self, image: np.ndarray) -> tuple[torch.Tensor, float]:
        """Convert a HWC uint8 BGR image to a CHW float32 tensor.

        Returns
        -------
        (tensor, scale)
            *tensor* is CHW float32 in [0, 255] (YOLOX convention).
            *scale* is the letterbox scale factor — divide detection
            coordinates by this value to map back to the original image.
        """
        img = image.copy()

        if self.swap_rb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Letterbox resize — maintain aspect ratio, pad with 114
        ih, iw = img.shape[:2]
        r = min(self.input_w / iw, self.input_h / ih)
        new_w, new_h = int(iw * r), int(ih * r)
        resized = cv2.resize(img, (new_w, new_h))

        canvas = np.full((self.input_h, self.input_w, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        # float32 in [0, 255] — YOLOX convention (no /255 normalisation)
        img_f = canvas.astype(np.float32)

        img_f = img_f.transpose(2, 0, 1)  # HWC → CHW

        return torch.from_numpy(np.ascontiguousarray(img_f)), r
