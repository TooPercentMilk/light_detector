from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from .config import PreprocessConfig


class Preprocessor:
    """Resize, normalize, and convert an image to a model-ready tensor."""

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config
        self.input_h, self.input_w = config.input_size
        self.mean = np.array(config.mean, dtype=np.float32)
        self.std = np.array(config.std, dtype=np.float32)
        self.swap_rb = config.swap_rb

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """Convert a HWC uint8 BGR image to a CHW float32 tensor.

        Steps:
          1. Optionally swap B↔R channels.
          2. Resize to ``(input_h, input_w)``.
          3. Convert to float32 and scale to [0, 1].
          4. Normalize by mean / std.
          5. Transpose HWC → CHW.
          6. Return as ``torch.Tensor``.
        """
        img = image.copy()

        if self.swap_rb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.input_w, self.input_h))

        img = img.astype(np.float32) / 255.0

        img = (img - self.mean) / self.std

        img = img.transpose(2, 0, 1)  # HWC → CHW

        return torch.from_numpy(np.ascontiguousarray(img))
