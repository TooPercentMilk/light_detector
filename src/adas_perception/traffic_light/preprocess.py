from __future__ import annotations

from typing import TYPE_CHECKING

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

        Steps (to be filled in):
          1. Optionally swap B↔R channels.
          2. Resize to ``(input_h, input_w)``.
          3. Convert to float32 and scale to [0, 1].
          4. Normalize by mean / std.
          5. Transpose HWC → CHW.
          6. Return as ``torch.Tensor``.
        """
        # TODO: implement full preprocessing pipeline
        raise NotImplementedError("Preprocessor.__call__ not yet implemented")
