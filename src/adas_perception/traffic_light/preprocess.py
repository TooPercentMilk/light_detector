from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

if TYPE_CHECKING:
    from .config import PreprocessConfig


DEFAULT_TOP_CROP_FRACTION = 0.5


def top_fraction_height(
    image_height: int,
    fraction: float = DEFAULT_TOP_CROP_FRACTION,
) -> int:
    """Return the number of source rows kept by a top-fraction crop."""
    if image_height <= 0:
        raise ValueError("image_height must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in the range (0, 1]")
    return max(1, int(np.ceil(image_height * fraction)))


def crop_top_fraction(
    image: np.ndarray,
    fraction: float = DEFAULT_TOP_CROP_FRACTION,
) -> np.ndarray:
    """Keep only the top fraction of an image."""
    crop_h = top_fraction_height(image.shape[0], fraction)
    return image[:crop_h, :, :]


def letterbox_image(
    image: np.ndarray,
    input_size: tuple[int, int],
    pad_value: int = 114,
) -> tuple[np.ndarray, float]:
    """Resize without distortion and pad to a YOLO-style canvas."""
    input_h, input_w = input_size
    ih, iw = image.shape[:2]
    r = min(input_w / iw, input_h / ih)
    new_w, new_h = int(iw * r), int(ih * r)
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.full((input_h, input_w, 3), pad_value, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized
    return canvas, r


class Preprocessor:
    """Resize, normalize, and convert an image to a model-ready tensor.

    Uses YOLOX-compatible preprocessing: letterbox resize that preserves
    aspect ratio (padding with value 114) and outputs float32 in [0, 255].
    """

    def __init__(self, config: PreprocessConfig) -> None:
        self.config = config
        self.input_h, self.input_w = config.input_size
        self.swap_rb = config.swap_rb
        self.top_crop_only = config.top_crop_only or config.top_third_only
        self.top_crop_fraction = config.top_crop_fraction

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
        if self.top_crop_only:
            img = crop_top_fraction(img, self.top_crop_fraction)

        if self.swap_rb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Letterbox resize — maintain aspect ratio, pad with 114
        canvas, r = letterbox_image(img, (self.input_h, self.input_w))

        # float32 in [0, 255] — YOLOX convention (no /255 normalisation)
        img_f = canvas.astype(np.float32)

        img_f = img_f.transpose(2, 0, 1)  # HWC → CHW

        return torch.from_numpy(np.ascontiguousarray(img_f)), r
