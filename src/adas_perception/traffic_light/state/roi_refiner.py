from __future__ import annotations

import numpy as np


class RoiRefiner:
    """Crop and optionally pad a detected bounding box from a full frame."""

    def __init__(self, padding_ratio: float = 0.1) -> None:
        self.padding_ratio = padding_ratio

    def refine(self, image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        """Extract a padded crop around *bbox* from *image*.

        Parameters
        ----------
        image:
            Full-frame HWC uint8 image.
        bbox:
            (4,) array in x1y1x2y2 format.

        Returns
        -------
        Cropped HWC uint8 sub-image.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox.astype(int)

        bw, bh = x2 - x1, y2 - y1
        pad_x = int(bw * self.padding_ratio)
        pad_y = int(bh * self.padding_ratio)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        return image[y1:y2, x1:x2].copy()
