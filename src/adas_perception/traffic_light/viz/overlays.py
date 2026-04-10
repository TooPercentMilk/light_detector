from __future__ import annotations

from typing import List

import numpy as np

from ..schemas import TrafficLight


def draw_traffic_lights(
    image: np.ndarray,
    lights: List[TrafficLight],
) -> np.ndarray:
    """Draw bounding boxes and state labels onto *image*.

    Parameters
    ----------
    image:
        HWC uint8 BGR frame (will **not** be modified in-place).
    lights:
        Traffic light results to visualise.

    Returns
    -------
    A copy of *image* with overlays drawn.
    """
    # TODO: cv2.rectangle + cv2.putText per light, colour-coded by state
    return image.copy()
