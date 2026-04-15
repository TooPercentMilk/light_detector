from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from ..schemas import LightState, TrafficLight

# BGR colours for each state
_STATE_COLOURS: Dict[LightState, Tuple[int, int, int]] = {
    LightState.RED: (0, 0, 255),
    LightState.YELLOW: (0, 255, 255),
    LightState.GREEN: (0, 255, 0),
    LightState.OFF: (128, 128, 128),
    LightState.UNKNOWN: (200, 200, 200),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_THICKNESS = 2
_TEXT_THICKNESS = 1


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
    vis = image.copy()

    for light in lights:
        colour = _STATE_COLOURS.get(light.state, (200, 200, 200))
        x1, y1, x2, y2 = light.bbox.astype(int)

        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, _THICKNESS)

        label = (
            f"#{light.track_id} {light.state.value} "
            f"d={light.detection_confidence:.2f} s={light.state_confidence:.2f}"
        )
        (tw, th), baseline = cv2.getTextSize(label, _FONT, _FONT_SCALE, _TEXT_THICKNESS)
        # Draw text background
        text_y = max(y1 - 6, th + baseline)
        cv2.rectangle(vis, (x1, text_y - th - baseline), (x1 + tw, text_y + baseline), colour, cv2.FILLED)
        cv2.putText(vis, label, (x1, text_y), _FONT, _FONT_SCALE, (0, 0, 0), _TEXT_THICKNESS, cv2.LINE_AA)

    return vis
