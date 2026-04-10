from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def export_onnx(
    model: Any,
    output_path: str,
    input_size: tuple[int, int] = (640, 640),
    opset_version: int = 11,
) -> None:
    """Export a PyTorch detector to ONNX format.

    Parameters
    ----------
    model:
        A loaded PyTorch model (e.g. ``YoloxWrapper.model``).
    output_path:
        Destination ``.onnx`` file path.
    input_size:
        (H, W) of the dummy input tensor.
    opset_version:
        ONNX opset version to target.
    """
    # TODO: torch.onnx.export with dynamic axes
    raise NotImplementedError("export_onnx not yet implemented")
