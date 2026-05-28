from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def export_onnx(
    model: Any,
    output_path: str,
    input_size: tuple[int, int] = (1280, 1280),
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
    model.eval()
    dummy = torch.randn(1, 3, *input_size)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        opset_version=opset_version,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
    )
    logger.info("Exported ONNX model to %s", output_path)
