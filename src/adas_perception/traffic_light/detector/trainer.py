from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class YoloxTrainer:
    """Trains a YOLOX model on a traffic light detection dataset."""

    def __init__(
        self,
        model_config: dict | None = None,
        output_dir: str = "runs/train",
    ) -> None:
        self.model_config = model_config or {}
        self.output_dir = Path(output_dir)

    def train(
        self,
        dataset_path: str,
        epochs: int = 50,
        batch_size: int = 16,
        lr: float = 1e-3,
    ) -> None:
        """Run the training loop.

        Parameters
        ----------
        dataset_path:
            Path to a COCO-format dataset directory.
        epochs, batch_size, lr:
            Standard training hyperparameters.
        """
        # TODO: build dataset, data-loader, optimizer, training loop
        raise NotImplementedError("YoloxTrainer.train not yet implemented")
