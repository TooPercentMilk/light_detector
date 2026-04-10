from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class YoloxTrainer:
    """Trains a YOLOX model on a traffic light detection dataset.

    Requires the ``yolox`` package (``pip install -e ".[models]"``).
    Leverages YOLOX's built-in Exp system for experiment configuration.
    """

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

        The intended integration pattern::

            from yolox.exp import get_exp

            exp = get_exp(None, "yolox-s")
            exp.data_dir = dataset_path
            exp.max_epoch = epochs
            exp.basic_lr_per_img = lr / (batch_size * 8)
            exp.output_dir = str(self.output_dir)

            trainer = exp.get_trainer(...)  # YOLOX's own Trainer
            trainer.train()
        """
        # TODO: build dataset, data-loader, optimizer, training loop
        raise NotImplementedError("YoloxTrainer.train not yet implemented")
