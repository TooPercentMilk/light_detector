from __future__ import annotations

import logging
from pathlib import Path

import torch

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
            Path to a COCO-format dataset directory containing
            ``annotations/instances_train.json`` and a ``train/`` image folder.
        epochs, batch_size, lr:
            Standard training hyperparameters.

        Additional keys accepted via *model_config* (passed at ``__init__``):

        * ``num_classes`` (int, default 1)
        * ``input_size`` (list[int], default [640, 640])
        * ``device`` (str, default "cuda" or "cpu")
        * ``pretrained_ckpt`` (str) — path to backbone weights for fine-tuning
        * ``exp_name`` (str, default "yolox-s") — YOLOX experiment variant
        * ``data_num_workers`` (int, default 4)
        """
        from yolox.data import COCODataset, TrainTransform
        from yolox.exp import get_exp

        dataset_path = Path(dataset_path).resolve()
        ann_file = dataset_path / "annotations" / "instances_train.json"
        if not ann_file.is_file():
            raise FileNotFoundError(
                f"Training annotation file not found: {ann_file}"
            )
        dataset_path = str(dataset_path)
        num_classes = self.model_config.get("num_classes", 1)
        input_size = tuple(self.model_config.get("input_size", (640, 640)))
        device = self.model_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        pretrained_ckpt = self.model_config.get("pretrained_ckpt")
        exp_name = self.model_config.get("exp_name", "yolox-s")

        # ---- model ----
        exp = get_exp(None, exp_name)
        exp.num_classes = num_classes
        model = exp.get_model()

        if pretrained_ckpt and Path(pretrained_ckpt).is_file():
            ckpt = torch.load(
                pretrained_ckpt, map_location="cpu", weights_only=True
            )
            model.load_state_dict(ckpt.get("model", ckpt), strict=False)
            logger.info("Loaded pretrained weights from %s", pretrained_ckpt)

        model.to(device)
        model.train()

        # ---- data ----
        dataset = COCODataset(
            data_dir=dataset_path,
            json_file="instances_train.json",
            name="train",
            img_size=input_size,
            preproc=TrainTransform(
                max_labels=50,
            ),
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.model_config.get("data_num_workers", 4),
            pin_memory=(device != "cpu"),
            drop_last=True,
        )

        if len(loader) == 0:
            raise RuntimeError(
                f"Training loader is empty ({len(dataset)} images, "
                f"batch_size={batch_size}). Verify dataset_path: {dataset_path}"
            )

        # ---- optimizer (3 param groups, matching YOLOX convention) ----
        pg_bn, pg_weight, pg_bias = [], [], []
        for module in model.modules():
            if hasattr(module, "bias") and isinstance(
                module.bias, torch.nn.Parameter
            ):
                pg_bias.append(module.bias)
            if isinstance(module, torch.nn.BatchNorm2d):
                pg_bn.append(module.weight)
            elif hasattr(module, "weight") and isinstance(
                module.weight, torch.nn.Parameter
            ):
                pg_weight.append(module.weight)

        optimizer = torch.optim.SGD(
            pg_bn, lr=lr, momentum=0.9, nesterov=True
        )
        optimizer.add_param_group({"params": pg_weight, "weight_decay": 5e-4})
        optimizer.add_param_group({"params": pg_bias})

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.05
        )

        use_amp = device != "cpu"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Training: %d epochs, batch=%d, lr=%.1e, device=%s, "
            "%d images, %d iters/epoch",
            epochs, batch_size, lr, device, len(dataset), len(loader),
        )

        # ---- training loop ----
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0

            for it, batch in enumerate(loader):
                imgs = batch[0].to(device, dtype=torch.float32)
                targets = batch[1].to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(imgs, targets)
                loss = outputs["total_loss"]

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()

                if (it + 1) % 50 == 0 or it + 1 == len(loader):
                    logger.info(
                        "Epoch [%d/%d]  Iter [%d/%d]  loss=%.4f  lr=%.2e",
                        epoch + 1, epochs, it + 1, len(loader),
                        loss.item(), optimizer.param_groups[0]["lr"],
                    )

            scheduler.step()
            avg = epoch_loss / len(loader)
            logger.info("Epoch %d/%d — avg_loss=%.4f", epoch + 1, epochs, avg)

            # periodic checkpoint
            save_interval = max(1, epochs // 5)
            if (epoch + 1) % save_interval == 0 or epoch + 1 == epochs:
                ckpt_path = self.output_dir / f"epoch_{epoch + 1}.pth"
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch + 1},
                    ckpt_path,
                )
                logger.info("Saved %s", ckpt_path)

        final_path = self.output_dir / "yolox_tl.pth"
        torch.save({"model": model.state_dict(), "epoch": epochs}, final_path)
        logger.info("Training complete — %s", final_path)
