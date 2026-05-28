"""Training pipeline for the traffic-light state classifier (MobileNetV3-Small).

Reads COCO-format annotations produced by ``convert_lisa_to_coco.py`` and
crops ROIs from the source images to build a classification dataset on the fly.

Usage (programmatic)::

    from adas_perception.traffic_light.state.trainer import ClassifierTrainer

    trainer = ClassifierTrainer(output_dir="runs/train_classifier")
    trainer.train("data/coco_tl", epochs=30, batch_size=64)

Usage (CLI)::

    python scripts/train_classifier.py \\
        --dataset data/coco_tl --epochs 30 --batch-size 64 --lr 3e-4
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from adas_perception.traffic_light.training_plots import write_loss_curve

logger = logging.getLogger(__name__)


def _torch_load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _to_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _state_dict_to_cpu(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return _to_cpu(state_dict)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class _ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9998) -> None:
        self.decay = float(decay)
        self.state = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        model_state = model.state_dict()
        for key, value in model_state.items():
            shadow = self.state[key]
            current = value.detach()
            if torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(current, alpha=1.0 - self.decay)
            else:
                shadow.copy_(current)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.state

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        restored = {}
        for key, value in state_dict.items():
            current = self.state.get(key)
            if current is not None and torch.is_tensor(value):
                restored[key] = value.detach().to(device=current.device, dtype=current.dtype).clone()
            elif torch.is_tensor(value):
                restored[key] = value.detach().clone()
            else:
                restored[key] = value
        self.state = restored

# LISA tag -> classifier class index  (must match config.classes order)
_TAG_TO_CLASS: Dict[str, int] = {
    "go": 2,         # green
    "goLeft": 2,
    "goForward": 2,
    "stop": 0,        # red
    "stopLeft": 0,
    "warning": 1,     # yellow
    "warningLeft": 1,
}

_CLASS_NAMES = {0: "red", 1: "yellow", 2: "green", 3: "off"}


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class _CropDataset(Dataset):
    """Lazily crops traffic-light ROIs from full images using COCO annotations."""

    def __init__(
        self,
        data_dir: Path,
        split: str,
        input_size: Tuple[int, int],
        augment: bool = False,
    ) -> None:
        self.image_dir = data_dir / split
        ann_path = data_dir / "annotations" / f"instances_{split}.json"
        if not ann_path.is_file():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

        # Build list of (image_path, bbox_xywh, class_idx) samples
        self.samples: List[Tuple[Path, List[float], int]] = []
        skipped = 0
        for ann in coco["annotations"]:
            tag = ann.get("attributes", {}).get("lisa_tag")
            if tag is None or tag not in _TAG_TO_CLASS:
                skipped += 1
                continue
            fname = id_to_file.get(ann["image_id"])
            if fname is None:
                continue
            cls = _TAG_TO_CLASS[tag]
            self.samples.append(
                (self.image_dir / fname, ann["bbox"], cls)
            )

        if skipped:
            logger.info(
                "Skipped %d annotations with unknown/missing tags", skipped
            )

        # input_size is (W, H) to match ClassifierConfig convention
        self.input_w, self.input_h = input_size
        self.augment = augment

        # Normalisation (ImageNet stats, matching inference in classifier.py)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, bbox, cls = self.samples[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        # bbox is COCO format [x, y, w, h]
        x, y, w, h = [int(round(v)) for v in bbox]
        ih, iw = image.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(iw, x + w)
        y2 = min(ih, y + h)
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            # Fallback: tiny black image (should be rare)
            crop = np.zeros((self.input_h, self.input_w, 3), dtype=np.uint8)

        crop = cv2.resize(crop, (self.input_w, self.input_h))

        # Augmentation
        if self.augment:
            if np.random.rand() < 0.5:
                crop = cv2.flip(crop, 1)  # horizontal flip
            # Random brightness/contrast jitter
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-15, 15)
            crop = np.clip(alpha * crop.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # BGR -> RGB, HWC -> CHW, uint8 -> float32 [0,1]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        tensor = self.normalize(tensor)

        return tensor, cls


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

class ClassifierTrainer:
    """Trains a MobileNetV3-Small traffic-light state classifier."""

    def __init__(
        self,
        num_classes: int = 4,
        input_size: Tuple[int, int] = (32, 64),
        output_dir: str = "runs/train_classifier",
        pretrained_backbone: bool = True,
    ) -> None:
        self.num_classes = num_classes
        self.input_size = input_size  # (W, H)
        self.output_dir = Path(output_dir)
        self.pretrained_backbone = pretrained_backbone

    def _build_model(self) -> torch.nn.Module:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if self.pretrained_backbone else None
        model = models.mobilenet_v3_small(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, self.num_classes)
        return model

    def train(
        self,
        dataset_path: str,
        epochs: int = 30,
        batch_size: int = 64,
        lr: float = 3e-4,
        device: str | None = None,
        num_workers: int = 4,
        val_interval: int = 1,
        patience: int = 0,
        resume_checkpoint: str | None = None,
    ) -> None:
        """Run the training loop.

        Parameters
        ----------
        dataset_path:
            Path to the COCO-format dataset directory (``data/coco_tl``).
            Must contain ``annotations/instances_train.json`` and ``train/``.
        epochs, batch_size, lr:
            Standard training hyperparameters.
        device:
            ``"cuda"`` or ``"cpu"``. Auto-detected if *None*.
        num_workers:
            DataLoader workers.
        val_interval:
            Run validation every N epochs.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dev = torch.device(device)
        data_dir = Path(dataset_path).resolve()

        # ---- datasets ----
        train_ds = _CropDataset(data_dir, "train", self.input_size, augment=True)
        logger.info("Training samples: %d", len(train_ds))

        val_ann = data_dir / "annotations" / "instances_val.json"
        val_ds = _CropDataset(data_dir, "val", self.input_size, augment=False) if val_ann.is_file() else None
        if val_ds is not None:
            logger.info("Validation samples: %d", len(val_ds))

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device != "cpu"),
            drop_last=True,
        )
        val_loader = (
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            if val_ds is not None
            else None
        )

        if len(train_loader) == 0:
            raise RuntimeError(
                f"Training loader is empty ({len(train_ds)} samples, "
                f"batch_size={batch_size}). Check dataset_path: {dataset_path}"
            )

        # ---- model ----
        model = self._build_model().to(dev)
        resume_data = None
        if resume_checkpoint:
            resume_path = Path(resume_checkpoint)
            if resume_path.is_file():
                resume_data = _torch_load_checkpoint(resume_path, map_location="cpu")
                src_state = (
                    resume_data.get("model", resume_data)
                    if isinstance(resume_data, dict)
                    else resume_data
                )
                model.load_state_dict(src_state)
                logger.info("Loaded classifier model state from resume checkpoint: %s", resume_path)
            else:
                logger.warning("Classifier resume checkpoint not found: %s", resume_path)

        # ---- optimizer ----
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr * 0.01
        )

        use_amp = device != "cpu"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        ema = _ModelEMA(model)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        best_val_acc = 0.0
        epochs_no_improve = 0
        start_epoch = 0
        last_completed_epoch = 0
        loss_history: list[dict[str, float | int | None]] = []

        if isinstance(resume_data, dict):
            start_epoch = int(resume_data.get("epoch", 0) or 0)
            last_completed_epoch = start_epoch
            best_val_acc = float(resume_data.get("best_val_acc", best_val_acc))
            epochs_no_improve = int(resume_data.get("epochs_no_improve", epochs_no_improve) or 0)
            saved_history = resume_data.get("loss_history", [])
            if isinstance(saved_history, list):
                loss_history = [
                    row for row in saved_history
                    if isinstance(row, dict)
                    and int(row.get("epoch", 0) or 0) <= start_epoch
                ]
            if "optimizer" in resume_data:
                optimizer.load_state_dict(resume_data["optimizer"])
                logger.info("Restored classifier optimizer state from %s", resume_checkpoint)
            else:
                logger.warning("Classifier resume checkpoint has no optimizer state: %s", resume_checkpoint)
            if "scheduler" in resume_data:
                scheduler.load_state_dict(resume_data["scheduler"])
                logger.info("Restored classifier scheduler state from %s", resume_checkpoint)
                saved_epochs = resume_data.get("training", {}).get("epochs")
                if saved_epochs is not None and int(saved_epochs) != int(epochs):
                    scheduler.T_max = int(epochs)
                    logger.info(
                        "Adjusted classifier scheduler T_max from saved %s to requested %s epochs",
                        saved_epochs,
                        epochs,
                    )
            else:
                logger.warning("Classifier resume checkpoint has no scheduler state: %s", resume_checkpoint)
            if "scaler" in resume_data:
                scaler.load_state_dict(resume_data["scaler"])
            ema_state = resume_data.get("ema_model") or resume_data.get("ema")
            if ema_state is not None:
                ema.load_state_dict(ema_state)
                logger.info("Restored classifier EMA weights from %s", resume_checkpoint)
            else:
                logger.warning("Classifier resume checkpoint has no EMA weights: %s", resume_checkpoint)

        logger.info(
            "Training classifier: %d epochs, batch=%d, lr=%.1e, device=%s, start_epoch=%d, patience=%s",
            epochs, batch_size, lr, device, start_epoch,
            patience if patience > 0 else "off",
        )
        if start_epoch >= epochs:
            logger.warning(
                "Classifier resume checkpoint epoch %d is already at or beyond requested epochs=%d",
                start_epoch,
                epochs,
            )

        def save_checkpoint(
            path: Path,
            epoch_number: int,
            metric_name: str,
            metric_value: float,
            training_stopped_at: str | None = None,
        ) -> None:
            checkpoint = {
                "format_version": 2,
                "checkpoint_type": "classifier",
                "epoch": int(epoch_number),
                "model": _state_dict_to_cpu(model.state_dict()),
                "optimizer": _to_cpu(optimizer.state_dict()),
                "scheduler": _to_cpu(scheduler.state_dict()),
                "scaler": _to_cpu(scaler.state_dict()),
                "ema_model": _state_dict_to_cpu(ema.state_dict()),
                "ema_decay": ema.decay,
                "best_val_acc": float(best_val_acc),
                "epochs_no_improve": int(epochs_no_improve),
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "loss_history": loss_history,
                "training": {
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "lr": float(lr),
                    "patience": int(patience),
                    "val_interval": int(val_interval),
                    "input_size": list(self.input_size),
                    "num_classes": int(self.num_classes),
                },
            }
            if training_stopped_at is not None:
                checkpoint["training_stopped_at"] = training_stopped_at
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, path)

        # ---- training loop ----
        for epoch in range(start_epoch + 1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for it, (imgs, labels) in enumerate(train_loader, 1):
                imgs = imgs.to(dev)
                labels = labels.to(dev)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(imgs)
                    loss = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                ema.update(model)

                epoch_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if it % 50 == 0 or it == len(train_loader):
                    logger.info(
                        "Epoch [%d/%d]  Iter [%d/%d]  loss=%.4f  lr=%.2e",
                        epoch, epochs, it, len(train_loader),
                        loss.item(), optimizer.param_groups[0]["lr"],
                    )

            scheduler.step()
            train_acc = correct / total if total > 0 else 0.0
            avg_loss = epoch_loss / len(train_loader)
            logger.info(
                "Epoch %d/%d — avg_loss=%.4f  train_acc=%.2f%%",
                epoch, epochs, avg_loss, train_acc * 100,
            )

            # ---- validation ----
            stop_training = False
            latest_metric_name = "train_loss"
            latest_metric_value = avg_loss
            val_loss = None
            if val_loader is not None and epoch % val_interval == 0:
                val_acc, val_loss = self._validate(model, val_loader, dev, use_amp)
                logger.info("  val_loss=%.4f  val_acc=%.2f%%", val_loss, val_acc * 100)
                latest_metric_name = "val_acc"
                latest_metric_value = val_acc

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_no_improve = 0
                    best_path = self.output_dir / "best.pth"
                    save_checkpoint(best_path, epoch, "val_acc", val_acc)
                    logger.info("  New best — saved %s", best_path)
                else:
                    epochs_no_improve += 1

                if patience > 0 and epochs_no_improve >= patience:
                    logger.info(
                        "Early stopping: no improvement for %d epochs", patience
                    )
                    stop_training = True

            elif val_loader is None and train_acc > best_val_acc:
                best_val_acc = train_acc
                epochs_no_improve = 0
                best_path = self.output_dir / "best.pth"
                save_checkpoint(best_path, epoch, "train_acc", train_acc)
                logger.info("  New best train_acc=%.2f%% - saved %s", train_acc * 100, best_path)

            last_completed_epoch = epoch
            loss_history.append(
                {
                    "epoch": int(epoch),
                    "train_loss": float(avg_loss),
                    "val_loss": float(val_loss) if val_loss is not None else None,
                }
            )
            write_loss_curve(
                loss_history,
                self.output_dir / "loss_curve.png",
                "Classifier Training and Validation Loss",
            )

            latest_path = self.output_dir / "latest.pth"
            save_checkpoint(latest_path, epoch, latest_metric_name, latest_metric_value)
            logger.info("Saved latest checkpoint -> %s", latest_path)

            # periodic checkpoint
            save_interval = max(1, epochs // 5)
            if epoch % save_interval == 0 or epoch == epochs:
                ckpt_path = self.output_dir / f"epoch_{epoch}.pth"
                save_checkpoint(ckpt_path, epoch, latest_metric_name, latest_metric_value)
                logger.info("Saved %s", ckpt_path)

            if stop_training:
                break

        final_path = self.output_dir / "tl_state_classifier.pth"
        final_metric = avg_loss if "avg_loss" in locals() else float("nan")
        save_checkpoint(
            final_path,
            last_completed_epoch,
            "train_loss",
            final_metric,
            training_stopped_at=_utc_now_iso(),
        )
        logger.info("Training complete — %s", final_path)

    @staticmethod
    def _validate(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        use_amp: bool,
    ) -> tuple[float, float]:
        model.eval()
        correct = 0
        total = 0
        loss_total = 0.0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(imgs)
                    loss = F.cross_entropy(logits, labels)
                batch_size = labels.size(0)
                loss_total += loss.item() * batch_size
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += batch_size
        val_acc = correct / total if total > 0 else 0.0
        val_loss = loss_total / total if total > 0 else float("nan")
        return val_acc, val_loss
