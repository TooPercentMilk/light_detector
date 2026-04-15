from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Minimal COCO dataset compatible with YOLOX training
# ------------------------------------------------------------------

class _COCODetectionDataset(Dataset):
    """Loads COCO-format annotations and produces (image, labels) pairs for YOLOX.

    Each labels tensor has shape ``(max_labels, 5)`` where each row is
    ``[class_id, cx, cy, w, h]`` in pixel coordinates of the resized image.
    Zero-padded rows indicate unused slots.
    """

    def __init__(
        self,
        data_dir: str,
        json_file: str,
        split: str,
        input_size: Tuple[int, int],
        max_labels: int = 50,
        augment: bool = True,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / split
        self.input_size = input_size  # (H, W) — YOLOX convention
        self.max_labels = max_labels
        self.augment = augment

        ann_path = self.data_dir / "annotations" / json_file
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}
        self.img_ids = list(self.images.keys())

        # Group annotations by image_id
        self.anns_by_img: dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

        # Build category_id -> 0-based class index
        self.cat_to_idx = {
            cat["id"]: i for i, cat in enumerate(coco["categories"])
        }

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_id = self.img_ids[idx]
        info = self.images[img_id]
        img_path = self.image_dir / info["file_name"]

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        anns = self.anns_by_img.get(img_id, [])

        # Collect bboxes as [x1, y1, x2, y2] and class ids
        bboxes = []
        class_ids = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            cat_idx = self.cat_to_idx.get(ann["category_id"])
            if cat_idx is None:
                continue
            bboxes.append([x, y, x + w, y + h])
            class_ids.append(cat_idx)

        bboxes = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
        class_ids = np.array(class_ids, dtype=np.float32)

        # --- augmentation ---
        if self.augment and len(bboxes) > 0:
            image, bboxes = self._augment(image, bboxes)

        # --- resize to input_size (H, W) ---
        ih, iw = image.shape[:2]
        th, tw = self.input_size
        r = min(tw / iw, th / ih)
        new_w, new_h = int(iw * r), int(ih * r)
        resized = cv2.resize(image, (new_w, new_h))

        # Paste onto padded canvas
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        # Scale bboxes
        bboxes *= r

        # --- build labels: [class_id, cx, cy, w, h] ---
        padded = np.zeros((self.max_labels, 5), dtype=np.float32)
        n = min(len(bboxes), self.max_labels)
        if n > 0:
            x1 = bboxes[:n, 0]
            y1 = bboxes[:n, 1]
            x2 = bboxes[:n, 2]
            y2 = bboxes[:n, 3]
            padded[:n, 0] = class_ids[:n]
            padded[:n, 1] = (x1 + x2) / 2  # cx
            padded[:n, 2] = (y1 + y2) / 2  # cy
            padded[:n, 3] = x2 - x1        # w
            padded[:n, 4] = y2 - y1        # h

        # HWC BGR -> CHW RGB, float32 [0, 255] (YOLOX convention — no /255)
        img_t = canvas[:, :, ::-1].transpose(2, 0, 1).copy()
        img_t = np.ascontiguousarray(img_t, dtype=np.float32)

        return torch.from_numpy(img_t), torch.from_numpy(padded)

    @staticmethod
    def _augment(image: np.ndarray, bboxes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simple augmentation: horizontal flip + colour jitter."""
        if np.random.rand() < 0.5:
            h, w = image.shape[:2]
            image = cv2.flip(image, 1)
            x1 = bboxes[:, 0].copy()
            bboxes[:, 0] = w - bboxes[:, 2]
            bboxes[:, 2] = w - x1

        # Brightness / contrast jitter
        alpha = np.random.uniform(0.8, 1.2)
        beta = np.random.uniform(-15, 15)
        image = np.clip(alpha * image.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        return image, bboxes


# ------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------

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
        patience: int = 0,
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
        from yolox.exp import get_exp

        dataset_path = Path(dataset_path).resolve()
        ann_file = dataset_path / "annotations" / "instances_train.json"
        if not ann_file.is_file():
            raise FileNotFoundError(
                f"Training annotation file not found: {ann_file}"
            )
        num_classes = self.model_config.get("num_classes", 1)
        input_size = tuple(self.model_config.get("input_size", (640, 640)))
        device = self.model_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        pretrained_ckpt = self.model_config.get("pretrained_ckpt")
        exp_name = self.model_config.get("exp_name", "yolox-m")

        # ---- model ----
        exp = get_exp(None, exp_name)
        exp.num_classes = num_classes
        model = exp.get_model()

        if pretrained_ckpt and Path(pretrained_ckpt).is_file():
            ckpt = torch.load(
                pretrained_ckpt, map_location="cpu", weights_only=True
            )
            src_state = ckpt.get("model", ckpt)
            # Filter out keys whose shapes don't match (e.g. cls head from 80-class COCO)
            model_state = model.state_dict()
            compatible = {
                k: v for k, v in src_state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            skipped = set(src_state.keys()) - set(compatible.keys())
            if skipped:
                logger.info("Skipped %d keys with shape mismatch: %s", len(skipped), skipped)
            model.load_state_dict(compatible, strict=False)
            logger.info("Loaded pretrained weights from %s (%d/%d keys)", pretrained_ckpt, len(compatible), len(src_state))

        model.to(device)
        model.train()

        # ---- data ----
        dataset = _COCODetectionDataset(
            data_dir=str(dataset_path),
            json_file="instances_train.json",
            split="train",
            input_size=input_size,
            max_labels=50,
            augment=True,
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

        best_loss = float("inf")
        epochs_no_improve = 0

        logger.info(
            "Training: %d epochs, batch=%d, lr=%.1e, device=%s, "
            "%d images, %d iters/epoch, patience=%s",
            epochs, batch_size, lr, device, len(dataset), len(loader),
            patience if patience > 0 else "off",
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

            # early stopping based on training loss
            if avg < best_loss:
                best_loss = avg
                epochs_no_improve = 0
                best_path = self.output_dir / "best.pth"
                torch.save({"model": model.state_dict(), "epoch": epoch + 1}, best_path)
                logger.info("New best loss — saved %s", best_path)
            else:
                epochs_no_improve += 1

            if patience > 0 and epochs_no_improve >= patience:
                logger.info(
                    "Early stopping: no improvement for %d epochs", patience
                )
                break

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
