from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from adas_perception.traffic_light.detector.augmentations import (
    horizontal_flip,
    hsv_jitter,
    mosaic,
    random_crop_translate,
    random_scale_jitter,
)

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
        positive_images_only: bool = False,
        mosaic_prob: float = 1.0,
        scale_jitter_range: Tuple[float, float] = (0.5, 1.5),
        hsv_hue: float = 0.015,
        hsv_sat: float = 0.7,
        hsv_val: float = 0.4,
        flip_prob: float = 0.5,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / split
        self.input_size = input_size  # (H, W) — YOLOX convention
        self.max_labels = max_labels
        self.augment = augment

        # Augmentation parameters
        self.mosaic_prob = mosaic_prob
        self.scale_jitter_range = scale_jitter_range
        self.hsv_hue = hsv_hue
        self.hsv_sat = hsv_sat
        self.hsv_val = hsv_val
        self.flip_prob = flip_prob

        ann_path = self.data_dir / "annotations" / json_file
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images = {img["id"]: img for img in coco["images"]}

        # Build category_id -> 0-based class index.
        self.cat_to_idx = {
            cat["id"]: i for i, cat in enumerate(coco["categories"])
        }

        # Group only valid annotations by image_id.
        self.anns_by_img: dict[int, List[dict]] = {}
        for ann in coco["annotations"]:
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            if ann["category_id"] not in self.cat_to_idx:
                continue
            self.anns_by_img.setdefault(ann["image_id"], []).append(ann)

        self.img_ids = list(self.images.keys())
        if positive_images_only:
            total_images = len(self.img_ids)
            self.img_ids = [
                img_id for img_id in self.img_ids if self.anns_by_img.get(img_id)
            ]
            logger.info(
                "Using positive-only %s split: %d -> %d images",
                split,
                total_images,
                len(self.img_ids),
            )

    def __len__(self) -> int:
        return len(self.img_ids)

    def _load_raw(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a single image with bboxes and class ids (no augmentation).

        Returns ``(image, bboxes, class_ids)`` where *bboxes* is ``(N, 4)``
        ``[x1, y1, x2, y2]`` and *class_ids* is ``(N,)``.
        """
        img_id = self.img_ids[idx]
        info = self.images[img_id]
        img_path = self.image_dir / info["file_name"]

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        anns = self.anns_by_img.get(img_id, [])
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
        return image, bboxes, class_ids

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        th, tw = self.input_size

        if self.augment and random.random() < self.mosaic_prob:
            # --- mosaic: combine 4 random images ---
            indices = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
            imgs, bbs, cids = zip(*(self._load_raw(i) for i in indices))
            image, bboxes, class_ids = mosaic(
                list(imgs), list(bbs), list(cids), self.input_size,
            )
            # Scale jitter + crop/translate on the mosaic output
            if self.scale_jitter_range != (1.0, 1.0):
                image, bboxes = random_scale_jitter(
                    image, bboxes, self.scale_jitter_range,
                )
                image, bboxes = random_crop_translate(
                    image, bboxes, self.input_size,
                )
        else:
            image, bboxes, class_ids = self._load_raw(idx)

        # --- per-image augmentations (photometric + flip) ---
        if self.augment:
            if len(bboxes) > 0 and random.random() < self.flip_prob:
                image, bboxes = horizontal_flip(image, bboxes)
            image = hsv_jitter(
                image, self.hsv_hue, self.hsv_sat, self.hsv_val,
            )

        # --- resize to input_size (letterbox) ---
        ih, iw = image.shape[:2]
        r = min(tw / iw, th / ih)
        new_w, new_h = int(iw * r), int(ih * r)
        resized = cv2.resize(image, (new_w, new_h))

        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized
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
        no_mosaic_epochs: int = 15,
        augment: bool = True,
        positive_images_only: bool = False,
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
        * ``no_mosaic_epochs`` (int, default 15) — disable mosaic for the final
          N epochs so the model adapts to the single-image inference distribution.
          Set to 0 to keep mosaic on for all epochs.
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
        # Keep exp metadata in sync with our requested resolution. The YOLOX
        # model itself is fully convolutional, but exp.input_size / test_size
        # are read by the head/label assigner and by export utilities.
        exp.input_size = tuple(input_size)
        exp.test_size = tuple(input_size)
        # Disable YOLOX's built-in multi-scale jitter — at 960 it would push
        # VRAM over the edge and our augmentations are handled in-dataset.
        exp.random_size = None
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

        # Allow cuDNN to benchmark and cache the fastest conv algorithm for this
        # fixed input resolution.  Must be set before the first forward pass.
        if device != "cpu":
            torch.backends.cudnn.benchmark = True

        # ---- data ----
        dataset = _COCODetectionDataset(
            data_dir=str(dataset_path),
            json_file="instances_train.json",
            split="train",
            input_size=input_size,
            max_labels=50,
            augment=augment,
            positive_images_only=positive_images_only,
        )
        num_workers = self.model_config.get("data_num_workers", 4)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device != "cpu"),
            drop_last=True,
            # Keep worker processes alive between epochs to avoid the
            # spawn/teardown overhead on every epoch (significant on Windows).
            persistent_workers=(num_workers > 0),
            prefetch_factor=(4 if num_workers > 0 else None),
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

        no_mosaic_start = epochs - no_mosaic_epochs if no_mosaic_epochs > 0 else epochs

        logger.info(
            "Training: %d epochs, batch=%d, lr=%.1e, device=%s, "
            "%d images, %d iters/epoch, patience=%s, augment=%s, positive_images_only=%s, no_mosaic_after_epoch=%d",
            epochs, batch_size, lr, device, len(dataset), len(loader),
            patience if patience > 0 else "off",
            augment,
            positive_images_only,
            no_mosaic_start,
        )

        # ---- training loop ----
        for epoch in range(epochs):
            # Mosaic cooldown: disable for the final no_mosaic_epochs epochs.
            if augment and epoch == no_mosaic_start and no_mosaic_epochs > 0:
                dataset.mosaic_prob = 0.0
                logger.info(
                    "Epoch %d: mosaic disabled for final %d epochs",
                    epoch + 1, no_mosaic_epochs,
                )

            model.train()
            epoch_loss = 0.0

            for it, batch in enumerate(loader):
                # Zero gradients at the top with set_to_none for a small
                # memory/speed win (avoids memset to 0 on every buffer).
                optimizer.zero_grad(set_to_none=True)

                imgs = batch[0].to(device, dtype=torch.float32, non_blocking=True)
                targets = batch[1].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(imgs, targets)
                loss = outputs["total_loss"]

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
