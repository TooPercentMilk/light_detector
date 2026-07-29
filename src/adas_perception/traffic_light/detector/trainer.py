from __future__ import annotations

import json
import logging
import math
import random
from datetime import datetime, timezone
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
from adas_perception.traffic_light.preprocess import (
    DEFAULT_TOP_CROP_FRACTION,
    crop_top_fraction,
    letterbox_image,
    top_fraction_height,
)
from adas_perception.traffic_light.training_plots import write_loss_curve

logger = logging.getLogger(__name__)


def _torch_load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu"):
    """Load trusted local training checkpoints across PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _torch_load_weights(path: str | Path, map_location: str | torch.device = "cpu"):
    """Load a weights file while remaining compatible with older PyTorch."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
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
    """Minimal model EMA tracker that stores a state_dict-shaped shadow copy."""

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
        scale_jitter_range: Tuple[float, float] = (0.8, 1.2),
        hsv_hue: float = 0.015,
        hsv_sat: float = 0.7,
        hsv_val: float = 0.4,
        flip_prob: float = 0.5,
        small_light_flip: bool = False,
        small_light_flip_range: Tuple[float, float] = (8.0, 24.0),
        top_crop_only: bool = False,
        top_crop_fraction: float = DEFAULT_TOP_CROP_FRACTION,
        top_third_only: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / split
        self.input_size = input_size  # (H, W) — YOLOX convention
        self.max_labels = max_labels
        self.augment = augment
        self.top_crop_only = top_crop_only or top_third_only
        self.top_crop_fraction = float(top_crop_fraction)

        # Augmentation parameters
        self.mosaic_prob = mosaic_prob
        self.scale_jitter_range = scale_jitter_range
        self.hsv_hue = hsv_hue
        self.hsv_sat = hsv_sat
        self.hsv_val = hsv_val
        min_small_light, max_small_light = small_light_flip_range
        if min_small_light < 0 or max_small_light <= min_small_light:
            raise ValueError(
                "small_light_flip_range must be an increasing non-negative "
                "(min, max) pair"
            )

        self.flip_prob = flip_prob
        self.small_light_flip = small_light_flip
        self.small_light_flip_range = (
            float(min_small_light),
            float(max_small_light),
        )
        self.small_light_flip_count = 0

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
            img_info = self.images.get(ann["image_id"])
            if img_info is None:
                continue
            clipped_bbox = self._clip_bbox_to_training_region(
                ann["bbox"],
                img_width=int(img_info["width"]),
                img_height=int(img_info["height"]),
            )
            if clipped_bbox is None:
                continue
            x, y, w, h = clipped_bbox
            if w <= 0 or h <= 0:
                continue
            if ann["category_id"] not in self.cat_to_idx:
                continue
            ann = ann.copy()
            ann["bbox"] = [x, y, w, h]
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

        self.samples: list[tuple[int, bool]] = [
            (img_id, False) for img_id in self.img_ids
        ]
        if self.small_light_flip:
            min_size, max_size = self.small_light_flip_range
            small_flip_ids = [
                img_id
                for img_id in self.img_ids
                if self._has_light_in_size_range(img_id, min_size, max_size)
            ]
            self.samples.extend((img_id, True) for img_id in small_flip_ids)
            self.small_light_flip_count = len(small_flip_ids)
            logger.info(
                "Added runtime small-light horizontal flips for %s split: "
                "%d virtual samples (sqrt(area) in [%.1f, %.1f) px)",
                split,
                self.small_light_flip_count,
                min_size,
                max_size,
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _clip_bbox_to_training_region(
        self,
        bbox: list[float],
        img_width: int,
        img_height: int,
    ) -> list[float] | None:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return None

        crop_h = (
            top_fraction_height(img_height, self.top_crop_fraction)
            if self.top_crop_only
            else img_height
        )
        x1 = min(max(float(x), 0.0), float(img_width))
        y1 = min(max(float(y), 0.0), float(crop_h))
        x2 = min(max(float(x + w), 0.0), float(img_width))
        y2 = min(max(float(y + h), 0.0), float(crop_h))
        clipped_w = x2 - x1
        clipped_h = y2 - y1
        if clipped_w <= 0 or clipped_h <= 0:
            return None
        return [x1, y1, clipped_w, clipped_h]

    def _has_light_in_size_range(
        self,
        img_id: int,
        min_size: float,
        max_size: float,
    ) -> bool:
        for ann in self.anns_by_img.get(img_id, []):
            _, _, w, h = ann["bbox"]
            size = math.sqrt(max(0.0, float(w)) * max(0.0, float(h)))
            if min_size <= size < max_size:
                return True
        return False

    def _load_raw_image(
        self,
        img_id: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load a single image with bboxes and class ids (no augmentation).

        Returns ``(image, bboxes, class_ids)`` where *bboxes* is ``(N, 4)``
        ``[x1, y1, x2, y2]`` and *class_ids* is ``(N,)``.
        """
        info = self.images[img_id]
        img_path = self.image_dir / info["file_name"]

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        if self.top_crop_only:
            image = crop_top_fraction(image, self.top_crop_fraction)

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

    def _load_raw(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img_id, force_hflip = self.samples[idx]
        image, bboxes, class_ids = self._load_raw_image(img_id)
        if force_hflip:
            image, bboxes = horizontal_flip(image, bboxes)
        return image, bboxes, class_ids

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        th, tw = self.input_size
        _, force_hflip = self.samples[idx]

        if not force_hflip and self.augment and random.random() < self.mosaic_prob:
            # --- mosaic: combine 4 random images ---
            img_ids = [self.samples[idx][0]] + [
                random.choice(self.img_ids) for _ in range(3)
            ]
            imgs, bbs, cids = zip(*(self._load_raw_image(i) for i in img_ids))
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
            if (
                not force_hflip
                and len(bboxes) > 0
                and random.random() < self.flip_prob
            ):
                image, bboxes = horizontal_flip(image, bboxes)
            image = hsv_jitter(
                image, self.hsv_hue, self.hsv_sat, self.hsv_val,
            )

        # --- resize to input_size (letterbox) ---
        canvas, r = letterbox_image(image, (th, tw))
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
        batch_size: int = 8,
        lr: float = 1e-3,
        patience: int = 0,
        no_mosaic_epochs: int = 15,
        augment: bool = True,
        positive_images_only: bool = False,
        val_every: int = 1,
        classifier_config: dict | None = None,
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
        * ``input_size`` (list[int], default [960, 960])
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
        input_size = tuple(self.model_config.get("input_size", (960, 960)))
        top_crop_only = bool(
            self.model_config.get("top_crop_only")
            or self.model_config.get("top_third_only", False)
        )
        top_crop_fraction = float(
            self.model_config.get("top_crop_fraction", DEFAULT_TOP_CROP_FRACTION)
        )
        device = self.model_config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        pretrained_ckpt = self.model_config.get("pretrained_ckpt")
        resume_ckpt = self.model_config.get("resume_ckpt")
        exp_name = self.model_config.get("exp_name", "yolox-m")

        # ---- model ----
        exp = get_exp(None, exp_name)
        exp.num_classes = num_classes
        # Keep exp metadata in sync with our requested resolution. The YOLOX
        # model itself is fully convolutional, but exp.input_size / test_size
        # are read by the head/label assigner and by export utilities.
        exp.input_size = tuple(input_size)
        exp.test_size = tuple(input_size)
        # Disable YOLOX's built-in multi-scale jitter. Our image size is a
        # runtime decision, and our augmentations are handled in-dataset.
        exp.random_size = None
        model = exp.get_model()

        resume_data = None
        if resume_ckpt and Path(resume_ckpt).is_file():
            resume_data = _torch_load_checkpoint(resume_ckpt, map_location="cpu")
            src_state = resume_data.get("model", resume_data) if isinstance(resume_data, dict) else resume_data
            try:
                model.load_state_dict(src_state)
                logger.info("Loaded detector model state from resume checkpoint: %s", resume_ckpt)
            except RuntimeError as exc:
                logger.warning(
                    "Strict resume model load failed (%s); loading compatible keys only",
                    exc,
                )
                model_state = model.state_dict()
                compatible = {
                    k: v for k, v in src_state.items()
                    if k in model_state and v.shape == model_state[k].shape
                }
                model.load_state_dict(compatible, strict=False)
                logger.info(
                    "Loaded compatible resume weights from %s (%d/%d keys)",
                    resume_ckpt,
                    len(compatible),
                    len(src_state),
                )
        elif resume_ckpt:
            logger.warning("Resume checkpoint not found: %s", resume_ckpt)
        elif pretrained_ckpt and Path(pretrained_ckpt).is_file():
            ckpt = _torch_load_weights(pretrained_ckpt, map_location="cpu")
            src_state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
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
            mosaic_prob=self.model_config.get("mosaic_prob", 1.0),
            scale_jitter_range=tuple(
                self.model_config.get("scale_jitter_range", (0.5, 1.5))
            ),
            hsv_hue=self.model_config.get("hsv_hue", 0.015),
            hsv_sat=self.model_config.get("hsv_sat", 0.7),
            hsv_val=self.model_config.get("hsv_val", 0.4),
            flip_prob=self.model_config.get("flip_prob", 0.5),
            small_light_flip=self.model_config.get("small_light_flip", False),
            small_light_flip_range=tuple(
                self.model_config.get("small_light_flip_range", (8.0, 24.0))
            ),
            top_crop_only=top_crop_only,
            top_crop_fraction=top_crop_fraction,
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

        val_loss_loader = None
        val_ann_file = dataset_path / "annotations" / "instances_val.json"
        if val_ann_file.is_file():
            val_loss_dataset = _COCODetectionDataset(
                data_dir=str(dataset_path),
                json_file="instances_val.json",
                split="val",
                input_size=input_size,
                max_labels=50,
                augment=False,
                positive_images_only=positive_images_only,
                top_crop_only=top_crop_only,
                top_crop_fraction=top_crop_fraction,
            )
            val_loss_loader = torch.utils.data.DataLoader(
                val_loss_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device != "cpu"),
                drop_last=False,
                persistent_workers=(num_workers > 0),
                prefetch_factor=(4 if num_workers > 0 else None),
            )
            logger.info("Validation loss samples: %d", len(val_loss_dataset))

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
        ema = _ModelEMA(
            model,
            decay=float(self.model_config.get("ema_decay", 0.9998)),
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optionally load the classifier for per-epoch e2e accuracy monitoring.
        classifier = (
            self._load_classifier(classifier_config, str(device))
            if classifier_config
            else None
        )

        # When val_every > 0, best.pth is saved/patience tracked by COCO
        # val mAP averaged over IoU 0.50:0.95.
        # When val_every == 0, fall back to training-loss tracking (old behaviour).
        best_val_map = -1.0
        best_loss = float("inf")
        epochs_no_improve = 0
        start_epoch = 0
        last_completed_epoch = 0
        loss_history: list[dict[str, float | int | None]] = []

        if isinstance(resume_data, dict):
            start_epoch = int(resume_data.get("epoch", 0) or 0)
            last_completed_epoch = start_epoch
            if "best_val_map" in resume_data:
                best_val_map = float(resume_data.get("best_val_map", best_val_map))
            elif resume_data.get("metric_name") == "val_mAP":
                best_val_map = float(resume_data.get("metric_value", best_val_map))
            best_loss = float(resume_data.get("best_loss", best_loss))
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
                logger.info("Restored optimizer state from %s", resume_ckpt)
            else:
                logger.warning("Resume checkpoint has no optimizer state: %s", resume_ckpt)

            if "scheduler" in resume_data:
                scheduler.load_state_dict(resume_data["scheduler"])
                logger.info("Restored scheduler state from %s", resume_ckpt)
                saved_epochs = resume_data.get("training", {}).get("epochs")
                if saved_epochs is not None and int(saved_epochs) != int(epochs):
                    scheduler.T_max = int(epochs)
                    logger.info(
                        "Adjusted scheduler T_max from saved %s to requested %s epochs",
                        saved_epochs,
                        epochs,
                    )
            else:
                logger.warning("Resume checkpoint has no scheduler state: %s", resume_ckpt)

            if "scaler" in resume_data:
                scaler.load_state_dict(resume_data["scaler"])
                logger.info("Restored AMP scaler state from %s", resume_ckpt)

            ema_state = resume_data.get("ema_model") or resume_data.get("ema")
            if ema_state is not None:
                ema.load_state_dict(ema_state)
                logger.info("Restored EMA weights from %s", resume_ckpt)
            else:
                logger.warning("Resume checkpoint has no EMA weights: %s", resume_ckpt)

        no_mosaic_start = epochs - no_mosaic_epochs if no_mosaic_epochs > 0 else epochs
        if augment and start_epoch >= no_mosaic_start and no_mosaic_epochs > 0:
            dataset.mosaic_prob = 0.0

        logger.info(
            "Training: %d epochs, batch=%d, lr=%.1e, device=%s, "
            "%d images, %d iters/epoch, start_epoch=%d, patience=%s, augment=%s, positive_images_only=%s, top_crop_only=%s, top_crop_fraction=%.2f, no_mosaic_after_epoch=%d",
            epochs, batch_size, lr, device, len(dataset), len(loader),
            start_epoch,
            patience if patience > 0 else "off",
            augment,
            positive_images_only,
            top_crop_only,
            top_crop_fraction,
            no_mosaic_start,
        )
        if start_epoch >= epochs:
            logger.warning(
                "Resume checkpoint epoch %d is already at or beyond requested epochs=%d",
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
                "checkpoint_type": "detector",
                "epoch": int(epoch_number),
                "model": _state_dict_to_cpu(model.state_dict()),
                "optimizer": _to_cpu(optimizer.state_dict()),
                "scheduler": _to_cpu(scheduler.state_dict()),
                "scaler": _to_cpu(scaler.state_dict()),
                "ema_model": _state_dict_to_cpu(ema.state_dict()),
                "ema_decay": ema.decay,
                "best_val_map": float(best_val_map),
                "best_loss": float(best_loss),
                "epochs_no_improve": int(epochs_no_improve),
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "loss_history": loss_history,
                "training": {
                    "epochs": int(epochs),
                    "batch_size": int(batch_size),
                    "lr": float(lr),
                    "patience": int(patience),
                    "no_mosaic_epochs": int(no_mosaic_epochs),
                    "augment": bool(augment),
                    "positive_images_only": bool(positive_images_only),
                    "top_crop_only": bool(top_crop_only),
                    "top_crop_fraction": float(top_crop_fraction),
                    "val_every": int(val_every),
                    "input_size": list(input_size),
                    "num_classes": int(num_classes),
                    "exp_name": str(exp_name),
                },
            }
            if training_stopped_at is not None:
                checkpoint["training_stopped_at"] = training_stopped_at
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, path)

        # ---- training loop ----
        for epoch in range(start_epoch, epochs):
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
                ema.update(model)

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

            # ---- validation & early stopping ----
            stop_training = False
            latest_metric_name = "train_loss"
            latest_metric_value = avg
            val_loss = None
            if val_every > 0 and (epoch + 1) % val_every == 0:
                if val_loss_loader is not None:
                    val_loss = self._validation_loss(
                        model=model,
                        loader=val_loss_loader,
                        device=str(device),
                        use_amp=use_amp,
                    )
                val_metrics = self._eval_val(
                    model=model,
                    dataset_path=dataset_path,
                    input_size=input_size,
                    device=str(device),
                    batch_size=batch_size,
                    positive_images_only=positive_images_only,
                    num_classes=int(num_classes),
                    classifier=classifier,
                )
                val_map = val_metrics.get("mAP", 0.0)
                val_map50 = val_metrics.get("mAP_50", 0.0)
                e2e = val_metrics.get("e2e_accuracy")
                log_parts = [
                    f"Epoch {epoch + 1}/{epochs}",
                    f"train_loss={avg:.4f}",
                    f"val_loss={val_loss:.4f}" if val_loss is not None else "val_loss=n/a",
                    f"val_mAP={val_map:.4f}",
                    f"val_mAP_50={val_map50:.4f}",
                ]
                if e2e is not None:
                    log_parts.append(f"e2e_acc={e2e * 100:.1f}%")
                logger.info(" | ".join(log_parts))
                latest_metric_name = "val_mAP"
                latest_metric_value = val_map

                if val_map > best_val_map:
                    best_val_map = val_map
                    epochs_no_improve = 0
                    best_path = self.output_dir / "best.pth"
                    save_checkpoint(best_path, epoch + 1, "val_mAP", val_map)
                    logger.info("New best val_mAP=%.4f - saved %s", val_map, best_path)
                else:
                    epochs_no_improve += 1
                    if patience > 0 and epochs_no_improve >= patience:
                        logger.info(
                            "Early stopping: val mAP did not improve for %d epochs",
                            patience,
                        )
                        stop_training = True
            elif val_every == 0:
                # val_every == 0: use training loss for best.pth and patience.
                if avg < best_loss:
                    best_loss = avg
                    epochs_no_improve = 0
                    best_path = self.output_dir / "best.pth"
                    save_checkpoint(best_path, epoch + 1, "train_loss", avg)
                    logger.info("New best train_loss=%.4f — saved %s", avg, best_path)
                else:
                    epochs_no_improve += 1
                    if patience > 0 and epochs_no_improve >= patience:
                        logger.info(
                            "Early stopping: train loss did not improve for %d epochs",
                            patience,
                        )
                        stop_training = True

            last_completed_epoch = epoch + 1
            loss_history.append(
                {
                    "epoch": int(last_completed_epoch),
                    "train_loss": float(avg),
                    "val_loss": float(val_loss) if val_loss is not None else None,
                }
            )
            write_loss_curve(
                loss_history,
                self.output_dir / "loss_curve.png",
                "Detector Training and Validation Loss",
            )

            latest_path = self.output_dir / "latest.pth"
            save_checkpoint(
                latest_path,
                last_completed_epoch,
                latest_metric_name,
                latest_metric_value,
            )
            logger.info("Saved latest checkpoint -> %s", latest_path)

            # periodic checkpoint
            save_interval = max(1, epochs // 5)
            if (epoch + 1) % save_interval == 0 or epoch + 1 == epochs:
                ckpt_path = self.output_dir / f"epoch_{epoch + 1}.pth"
                save_checkpoint(
                    ckpt_path,
                    epoch + 1,
                    latest_metric_name,
                    latest_metric_value,
                )
                logger.info("Saved %s", ckpt_path)

            if stop_training:
                break

        final_path = self.output_dir / "yolox_tl.pth"
        final_metric = avg if "avg" in locals() else float("nan")
        save_checkpoint(
            final_path,
            last_completed_epoch,
            "train_loss",
            final_metric,
            training_stopped_at=_utc_now_iso(),
        )
        logger.info("Training complete — %s", final_path)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validation_loss(
        model: torch.nn.Module,
        loader: torch.utils.data.DataLoader,
        device: str,
        use_amp: bool,
    ) -> float:
        was_training = model.training
        bn_modes: list[tuple[torch.nn.Module, bool]] = []
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                bn_modes.append((module, module.training))
                module.eval()

        loss_total = 0.0
        total = 0
        try:
            with torch.no_grad():
                for imgs, targets in loader:
                    imgs = imgs.to(device, dtype=torch.float32, non_blocking=True)
                    targets = targets.to(device, non_blocking=True)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(imgs, targets)
                        loss = outputs["total_loss"]
                    batch_size = imgs.size(0)
                    loss_total += loss.item() * batch_size
                    total += batch_size
        finally:
            for module, training in bn_modes:
                module.train(training)
            model.train(was_training)

        return loss_total / total if total > 0 else float("nan")

    @staticmethod
    def _load_classifier(classifier_config: dict, device: str):
        """Load a StateClassifier for e2e monitoring. Returns None on failure."""
        try:
            from adas_perception.traffic_light.config import ClassifierConfig
            from adas_perception.traffic_light.state.classifier import StateClassifier

            cfg = ClassifierConfig(
                **{k: v for k, v in classifier_config.items()
                   if k in {"type", "model_path", "device", "input_size", "classes"}}
            )
            clf = StateClassifier(cfg)
            clf.load_model(cfg.model_path, device)
            logger.info("Loaded classifier for e2e monitoring: %s", cfg.model_path)
            return clf
        except Exception as exc:
            logger.warning("Could not load classifier for e2e monitoring: %s", exc)
            return None

    def _eval_val(
        self,
        model: torch.nn.Module,
        dataset_path: Path,
        input_size: tuple,
        device: str,
        batch_size: int,
        positive_images_only: bool,
        num_classes: int,
        classifier=None,
    ) -> dict:
        """Run detector COCO mAP on the val split, plus optional e2e accuracy."""
        from adas_perception.traffic_light.detector.evaluator import evaluate as det_evaluate

        top_crop_only = bool(
            self.model_config.get("top_crop_only")
            or self.model_config.get("top_third_only", False)
        )
        top_crop_fraction = float(
            self.model_config.get("top_crop_fraction", DEFAULT_TOP_CROP_FRACTION)
        )

        det_metrics = det_evaluate(
            model=model,
            dataset_path=str(dataset_path),
            input_size=input_size,
            conf_threshold=0.01,  # COCO-standard: sweep score thresholds
            nms_threshold=0.65,
            batch_size=batch_size,
            device=device,
            positive_images_only=positive_images_only,
            top_crop_only=top_crop_only,
            top_crop_fraction=top_crop_fraction,
        )

        if classifier is not None:
            e2e = self._compute_e2e_accuracy(
                model=model,
                dataset_path=dataset_path,
                input_size=input_size,
                device=device,
                classifier=classifier,
                positive_images_only=positive_images_only,
                num_classes=num_classes,
                top_crop_only=top_crop_only,
                top_crop_fraction=top_crop_fraction,
                conf_threshold=self.model_config.get("conf_threshold", 0.25),
                nms_threshold=self.model_config.get("nms_threshold", 0.45),
            )
            det_metrics["e2e_accuracy"] = e2e

        model.train()
        return det_metrics

    def _compute_e2e_accuracy(
        self,
        model: torch.nn.Module,
        dataset_path: Path,
        input_size: tuple,
        device: str,
        classifier,
        positive_images_only: bool,
        num_classes: int,
        top_crop_only: bool = False,
        top_crop_fraction: float = DEFAULT_TOP_CROP_FRACTION,
        conf_threshold: float = 0.1,
        nms_threshold: float = 0.45,
        iou_threshold: float = 0.5,
    ) -> float:
        """Per-frame detect+classify e2e accuracy on val (no tracker)."""
        import json
        from collections import defaultdict

        from yolox.utils import postprocess

        _TAG_TO_CLASS: dict[str, int] = {
            "go": 2, "goLeft": 2, "goForward": 2,
            "stop": 0, "stopLeft": 0,
            "warning": 1, "warningLeft": 1,
            "off": 3,
        }
        state_to_idx = {"red": 0, "yellow": 1, "green": 2, "off": 3}

        ann_path = dataset_path / "annotations" / "instances_val.json"
        with open(ann_path, "r", encoding="utf-8") as f:
            coco = json.load(f)

        id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
        anns_by_img: dict = defaultdict(list)
        for ann in coco["annotations"]:
            anns_by_img[ann["image_id"]].append(ann)

        if positive_images_only:
            positive_ids = {
                ann["image_id"] for ann in coco["annotations"]
                if len(ann.get("bbox", [])) == 4
                and ann["bbox"][2] > 0 and ann["bbox"][3] > 0
            }
            eval_ids = [img_id for img_id in sorted(id_to_file) if img_id in positive_ids]
        else:
            eval_ids = sorted(id_to_file.keys())

        image_dir = dataset_path / "val"
        th, tw = input_size
        use_amp = device != "cpu"
        total_gt = 0
        total_correct = 0

        model.eval()
        with torch.no_grad():
            for img_id in eval_ids:
                fname = id_to_file[img_id]
                image = cv2.imread(str(image_dir / fname))
                if image is None:
                    continue

                ih, iw = image.shape[:2]
                detector_image = (
                    crop_top_fraction(image, top_crop_fraction)
                    if top_crop_only
                    else image
                )
                effective_h = detector_image.shape[0]
                canvas, scale = letterbox_image(detector_image, (th, tw))
                img_t = canvas[:, :, ::-1].transpose(2, 0, 1).copy().astype(np.float32)
                img_tensor = torch.from_numpy(img_t).unsqueeze(0).to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    raw = model(img_tensor)
                dets = postprocess(raw, num_classes, conf_threshold, nms_threshold)

                gt_anns = anns_by_img.get(img_id, [])
                gt_boxes: list = []
                gt_classes: list = []
                for ann in gt_anns:
                    tag = ann.get("attributes", {}).get("lisa_tag")
                    if tag not in _TAG_TO_CLASS:
                        continue
                    x, y, w, h = ann["bbox"]
                    if w <= 0 or h <= 0:
                        continue
                    gt_boxes.append([x, y, x + w, y + h])
                    gt_classes.append(_TAG_TO_CLASS[tag])
                total_gt += len(gt_boxes)

                if dets[0] is None or not gt_boxes:
                    continue

                output = dets[0].cpu()
                pred_boxes = output[:, :4] / scale  # xyxy in original image coords
                pred_boxes[:, 0::2].clamp_(0, float(iw))
                pred_boxes[:, 1::2].clamp_(0, float(effective_h))
                gt_arr = np.array(gt_boxes, dtype=np.float32)
                pred_arr = pred_boxes.numpy()

                # Pairwise IoU
                x1 = np.maximum(pred_arr[:, 0:1], gt_arr[:, 0])
                y1 = np.maximum(pred_arr[:, 1:2], gt_arr[:, 1])
                x2 = np.minimum(pred_arr[:, 2:3], gt_arr[:, 2])
                y2 = np.minimum(pred_arr[:, 3:4], gt_arr[:, 3])
                inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
                area_p = (pred_arr[:, 2] - pred_arr[:, 0]) * (pred_arr[:, 3] - pred_arr[:, 1])
                area_g = (gt_arr[:, 2] - gt_arr[:, 0]) * (gt_arr[:, 3] - gt_arr[:, 1])
                union = area_p[:, None] + area_g[None, :] - inter
                ious = inter / np.maximum(union, 1e-6)

                # Greedy match (highest IoU first)
                matched_gt: set = set()
                matched_pred: set = set()
                order = np.dstack(
                    np.unravel_index(np.argsort(-ious, axis=None), ious.shape)
                )[0]
                for p_idx, g_idx in order:
                    p_idx, g_idx = int(p_idx), int(g_idx)
                    if p_idx in matched_pred or g_idx in matched_gt:
                        continue
                    if ious[p_idx, g_idx] < iou_threshold:
                        break
                    matched_pred.add(p_idx)
                    matched_gt.add(g_idx)

                    x1b = max(0, int(pred_boxes[p_idx, 0]))
                    y1b = max(0, int(pred_boxes[p_idx, 1]))
                    x2b = min(iw, int(pred_boxes[p_idx, 2]))
                    y2b = min(ih, int(pred_boxes[p_idx, 3]))
                    roi = image[y1b:y2b, x1b:x2b]
                    if roi.size == 0:
                        continue
                    pred_state, _ = classifier.classify(roi)
                    pred_cls = state_to_idx.get(pred_state.value, -1)
                    if pred_cls == gt_classes[g_idx]:
                        total_correct += 1

        model.train()
        return total_correct / total_gt if total_gt else 0.0
