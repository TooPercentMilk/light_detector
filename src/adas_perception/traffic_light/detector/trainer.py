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
            mosaic_prob=self.model_config.get("mosaic_prob", 1.0),
            scale_jitter_range=tuple(
                self.model_config.get("scale_jitter_range", (0.5, 1.5))
            ),
            hsv_hue=self.model_config.get("hsv_hue", 0.015),
            hsv_sat=self.model_config.get("hsv_sat", 0.7),
            hsv_val=self.model_config.get("hsv_val", 0.4),
            flip_prob=self.model_config.get("flip_prob", 0.5),
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

        # Optionally load the classifier for per-epoch e2e accuracy monitoring.
        classifier = (
            self._load_classifier(classifier_config, str(device))
            if classifier_config
            else None
        )

        # When val_every > 0, best.pth is saved/patience tracked by val mAP_50.
        # When val_every == 0, fall back to training-loss tracking (old behaviour).
        best_val_map50 = -1.0
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

            # ---- validation & early stopping ----
            if val_every > 0 and (epoch + 1) % val_every == 0:
                val_metrics = self._eval_val(
                    model=model,
                    dataset_path=dataset_path,
                    input_size=input_size,
                    device=str(device),
                    batch_size=batch_size,
                    positive_images_only=positive_images_only,
                    classifier=classifier,
                )
                val_map50 = val_metrics.get("mAP_50", 0.0)
                e2e = val_metrics.get("e2e_accuracy")
                log_parts = [
                    f"Epoch {epoch + 1}/{epochs}",
                    f"train_loss={avg:.4f}",
                    f"val_mAP_50={val_map50:.4f}",
                    f"val_mAP={val_metrics.get('mAP', 0.0):.4f}",
                ]
                if e2e is not None:
                    log_parts.append(f"e2e_acc={e2e * 100:.1f}%")
                logger.info(" | ".join(log_parts))

                if val_map50 > best_val_map50:
                    best_val_map50 = val_map50
                    epochs_no_improve = 0
                    best_path = self.output_dir / "best.pth"
                    torch.save({"model": model.state_dict(), "epoch": epoch + 1}, best_path)
                    logger.info("New best val_mAP_50=%.4f — saved %s", val_map50, best_path)
                else:
                    epochs_no_improve += 1
                    if patience > 0 and epochs_no_improve >= patience:
                        logger.info(
                            "Early stopping: val mAP_50 did not improve for %d epochs",
                            patience,
                        )
                        break
            else:
                # val_every == 0: use training loss for best.pth and patience.
                if avg < best_loss:
                    best_loss = avg
                    epochs_no_improve = 0
                    best_path = self.output_dir / "best.pth"
                    torch.save({"model": model.state_dict(), "epoch": epoch + 1}, best_path)
                    logger.info("New best train_loss=%.4f — saved %s", avg, best_path)
                else:
                    epochs_no_improve += 1
                    if patience > 0 and epochs_no_improve >= patience:
                        logger.info(
                            "Early stopping: train loss did not improve for %d epochs",
                            patience,
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

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

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
        classifier=None,
    ) -> dict:
        """Run detector COCO mAP on the val split, plus optional e2e accuracy."""
        from adas_perception.traffic_light.detector.evaluator import evaluate as det_evaluate

        det_metrics = det_evaluate(
            model=model,
            dataset_path=str(dataset_path),
            input_size=input_size,
            conf_threshold=0.01,  # COCO-standard: sweep score thresholds
            nms_threshold=0.65,
            batch_size=batch_size,
            device=device,
            positive_images_only=positive_images_only,
        )

        if classifier is not None:
            e2e = self._compute_e2e_accuracy(
                model=model,
                dataset_path=dataset_path,
                input_size=input_size,
                device=device,
                classifier=classifier,
                positive_images_only=positive_images_only,
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
                scale = min(tw / iw, th / ih)
                new_w, new_h = int(iw * scale), int(ih * scale)
                canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
                canvas[:new_h, :new_w] = cv2.resize(image, (new_w, new_h))
                img_t = canvas[:, :, ::-1].transpose(2, 0, 1).copy().astype(np.float32)
                img_tensor = torch.from_numpy(img_t).unsqueeze(0).to(device)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    raw = model(img_tensor)
                dets = postprocess(raw, 1, conf_threshold, nms_threshold)

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
