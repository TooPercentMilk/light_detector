from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def _positive_image_ids(coco) -> set[int]:
    """Return image IDs that contain at least one valid annotation."""
    valid_cat_ids = set(coco.getCatIds())
    image_ids: set[int] = set()

    for ann in coco.dataset.get("annotations", []):
        bbox = ann.get("bbox", [])
        if len(bbox) < 4 or bbox[2] <= 0 or bbox[3] <= 0:
            continue
        if ann.get("category_id") not in valid_cat_ids:
            continue
        image_ids.add(ann["image_id"])

    return image_ids


# ------------------------------------------------------------------
# Lightweight COCO val dataset (no dependency on yolox.data.COCODataset)
# ------------------------------------------------------------------

class _COCOValDataset(Dataset):
    """Minimal COCO dataset for evaluation that applies letterbox resizing.

    Each item returns ``(image_tensor, None, (orig_h, orig_w), image_id)``.
    """

    def __init__(
        self,
        data_dir: str,
        json_file: str,
        name: str = "val",
        img_size: Tuple[int, int] = (640, 640),
        image_ids: list[int] | None = None,
        positive_images_only: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / name
        self.img_size = img_size  # (H, W)

        from pycocotools.coco import COCO

        ann_path = self.data_dir / "annotations" / json_file
        self.coco = COCO(str(ann_path))
        all_img_ids: List[int] = sorted(self.coco.getImgIds())
        if positive_images_only:
            positive_ids = _positive_image_ids(self.coco)
            filtered_img_ids = [img_id for img_id in all_img_ids if img_id in positive_ids]
            logger.info(
                "Using positive-only %s split for evaluation: %d -> %d images",
                name,
                len(all_img_ids),
                len(filtered_img_ids),
            )
            all_img_ids = filtered_img_ids

        if image_ids is not None:
            image_id_set = set(image_ids)
            all_img_ids = [img_id for img_id in all_img_ids if img_id in image_id_set]

        self.img_ids = all_img_ids
        self.class_ids: List[int] = sorted(self.coco.getCatIds())

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = self.image_dir / img_info["file_name"]

        image = cv2.imread(str(img_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {img_path}")

        ih, iw = image.shape[:2]
        th, tw = self.img_size
        r = min(tw / iw, th / ih)
        new_w, new_h = int(iw * r), int(ih * r)
        resized = cv2.resize(image, (new_w, new_h))

        # Letterbox: paste onto 114-padded canvas
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        canvas[:new_h, :new_w] = resized

        # HWC BGR → CHW RGB float32 (YOLOX convention: no /255)
        img_t = canvas[:, :, ::-1].transpose(2, 0, 1).copy()
        img_t = np.ascontiguousarray(img_t, dtype=np.float32)

        return (
            torch.from_numpy(img_t),
            None,
            (ih, iw),
            img_id,
        )


def _collate_fn(batch):
    imgs, _, info_imgs, ids = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    orig_hs = [info[0] for info in info_imgs]
    orig_ws = [info[1] for info in info_imgs]
    return imgs, None, (orig_hs, orig_ws), list(ids)


def evaluate(
    model: Any,
    dataset_path: str,
    iou_thresholds: list[float] | None = None,
    input_size: tuple[int, int] = (640, 640),
    conf_threshold: float = 0.01,
    nms_threshold: float = 0.65,
    batch_size: int = 8,
    device: str | None = None,
    image_ids: list[int] | None = None,
    positive_images_only: bool = False,
) -> Dict[str, float]:
    """Evaluate a detector on a COCO-format dataset.

    Parameters
    ----------
    model:
        A loaded YOLOX ``torch.nn.Module``.
    dataset_path:
        Root of a COCO-format dataset containing
        ``annotations/instances_val.json`` and a ``val/`` image folder.
    iou_thresholds:
        Custom IoU thresholds for AP computation
        (default: COCO standard 0.50:0.05:0.95).
    input_size:
        (H, W) used for inference pre-processing.
    conf_threshold, nms_threshold:
        Post-processing confidence and NMS thresholds.
    batch_size:
        Evaluation batch size.
    device:
        Inference device; auto-detected from *model* when ``None``.
    image_ids:
        If given, restrict COCO evaluation to this subset of image IDs.

    Returns
    -------
    Dict with metrics such as ``mAP``, ``mAP_50``, ``mAP_75``.
    """
    from pycocotools.cocoeval import COCOeval
    from yolox.utils import postprocess

    if model is None:
        logger.warning("model is None — returning zero metrics")
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

    dataset_path = str(Path(dataset_path).resolve())

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    # ---- validation data ----
    val_dataset = _COCOValDataset(
        data_dir=dataset_path,
        json_file="instances_val.json",
        name="val",
        img_size=input_size,
        image_ids=image_ids,
        positive_images_only=positive_images_only,
    )
    if len(val_dataset) == 0:
        logger.warning("Validation dataset is empty after image filtering")
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(str(device) != "cpu"),
        collate_fn=_collate_fn,
    )

    model.eval()
    num_classes = len(val_dataset.class_ids)
    data_list: list[dict] = []

    for imgs, _, info_imgs, ids in val_loader:
        with torch.no_grad():
            outputs = model(imgs.to(device))

        outputs = postprocess(outputs, num_classes, conf_threshold, nms_threshold)

        for output, img_h, img_w, img_id in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()
            bboxes = output[:, 0:4]

            # undo letterbox scaling
            scale = min(
                input_size[0] / float(img_h), input_size[1] / float(img_w)
            )
            bboxes /= scale

            # xyxy → xywh (COCO format)
            bboxes_xywh = bboxes.clone()
            bboxes_xywh[:, 2] -= bboxes_xywh[:, 0]
            bboxes_xywh[:, 3] -= bboxes_xywh[:, 1]

            scores = (output[:, 4] * output[:, 5]).numpy()
            cls = output[:, 6].numpy().astype(int)

            for j in range(len(bboxes_xywh)):
                cat_idx = cls[j]
                if cat_idx < len(val_dataset.class_ids):
                    cat_id = val_dataset.class_ids[cat_idx]
                else:
                    continue
                data_list.append(
                    {
                        "image_id": int(img_id),
                        "category_id": cat_id,
                        "bbox": bboxes_xywh[j].numpy().tolist(),
                        "score": float(scores[j]),
                    }
                )

    # ---- compute COCO metrics ----
    if not data_list:
        logger.warning("No detections produced — all metrics zero")
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

    coco_gt = val_dataset.coco
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data_list, f)
        coco_dt = coco_gt.loadRes(tmp_path)
    finally:
        try:
            Path(tmp_path).unlink()
        except OSError:
            pass

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    if iou_thresholds is not None:
        coco_eval.params.iouThrs = np.array(iou_thresholds)
    coco_eval.params.imgIds = list(val_dataset.img_ids)

    coco_eval.evaluate()
    coco_eval.accumulate()

    redirect = io.StringIO()
    with contextlib.redirect_stdout(redirect):
        coco_eval.summarize()
    logger.info("\n%s", redirect.getvalue())

    stats = coco_eval.stats
    return {
        "mAP": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
    }
