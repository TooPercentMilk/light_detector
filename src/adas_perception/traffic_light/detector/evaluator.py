from __future__ import annotations

import contextlib
import io
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def evaluate(
    model: Any,
    dataset_path: str,
    iou_thresholds: list[float] | None = None,
    input_size: tuple[int, int] = (640, 640),
    conf_threshold: float = 0.01,
    nms_threshold: float = 0.65,
    batch_size: int = 8,
    device: str | None = None,
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

    Returns
    -------
    Dict with metrics such as ``mAP``, ``mAP_50``, ``mAP_75``.
    """
    from pycocotools.cocoeval import COCOeval
    from yolox.data import COCODataset, ValTransform
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
    val_dataset = COCODataset(
        data_dir=dataset_path,
        json_file="instances_val.json",
        name="val",
        img_size=input_size,
        preproc=ValTransform(),
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(str(device) != "cpu"),
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
                data_list.append(
                    {
                        "image_id": int(img_id),
                        "category_id": val_dataset.class_ids[cls[j]],
                        "bbox": bboxes_xywh[j].numpy().tolist(),
                        "score": float(scores[j]),
                    }
                )

    # ---- compute COCO metrics ----
    if not data_list:
        logger.warning("No detections produced — all metrics zero")
        return {"mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}

    coco_gt = val_dataset.coco
    _, tmp_path = tempfile.mkstemp(suffix=".json")
    try:
        with open(tmp_path, "w") as f:
            json.dump(data_list, f)
        coco_dt = coco_gt.loadRes(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    if iou_thresholds is not None:
        coco_eval.params.iouThrs = np.array(iou_thresholds)

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
