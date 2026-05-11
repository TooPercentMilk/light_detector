"""Data augmentations for the YOLOX detector training pipeline.

Implements mosaic, random scale jitter, random crop/translate, and HSV jitter.
All functions operate on BGR uint8 images and bboxes in ``[x1, y1, x2, y2]`` format.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import cv2
import numpy as np


def mosaic(
    images: List[np.ndarray],
    bboxes_list: List[np.ndarray],
    class_ids_list: List[np.ndarray],
    target_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4-image mosaic augmentation.

    Stitches four images into a single canvas of *target_size* with a
    random meeting point, effectively quadrupling context per sample.

    Parameters
    ----------
    images : list of 4 BGR uint8 images (any size).
    bboxes_list : list of 4 arrays, each ``(N, 4)`` ``[x1, y1, x2, y2]``.
    class_ids_list : list of 4 arrays, each ``(N,)`` class indices.
    target_size : ``(H, W)`` output size.

    Returns
    -------
    canvas : ``(H, W, 3)`` BGR uint8 mosaic image.
    merged_bboxes : ``(M, 4)`` ``[x1, y1, x2, y2]``.
    merged_class_ids : ``(M,)`` class indices.
    """
    th, tw = target_size
    yc = int(random.uniform(th * 0.25, th * 0.75))
    xc = int(random.uniform(tw * 0.25, tw * 0.75))

    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    all_bboxes: list[np.ndarray] = []
    all_class_ids: list[np.ndarray] = []

    for i, (img, bboxes, cids) in enumerate(
        zip(images, bboxes_list, class_ids_list)
    ):
        h, w = img.shape[:2]
        scale = min(th / h, tw / w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h))

        # Origin: where pixel (0,0) of the resized image maps on the canvas.
        if i == 0:  # top-left — bottom-right corner at (xc, yc)
            ox, oy = xc - new_w, yc - new_h
        elif i == 1:  # top-right — bottom-left corner at (xc, yc)
            ox, oy = xc, yc - new_h
        elif i == 2:  # bottom-left — top-right corner at (xc, yc)
            ox, oy = xc - new_w, yc
        else:  # bottom-right — top-left corner at (xc, yc)
            ox, oy = xc, yc

        # Clip source (resized image) and destination (canvas) regions.
        src_x1 = max(0, -ox)
        src_y1 = max(0, -oy)
        src_x2 = min(new_w, tw - ox)
        src_y2 = min(new_h, th - oy)
        dst_x1 = max(0, ox)
        dst_y1 = max(0, oy)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized[
                src_y1:src_y2, src_x1:src_x2
            ]

        # Adjust bboxes: scale then translate by origin offset.
        if len(bboxes) > 0:
            adj = bboxes.copy() * scale
            adj[:, [0, 2]] += ox
            adj[:, [1, 3]] += oy
            all_bboxes.append(adj)
            all_class_ids.append(cids)

    if all_bboxes:
        merged_bboxes = np.concatenate(all_bboxes)
        merged_cids = np.concatenate(all_class_ids)
        # Clip to canvas and drop degenerate boxes.
        merged_bboxes[:, [0, 2]] = np.clip(merged_bboxes[:, [0, 2]], 0, tw)
        merged_bboxes[:, [1, 3]] = np.clip(merged_bboxes[:, [1, 3]], 0, th)
        bw = merged_bboxes[:, 2] - merged_bboxes[:, 0]
        bh = merged_bboxes[:, 3] - merged_bboxes[:, 1]
        keep = (bw > 2) & (bh > 2)
        merged_bboxes = merged_bboxes[keep]
        merged_cids = merged_cids[keep]
    else:
        merged_bboxes = np.zeros((0, 4), dtype=np.float32)
        merged_cids = np.zeros((0,), dtype=np.float32)

    return canvas, merged_bboxes, merged_cids


def random_scale_jitter(
    image: np.ndarray,
    bboxes: np.ndarray,
    scale_range: Tuple[float, float] = (0.5, 1.5),
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly resize an image (and its bboxes) by a uniform scale factor.

    Parameters
    ----------
    image : BGR uint8 image.
    bboxes : ``(N, 4)`` ``[x1, y1, x2, y2]``.
    scale_range : ``(min_scale, max_scale)`` — uniform sampling bounds.

    Returns
    -------
    scaled_image, scaled_bboxes
    """
    s = random.uniform(*scale_range)
    h, w = image.shape[:2]
    new_w, new_h = int(w * s), int(h * s)
    if new_w < 1 or new_h < 1:
        return image, bboxes
    scaled = cv2.resize(image, (new_w, new_h))
    if len(bboxes) > 0:
        bboxes = bboxes.copy() * s
    return scaled, bboxes


def random_crop_translate(
    image: np.ndarray,
    bboxes: np.ndarray,
    target_size: Tuple[int, int],
    min_box_area: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Random crop/translate: place *image* at a random offset on a padded
    canvas of *target_size*, then clip bboxes.

    If the image is larger than the target, a random crop is taken.
    If smaller, it is placed at a random position on a pad-114 canvas.

    Parameters
    ----------
    image : BGR uint8 image (any size).
    bboxes : ``(N, 4)`` ``[x1, y1, x2, y2]`` in image coordinates.
    target_size : ``(H, W)`` desired output size.
    min_box_area : minimum remaining area (pixels²) to keep a box.

    Returns
    -------
    cropped_image : ``(H, W, 3)`` BGR uint8.
    clipped_bboxes : ``(M, 4)`` surviving bboxes.
    """
    th, tw = target_size
    h, w = image.shape[:2]

    # Random offset: how much to shift the image origin on the canvas.
    # Negative offset = crop from the image; positive = pad on the left/top.
    max_dx = max(w - tw, 0)
    max_dy = max(h - th, 0)
    pad_x = max(tw - w, 0)
    pad_y = max(th - h, 0)

    # offset in image coords: how many pixels into the image the crop starts
    crop_x = random.randint(0, max_dx) if max_dx > 0 else 0
    crop_y = random.randint(0, max_dy) if max_dy > 0 else 0
    # where on the canvas the image top-left lands
    place_x = random.randint(0, pad_x) if pad_x > 0 else 0
    place_y = random.randint(0, pad_y) if pad_y > 0 else 0

    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    # Region of image to copy
    src_x2 = min(crop_x + tw - place_x, w)
    src_y2 = min(crop_y + th - place_y, h)
    copy_w = src_x2 - crop_x
    copy_h = src_y2 - crop_y
    if copy_w > 0 and copy_h > 0:
        canvas[place_y : place_y + copy_h, place_x : place_x + copy_w] = image[
            crop_y : crop_y + copy_h, crop_x : crop_x + copy_w
        ]

    # Shift bboxes: subtract crop offset, add placement offset.
    if len(bboxes) > 0:
        shifted = bboxes.copy()
        shifted[:, [0, 2]] += place_x - crop_x
        shifted[:, [1, 3]] += place_y - crop_y
        # Clip to canvas
        shifted[:, [0, 2]] = np.clip(shifted[:, [0, 2]], 0, tw)
        shifted[:, [1, 3]] = np.clip(shifted[:, [1, 3]], 0, th)
        area = (shifted[:, 2] - shifted[:, 0]) * (shifted[:, 3] - shifted[:, 1])
        keep = area >= min_box_area
        bboxes = shifted[keep]
    return canvas, bboxes


def hsv_jitter(
    image: np.ndarray,
    hue_delta: float = 0.015,
    sat_scale: float = 0.7,
    val_scale: float = 0.4,
) -> np.ndarray:
    """Random HSV colour-space jitter.

    Parameters
    ----------
    image : BGR uint8 image.
    hue_delta : max absolute hue shift as fraction of 180 (OpenCV range).
    sat_scale : max multiplicative deviation for saturation.
    val_scale : max multiplicative deviation for value.

    Returns
    -------
    Jittered BGR uint8 image.
    """
    h_gain = random.uniform(-hue_delta, hue_delta) * 180  # in OpenCV H ∈ [0,180)
    s_gain = random.uniform(1 - sat_scale, 1 + sat_scale)
    v_gain = random.uniform(1 - val_scale, 1 + val_scale)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_gain) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_gain, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_gain, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def horizontal_flip(
    image: np.ndarray, bboxes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Flip image and bboxes horizontally."""
    h, w = image.shape[:2]
    image = cv2.flip(image, 1)
    if len(bboxes) > 0:
        bboxes = bboxes.copy()
        x1 = bboxes[:, 0].copy()
        bboxes[:, 0] = w - bboxes[:, 2]
        bboxes[:, 2] = w - x1
    return image, bboxes
