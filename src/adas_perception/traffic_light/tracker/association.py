from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of bounding boxes.

    Parameters
    ----------
    bboxes_a:
        (N, 4) array of boxes in x1y1x2y2 format.
    bboxes_b:
        (M, 4) array of boxes in x1y1x2y2 format.

    Returns
    -------
    (N, M) IoU matrix.
    """
    # Expand to (N, 1, 4) and (1, M, 4) for broadcasting
    a = bboxes_a[:, None, :]  # (N, 1, 4)
    b = bboxes_b[None, :, :]  # (1, M, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h  # (N, M)

    area_a = (bboxes_a[:, 2] - bboxes_a[:, 0]) * (bboxes_a[:, 3] - bboxes_a[:, 1])  # (N,)
    area_b = (bboxes_b[:, 2] - bboxes_b[:, 0]) * (bboxes_b[:, 3] - bboxes_b[:, 1])  # (M,)
    union_area = area_a[:, None] + area_b[None, :] - inter_area  # (N, M)

    return inter_area / np.maximum(union_area, 1e-9)


def linear_assignment(
    cost_matrix: np.ndarray,
    thresh: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve a linear assignment problem on *cost_matrix*.

    Parameters
    ----------
    cost_matrix:
        (N, M) cost matrix (lower is better).
    thresh:
        Maximum admissible cost; assignments above this are rejected.

    Returns
    -------
    matches:
        (K, 2) array of matched (row, col) index pairs.
    unmatched_rows:
        1-D array of unmatched row indices.
    unmatched_cols:
        1-D array of unmatched column indices.
    """
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(cost_matrix.shape[0], dtype=int),
            np.arange(cost_matrix.shape[1], dtype=int),
        )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Reject assignments whose cost exceeds the threshold
    valid = cost_matrix[row_ind, col_ind] <= thresh
    matched_rows = row_ind[valid]
    matched_cols = col_ind[valid]

    matches = np.stack([matched_rows, matched_cols], axis=1)
    unmatched_rows = np.array(
        [r for r in range(cost_matrix.shape[0]) if r not in matched_rows], dtype=int
    )
    unmatched_cols = np.array(
        [c for c in range(cost_matrix.shape[1]) if c not in matched_cols], dtype=int
    )
    return matches, unmatched_rows, unmatched_cols
