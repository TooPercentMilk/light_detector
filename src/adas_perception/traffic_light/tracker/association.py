from __future__ import annotations

from typing import Tuple

import numpy as np


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
    # TODO: vectorised IoU computation
    raise NotImplementedError("iou_batch not yet implemented")


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
    # TODO: Hungarian algorithm via scipy.optimize.linear_sum_assignment
    raise NotImplementedError("linear_assignment not yet implemented")
