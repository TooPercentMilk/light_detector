from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def write_loss_curve(
    history: list[dict[str, Any]],
    output_path: Path,
    title: str,
) -> None:
    """Write an overwritten PNG with train and validation loss history."""
    if not history:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not available; skipping loss plot")
        return

    train_points = [
        (int(row["epoch"]), float(row["train_loss"]))
        for row in history
        if _is_finite(row.get("train_loss"))
    ]
    val_points = [
        (int(row["epoch"]), float(row["val_loss"]))
        for row in history
        if _is_finite(row.get("val_loss"))
    ]
    if not train_points and not val_points:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    if train_points:
        epochs, losses = zip(*train_points)
        ax.plot(epochs, losses, marker="o", linewidth=2, label="Training loss")
    if val_points:
        epochs, losses = zip(*val_points)
        ax.plot(epochs, losses, marker="o", linewidth=2, label="Validation loss")

    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote loss plot -> %s", output_path)


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
