from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Type

from .base import BaseTracker

if TYPE_CHECKING:
    from ..config import TrackerConfig

# ---------------------------------------------------------------------------
# Registry — maps a short string key to a concrete BaseTracker subclass.
# ---------------------------------------------------------------------------
TRACKER_REGISTRY: Dict[str, Type[BaseTracker]] = {}


def register_tracker(name: str, cls: Type[BaseTracker]) -> None:
    """Add *cls* to the global tracker registry under *name*."""
    TRACKER_REGISTRY[name] = cls


def build_tracker(config: TrackerConfig) -> BaseTracker:
    """Instantiate and return the tracker specified by *config.type*."""
    cls = TRACKER_REGISTRY.get(config.type)
    if cls is None:
        available = ", ".join(sorted(TRACKER_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown tracker type '{config.type}'. "
            f"Available: {available}"
        )
    return cls(config)


# ------ auto-register bundled implementations ------
from .bytetrack_wrapper import ByteTrackWrapper  # noqa: E402

register_tracker("bytetrack", ByteTrackWrapper)
