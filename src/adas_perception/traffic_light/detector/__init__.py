from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Type

from .base import BaseDetector

if TYPE_CHECKING:
    from ..config import DetectorConfig

# ---------------------------------------------------------------------------
# Registry — maps a short string key to a concrete BaseDetector subclass.
# To register a new detector, import it here and add an entry.
# ---------------------------------------------------------------------------
DETECTOR_REGISTRY: Dict[str, Type[BaseDetector]] = {}


def register_detector(name: str, cls: Type[BaseDetector]) -> None:
    """Add *cls* to the global detector registry under *name*."""
    DETECTOR_REGISTRY[name] = cls


def build_detector(config: DetectorConfig) -> BaseDetector:
    """Instantiate, load, and return the detector specified by *config.type*."""
    cls = DETECTOR_REGISTRY.get(config.type)
    if cls is None:
        available = ", ".join(sorted(DETECTOR_REGISTRY)) or "(none)"
        raise ValueError(
            f"Unknown detector type '{config.type}'. "
            f"Available: {available}"
        )

    detector = cls(config)
    detector.load_model(config.model_path, config.device)
    return detector


# ------ auto-register bundled implementations ------
from .yolox_wrapper import YoloxWrapper  # noqa: E402

register_detector("yolox", YoloxWrapper)
