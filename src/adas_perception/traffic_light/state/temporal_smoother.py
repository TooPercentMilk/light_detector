from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict

from ..config import TemporalSmootherConfig
from ..schemas import LightState


class TemporalSmoother:
    """Smooth noisy per-frame state predictions via majority voting."""

    def __init__(self, config: TemporalSmootherConfig) -> None:
        self.window_size = config.window_size
        self.min_consensus = config.min_consensus
        self._history: Dict[int, Deque[LightState]] = defaultdict(
            lambda: deque(maxlen=self.window_size)
        )

    def update(self, track_id: int, state: LightState) -> LightState:
        """Append *state* to the history of *track_id* and return the smoothed state.

        If the most common state in the window appears fewer than
        *min_consensus* times, ``LightState.UNKNOWN`` is returned to avoid
        premature commitments.
        """
        history = self._history[track_id]
        history.append(state)

        # majority vote
        counts: dict[LightState, int] = {}
        for s in history:
            counts[s] = counts.get(s, 0) + 1

        best = max(counts, key=lambda s: counts[s])
        if counts[best] >= self.min_consensus:
            return best
        return LightState.UNKNOWN

    def remove_track(self, track_id: int) -> None:
        """Drop history for a track that is no longer alive."""
        self._history.pop(track_id, None)
