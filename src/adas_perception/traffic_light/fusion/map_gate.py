from __future__ import annotations

from typing import Any, List

from ..schemas import TrafficLight


class MapGate:
    """Filter traffic light results using HD-map priors.

    When map data is available, this gate discards lights that fall outside
    regions of interest (e.g. lights belonging to cross-traffic lanes).
    """

    def __init__(self, map_data: Any | None = None) -> None:
        self.map_data = map_data

    def filter(
        self,
        lights: List[TrafficLight],
        vehicle_pose: Any | None = None,
    ) -> List[TrafficLight]:
        """Return only the lights that are relevant given the vehicle pose.

        Without loaded map data this is a no-op pass-through.
        """
        if self.map_data is None:
            return lights

        # TODO: project lights into map frame, check relevance
        raise NotImplementedError("MapGate.filter with map data not yet implemented")
