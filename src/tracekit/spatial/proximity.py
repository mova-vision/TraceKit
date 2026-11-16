"""Proximity: pair events when tracks come within a distance."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tracekit.core.events import Event, EventProducer, Handler

if TYPE_CHECKING:  # pragma: no cover
    from tracekit.core.track import Track


class ProximityWatcher(EventProducer):
    """Event producer: fires when two tracks come within ``max_distance``.

    Emits ``proximity`` when a pair *becomes* close (transition-based — no
    repeat while they stay close; it fires again only after they separate
    and get close once more). Handlers receive a single :class:`Event` with
    ``event.track`` / ``event.other`` as the pair and ``event.distance``.
    Pair cost is O(n²) in track count — fine for the dozens a frame
    typically holds.
    """

    def __init__(self, max_distance: float) -> None:
        if max_distance <= 0.0:
            raise ValueError("max_distance must be positive")
        super().__init__(key=max_distance)
        self.max_distance = max_distance
        self._near: set[tuple[int, int]] = set()  # sorted (id_a, id_b) pairs

    def on(self, event: str, handler: Handler) -> None:
        if event != "proximity":
            raise ValueError(f"ProximityWatcher supports 'proximity', got {event!r}")
        super().on(event, handler)

    def update(self, tracks: list["Track"]) -> None:
        """Evaluate one frame's tracks and fire proximity events."""
        near: set[tuple[int, int]] = set()
        for i, a in enumerate(tracks):
            for b in tracks[i + 1:]:
                distance = math.dist(a.position, b.position)
                if distance <= self.max_distance:
                    pair = (a.id, b.id) if a.id < b.id else (b.id, a.id)
                    near.add(pair)
                    if pair not in self._near:
                        first, second = (a, b) if a.id < b.id else (b, a)
                        self._fire(first, second, distance)
        self._near = near

    def _fire(self, a: "Track", b: "Track", distance: float) -> None:
        self.emit("proximity", Event(
            name="proximity",
            track=a,
            other=b,
            distance=distance,
        ))
