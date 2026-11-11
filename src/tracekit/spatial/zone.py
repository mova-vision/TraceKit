"""Zones: named polygon regions for enter/exit event producers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tracekit.core.events import Event, EventProducer

if TYPE_CHECKING:  # pragma: no cover
    from tracekit.core.track import Track

_COVERAGE_GRID = 10  # samples per bbox axis for coverage estimation


@dataclass
class Zone:
    """Named polygon region in image coordinates.

    Used to produce ``enter`` / ``exit`` events. ``min_coverage`` is the
    fraction of a track's bbox area that must lie inside the polygon for the
    track to count as inside: ``0.0`` counts on first touch, ``0.5`` requires
    half the object in, ``1.0`` requires the whole object.
    """

    points: list[tuple[float, float]]
    name: str = "zone"
    min_coverage: float = 0.0

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError("Zone needs at least 3 points to form a polygon")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be between 0.0 and 1.0")

    def contains(self, point: tuple[float, float]) -> bool:
        """Ray-casting point-in-polygon test."""
        px, py = point
        inside = False
        n = len(self.points)
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            if (y1 > py) != (y2 > py):
                xin = (x2 - x1) * (py - y1) / (y2 - y1) + x1
                if px < xin:
                    inside = not inside
        return inside

    def coverage(self, bbox: tuple[float, float, float, float]) -> float:
        """Fraction of the bbox area inside the polygon (0.0 to 1.0).

        Estimated by sampling a grid of points across the bbox — good enough
        for counting semantics, cheap enough to run every frame.
        """
        x1, y1, x2, y2 = bbox
        dx = (x2 - x1) / _COVERAGE_GRID
        dy = (y2 - y1) / _COVERAGE_GRID
        hits = 0
        total = _COVERAGE_GRID * _COVERAGE_GRID
        for i in range(_COVERAGE_GRID):
            for j in range(_COVERAGE_GRID):
                if self.contains((x1 + (i + 0.5) * dx, y1 + (j + 0.5) * dy)):
                    hits += 1
        return hits / total

    def is_inside(self, bbox: tuple[float, float, float, float] | None,
                  position: tuple[float, float]) -> bool:
        """Decide whether a track counts as inside this zone."""
        if bbox is None:
            return self.contains(position) if self.min_coverage <= 0.0 else False
        cov = self.coverage(bbox)
        return cov > 0.0 and cov >= self.min_coverage


class ZoneWatcher(EventProducer):
    """Event producer: watches tracks against one Zone.

    Emits ``enter`` when a track moves inside and ``exit`` when it leaves or
    disappears. Handlers receive a single :class:`Event`.
    """

    def __init__(self, zone: Zone) -> None:
        super().__init__(key=zone)
        self.zone = zone
        self._inside: dict[int, bool] = {}
        self._last: dict[int, "Track"] = {}

    def on(self, event: str, handler) -> None:
        if event not in ("enter", "exit"):
            raise ValueError(f"ZoneWatcher supports 'enter'/'exit', got {event!r}")
        super().on(event, handler)

    def update(self, tracks: list["Track"]) -> None:
        """Evaluate one frame's tracks and fire transition events."""
        present: set[int] = set()
        for track in tracks:
            present.add(track.id)
            self._last[track.id] = track
            inside = self.zone.is_inside(track.bbox, track.position)
            was = self._inside.get(track.id, False)
            if inside and not was:
                self._fire("enter", track)
            elif not inside and was:
                self._fire("exit", track)
            self._inside[track.id] = inside

        # Tracks that vanished (left frame / dropped by tracker) count as out.
        gone = [tid for tid in self._inside if tid not in present]
        for tid in gone:
            if self._inside.pop(tid):
                self._fire("exit", self._last[tid])
            self._last.pop(tid, None)

    def _fire(self, event: str, track: "Track") -> None:
        self.emit(event, Event(
            name=event,
            track=track,
            zone=self.zone,
            coverage=self.zone.coverage(track.bbox) if track.bbox else 0.0,
        ))
