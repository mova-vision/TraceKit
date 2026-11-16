"""Lines: tripwire segments for cross event producers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tracekit.core.events import Event, EventProducer, Handler

if TYPE_CHECKING:  # pragma: no cover
    from tracekit.core.track import Track


@dataclass
class Line:
    """Named line segment (defined by two points) in image coordinates.

    ``side()`` uses the infinite line through a→b: the sign of the cross
    product against the line direction — ``+1`` left of a→b, ``-1`` right,
    ``0`` on the line. A track ``cross``es only when the actual crossing
    point lands **on the segment** between ``a`` and ``b`` (checked via
    ``crosses()``), so parallel lanes past the endpoints do not trip it.
    """

    a: tuple[float, float]
    b: tuple[float, float]
    name: str = "line"

    dx: float = field(init=False, repr=False)  # cached b - a, set in __post_init__
    dy: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.dx = self.b[0] - self.a[0]
        self.dy = self.b[1] - self.a[1]
        if self.dx == 0 and self.dy == 0:
            raise ValueError("Line endpoints must differ")

    def side(self, point: tuple[float, float]) -> int:
        """Which side of the line ``point`` is on: ``+1``, ``-1``, or ``0``."""
        z = (point[0] - self.a[0]) * self.dy - (point[1] - self.a[1]) * self.dx
        return 1 if z > 0.0 else -1 if z < 0.0 else 0

    def crosses(self, p1: tuple[float, float], p2: tuple[float, float]) -> bool:
        """Whether the segment ``p1``→``p2`` crosses *this segment*.

        True only when ``p1`` and ``p2`` are on opposite sides of the
        infinite line **and** the intersection point lies between ``a``
        and ``b``.
        """
        return self.crossing_t(p1, p2) is not None

    def crossing_t(self, p1: tuple[float, float],
                   p2: tuple[float, float]) -> float | None:
        """Where ``p1``→``p2`` crosses this segment, or ``None``.

        Returns ``t`` in ``[0, 1]`` along a→b at the intersection, or
        ``None`` if the segments are parallel, do not intersect, or the
        crossing falls outside ``a``→``b``.
        """
        z1 = (p1[0] - self.a[0]) * self.dy - (p1[1] - self.a[1]) * self.dx
        z2 = (p2[0] - self.a[0]) * self.dy - (p2[1] - self.a[1]) * self.dx
        if z1 == z2:  # same side, on the line, or parallel
            return None
        u = z1 / (z1 - z2)  # along p1→p2
        if not 0.0 <= u <= 1.0:
            return None
        cx = p1[0] + u * (p2[0] - p1[0])
        cy = p1[1] + u * (p2[1] - p1[1])
        denom = self.dx * self.dx + self.dy * self.dy
        t = ((cx - self.a[0]) * self.dx + (cy - self.a[1]) * self.dy) / denom
        return t if 0.0 <= t <= 1.0 else None


class LineWatcher(EventProducer):
    """Event producer: watches tracks against one Line.

    Emits ``cross`` when a track moves from one side of the line to the
    other **across the segment** (crossings past the endpoints don't
    count). Handlers receive a single :class:`Event`
    (``event.direction`` is the side moved onto, ``+1`` or ``-1`` —
    filter on it for one-way gates).
    """

    def __init__(self, line: Line, edge_margin: float = 0.0) -> None:
        """``edge_margin``: fraction of the segment length at each end where
        crossings are ignored — objects grazing the line's tip don't count."""
        if not 0.0 <= edge_margin < 0.5:
            raise ValueError("edge_margin must be in [0.0, 0.5)")
        super().__init__(key=line)
        self.line = line
        self.edge_margin = edge_margin
        self._side: dict[int, int] = {}  # track id → last known nonzero side
        self._pos: dict[int, tuple[float, float]] = {}  # id → position at that side

    def on(self, event: str, handler: Handler) -> None:
        if event != "cross":
            raise ValueError(f"LineWatcher supports 'cross', got {event!r}")
        super().on(event, handler)

    def update(self, tracks: list["Track"]) -> None:
        """Evaluate one frame's tracks and fire cross events."""
        present: set[int] = set()
        for track in tracks:
            present.add(track.id)
            side = self.line.side(track.position)
            prev = self._side.get(track.id, 0)
            if side != 0 and prev != 0 and side != prev:
                prev_pos = self._pos.get(track.id)
                if prev_pos is not None:
                    t = self.line.crossing_t(prev_pos, track.position)
                    if t is not None and self.edge_margin <= t <= 1.0 - self.edge_margin:
                        self._fire(track, side)
            if side != 0:
                self._side[track.id] = side
                self._pos[track.id] = track.position

        # Tracks that vanished are forgotten; reappearing on the same side
        # must not fire a spurious cross.
        gone = [tid for tid in self._side if tid not in present]
        for tid in gone:
            self._side.pop(tid)
            self._pos.pop(tid)

    def _fire(self, track: "Track", direction: int) -> None:
        self.emit("cross", Event(
            name="cross",
            track=track,
            line=self.line,
            direction=direction,
        ))
