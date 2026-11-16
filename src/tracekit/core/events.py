"""Minimal extensible event emitter."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

    from tracekit.core.track import Track
    from tracekit.spatial.line import Line
    from tracekit.spatial.zone import Zone

_E = TypeVar("_E", bound="EventEmitter")

#: Signature every event handler must accept: a single :class:`Event`.
#: Annotating handlers with this means IDEs/type checkers know exactly
#: what ``event`` is inside the handler body.
Handler = Callable[["Event"], None]


@dataclass(slots=True)
class Event:
    """Single payload passed to every event handler.

    ``update`` events carry ``tracks`` (+ ``frame``).
    Zone events (``enter`` / ``exit`` / ``dwell`` / ``leave``) carry
    ``track``, ``zone`` and the fraction of the track's bbox currently
    inside the zone (``coverage``).
    Line events (``cross``) carry ``track``, ``line`` and the side moved
    onto (``direction``, ``+1`` / ``-1``).
    Pair events (``proximity``) carry ``track`` + ``other`` + ``distance``.
    Lifecycle events (``appear`` / ``lost``) carry ``track``.
    """

    name: str
    tracks: list["Track"] | None = None
    track: "Track | None" = None
    other: "Track | None" = None
    """Second track involved in pairings (collisions, overtakes, ...)."""
    frame: "np.ndarray | None" = None
    """BGR frame (numpy HxWx3) for ``update`` events, when a frame source is used."""
    zone: "Zone | None" = None
    line: "Line | None" = None
    coverage: float | None = None
    direction: int | None = None
    """Side moved onto for ``cross`` events (``+1`` / ``-1``)."""
    distance: float | None = None
    """Distance between the pair for ``proximity`` events."""


class EventProducer:
    """Base class for user-defined event producers.

    Subclass and implement :meth:`update` — it runs every frame with all
    current tracks. Mix any logic you like (zones, paths, speeds, other
    producers' state) and fire your own events with :meth:`emit`.
    Register with ``objects.add_producer(...)``.
    """

    def __init__(self, key: object = None) -> None:
        self.key = key
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    @property
    def kind(self) -> str:
        """Registry key component; defaults to the class name."""
        return type(self).__name__

    def on(self, event: str, handler: Handler) -> None:
        self._handlers[event].append(handler)

    def emit(self, event: str, payload: Event) -> None:
        for handler in list(self._handlers.get(event, [])):
            handler(payload)

    def update(self, tracks: list["Track"]) -> None:
        """Called every frame. Implement in subclasses to fire events."""
        raise NotImplementedError


class EventEmitter:
    """String-named event dispatch. New event names require no core changes."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)

    def on(self: _E, event: str, handler: Handler) -> _E:
        """Register ``handler`` for ``event``; returns self for chaining."""
        self._handlers[event].append(handler)
        return self

    def off(self, event: str, handler: Handler | None = None) -> None:
        if handler is None:
            self._handlers.pop(event, None)
            return
        handlers = self._handlers.get(event)
        if not handlers:
            return
        self._handlers[event] = [h for h in handlers if h is not handler]

    def emit(self, event: str, payload: Event) -> None:
        """Dispatch ``payload`` to every handler registered for ``event``."""
        for handler in list(self._handlers.get(event, [])):
            handler(payload)
