"""Minimal extensible event emitter."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any


Handler = Callable[..., Any]

if TYPE_CHECKING:  # pragma: no cover
    from tracekit.core.track import Track
    from tracekit.spatial.zone import Zone


@dataclass(slots=True)
class Event:
    """Single payload passed to every event handler.

    ``update`` events carry ``tracks`` (+ ``frame``).
    Zone events (``enter`` / ``exit``) carry ``track``, ``zone`` and the
    fraction of the track's bbox currently inside the zone (``coverage``).
    """

    name: str
    tracks: list["Track"] | None = None
    track: "Track | None" = None
    other: "Track | None" = None
    """Second track involved in pairings (collisions, overtakes, ...)."""
    frame: Any = None
    zone: "Zone | None" = None
    coverage: float | None = None


class EventProducer:
    """Base class for user-defined event producers.

    Subclass and implement :meth:`update` — it runs every frame with all
    current tracks. Mix any logic you like (zones, paths, speeds, other
    producers' state) and fire your own events with :meth:`emit`.
    Register with ``objects.add_producer(...)``.
    """

    def __init__(self, key: Any = None) -> None:
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

    def on(self, event: str, handler: Handler) -> None:
        self._handlers[event].append(handler)

    def off(self, event: str, handler: Handler | None = None) -> None:
        if handler is None:
            self._handlers.pop(event, None)
            return
        handlers = self._handlers.get(event)
        if not handlers:
            return
        self._handlers[event] = [h for h in handlers if h is not handler]

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for handler in list(self._handlers.get(event, [])):
            handler(*args, **kwargs)
