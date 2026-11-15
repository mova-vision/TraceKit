"""Track lifecycle producer: appear / lost events."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tracekit.core.events import Event, EventProducer

if TYPE_CHECKING:  # pragma: no cover
    from tracekit.core.track import Track


class LifecycleWatcher(EventProducer):
    """Event producer: watches track lifecycles. Needs no geometry.

    Emits ``appear`` the first time a track id shows up and ``lost`` when
    the tracker drops it. Note: a track that stops being detected is not
    lost immediately — ``lost`` fires only when it disappears from the
    track list (i.e. after the tracker's ``max_missed`` grace period).
    Handlers receive a single :class:`Event`.
    """

    def __init__(self) -> None:
        super().__init__()
        self._known: set[int] = set()
        self._last: dict[int, "Track"] = {}

    def on(self, event: str, handler) -> None:
        if event not in ("appear", "lost"):
            raise ValueError(f"LifecycleWatcher supports 'appear'/'lost', got {event!r}")
        super().on(event, handler)

    def update(self, tracks: list["Track"]) -> None:
        """Evaluate one frame's tracks and fire lifecycle events."""
        seen: set[int] = set()
        for track in tracks:
            seen.add(track.id)
            self._last[track.id] = track
            if track.id not in self._known:
                self._known.add(track.id)
                self._fire("appear", track)

        # Tracks dropped by the tracker are gone for good.
        for tid in self._known - seen:
            self._fire("lost", self._last.pop(tid))
            self._known.discard(tid)

    def _fire(self, event: str, track: "Track") -> None:
        self.emit(event, Event(name=event, track=track))
