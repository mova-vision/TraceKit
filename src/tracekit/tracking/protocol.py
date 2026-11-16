"""The tracker contract every tracking backend must satisfy.

``Tracker`` is a structural (duck-typed) protocol: a class doesn't need to
inherit from it — it just needs ``update()`` and ``reset()`` with these
signatures. Pass any conforming class to ``.track(tracker=...)`` and both
the runtime and your IDE accept it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover
    from tracekit.core.detection import Detection
    from tracekit.core.track import Track


@runtime_checkable
class Tracker(Protocol):
    """Stateful multi-object tracker over :class:`Detection` lists.

    Implementations own ID assignment; they receive per-frame detections
    (already normalized to :class:`Detection`) and return the live tracks,
    mutating each :class:`Track` in place with positions and paths.
    """

    def update(self, detections: list["Detection"]) -> list["Track"]:
        """Ingest one frame's detections and return the current tracks."""
        ...

    def reset(self) -> None:
        """Drop all state (used between videos / on source changes)."""
        ...
