"""Internal detect → track → event pipeline."""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

from tracekit.core.detector import Detector
from tracekit.core.detection import Detection
from tracekit.core.events import EventEmitter, Event, EventProducer, Handler
from tracekit.core.lifecycle import LifecycleWatcher
from tracekit.core.track import Track
from tracekit.spatial.line import Line, LineWatcher
from tracekit.spatial.proximity import ProximityWatcher
from tracekit.spatial.zone import DwellWatcher, Zone, ZoneWatcher
from tracekit.tracking.tracker import CentroidTracker


class TrackedObjects(EventEmitter):
    """Fluent handle returned by ``Media.detect(...).track()``.

    Events stay generic: each kind (zone, line, collision, loss, ...) is a
    *producer* object registered once in ``_producers``. New event types add
    a producer class plus one ``on_*`` method — no core rewrites.
    """

    def __init__(self) -> None:
        super().__init__()
        self._tracks: list[Track] = []
        self._producers: dict[tuple[str, Any], Any] = {}

    @property
    def tracks(self) -> list[Track]:
        return list(self._tracks)

    def on(self, event: str, handler: Handler) -> TrackedObjects:
        super().on(event, handler)
        return self

    def on_update(self, handler: Handler) -> TrackedObjects:
        """Fire ``handler`` every processed frame with all current tracks."""
        return self.on("update", handler)

    def on_enter(self, zone: Zone, handler: Handler) -> TrackedObjects:
        """Fire ``handler`` when a track enters ``zone``."""
        self._producer(("zone", id(zone)), lambda: ZoneWatcher(zone)).on("enter", handler)
        return self

    def on_exit(self, zone: Zone, handler: Handler) -> TrackedObjects:
        """Fire ``handler`` when a track leaves ``zone`` (or disappears)."""
        self._producer(("zone", id(zone)), lambda: ZoneWatcher(zone)).on("exit", handler)
        return self

    def on_cross(self, line: Line, handler: Handler) -> TrackedObjects:
        """Fire ``handler`` when a track crosses ``line`` (either direction).

        Filter inside the handler on ``event.direction`` for a one-way gate.
        """
        self._producer(("line", id(line)), lambda: LineWatcher(line)).on("cross", handler)
        return self

    def on_dwell(self, zone: Zone, handler: Handler, *,
                 min_frames: int = 30) -> TrackedObjects:
        """Fire ``handler`` after a track stays ``min_frames`` frames in
        ``zone``; ``on_dwell(zone, h).on("leave", h2)`` catches it leaving."""
        key = ("dwell", (id(zone), min_frames))
        self._producer(key, lambda: DwellWatcher(zone, min_frames)).on("dwell", handler)
        return self

    def on_appear(self, handler: Handler) -> TrackedObjects:
        """Fire ``handler`` the first time a track id shows up."""
        self._producer(("lifecycle", None), LifecycleWatcher).on("appear", handler)
        return self

    def on_lost(self, handler: Handler) -> TrackedObjects:
        """Fire ``handler`` when the tracker drops a track id."""
        self._producer(("lifecycle", None), LifecycleWatcher).on("lost", handler)
        return self

    def on_proximity(self, max_distance: float, handler: Handler) -> TrackedObjects:
        """Fire ``handler`` when two tracks come within ``max_distance``.

        Fires on *becoming* close; the pair distance is on ``event.distance``.
        """
        self._producer(
            ("proximity", max_distance),
            lambda: ProximityWatcher(max_distance),
        ).on("proximity", handler)
        return self

    def _producer(self, key: tuple[str, Any], factory: Callable[[], Any]) -> Any:
        """Find-or-create the producer for ``key`` — one per geometry."""
        if key not in self._producers:
            self._producers[key] = factory()
        return self._producers[key]

    def add_producer(self, producer: Any) -> TrackedObjects:
        """Register a custom :class:`EventProducer` (user-defined events).

        The producer's ``update(tracks)`` runs every frame; handlers are
        attached directly on it (``producer.on("my_event", fn)``).
        """
        self._producers[(producer.kind, id(producer.key))] = producer
        return self

    def _set_tracks(self, tracks: list[Track]) -> None:
        self._tracks = tracks
        for producer in self._producers.values():
            producer.update(tracks)


class Pipeline:
    """One detect → filter → track chain, driven by ``Media``."""

    def __init__(
        self,
        detector: Detector,
        *,
        confidence: float | None = None,
        labels: list[str] | None = None,
        tracker: CentroidTracker | None = None,
    ) -> None:
        self._detector = detector
        self._confidence = confidence
        self._labels = set(labels) if labels else None
        self._tracker = tracker or CentroidTracker()
        self.objects = TrackedObjects()

    def process_frame(self, frame: Any) -> list[Track]:
        """Detect and ingest a single frame (streaming demos)."""
        return self.ingest(frame, self._detector.detect(frame))

    def ingest(self, frame: Any, detections: list[Detection]) -> list[Track]:
        """Filter already-computed detections, track, and emit events."""
        tracks = self._tracker.update(self._filter(detections))
        self.objects._set_tracks(tracks)
        self.objects.emit("update", Event(name="update", tracks=tracks, frame=frame))
        return tracks

    def _filter(self, detections: list[Detection]) -> list[Detection]:
        result: list[Detection] = []
        for det in detections:
            if self._confidence is not None and det.confidence < self._confidence:
                continue
            if self._labels is not None and det.label not in self._labels:
                continue
            result.append(det)
        return result


class DetectionStage:
    """Intermediate fluent object after ``.detect()`` before ``.track()``."""

    def __init__(
        self,
        media: Any,
        detector: Detector,
        *,
        confidence: float | None = None,
        labels: list[str] | None = None,
    ) -> None:
        self._media = media
        self._detector = detector
        self._confidence = confidence
        self._labels = labels

    def track(self, tracker: CentroidTracker | None = None) -> TrackedObjects:
        pipeline = Pipeline(
            detector=self._detector,
            confidence=self._confidence,
            labels=self._labels,
            tracker=tracker,
        )
        self._media._add_pipeline(pipeline)
        return pipeline.objects
