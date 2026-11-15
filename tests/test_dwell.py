import pytest

from tracekit import Detection, Media, Zone
from tracekit.spatial import DwellWatcher
from tracekit.core.detector import Detector
from tracekit.tracking.tracker import CentroidTracker


class ScriptedDetector(Detector):
    """Returns pre-scripted detections, regardless of frame."""

    def __init__(self, per_frame: list[list[Detection]]) -> None:
        self._per_frame = per_frame
        self._i = 0

    def detect(self, frame):
        detections = self._per_frame[min(self._i, len(self._per_frame) - 1)]
        self._i += 1
        return detections


ZONE = Zone(points=[(0, 0), (100, 0), (100, 100), (0, 100)], min_coverage=0.5)
INSIDE = Detection(bbox=(10, 10, 60, 60), confidence=0.9, label="car")     # fully in
OUTSIDE = Detection(bbox=(130, 10, 180, 60), confidence=0.9, label="car")  # fully out
IN = [INSIDE]     # one frame's detections: object fully inside the zone
OUT = [OUTSIDE]   # one frame's detections: object fully outside


def test_dwell_fires_after_min_frames_then_leave_on_exit():
    per_frame = [IN, IN, IN, OUT]  # dwell on frame 3, leave on 4
    media = Media([object(), object(), object(), object()])
    cars = media.detect(detector=ScriptedDetector(per_frame), labels=["car"]).track()

    events: list[tuple[str, int]] = []
    cars.on_dwell(ZONE, lambda e: events.append(("dwell", e.track.id)), min_frames=3)
    cars.on_dwell(ZONE, lambda e: events.append(("leave", e.track.id)), min_frames=3)
    media.run()
    assert ("dwell", 1) in events
    assert ("leave", 1) in events
    assert events.index(("dwell", 1)) < events.index(("leave", 1))


def test_on_dwell_shares_one_producer_per_zone_and_min_frames():
    media = Media([object()])
    cars = media.detect(detector=ScriptedDetector([[INSIDE]]), labels=["car"]).track()
    cars.on_dwell(ZONE, lambda e: None, min_frames=3)
    cars.on_dwell(ZONE, lambda e: None, min_frames=3)
    same = [p for k, p in cars._producers.items() if k[0] == "dwell"]
    assert len(same) == 1
    cars.on_dwell(ZONE, lambda e: None, min_frames=5)
    assert len([p for k, p in cars._producers.items() if k[0] == "dwell"]) == 2


def test_dwell_does_not_fire_before_min_frames():
    per_frame = [IN, IN]  # 2 frames < min_frames=3
    media = Media([object(), object()])
    cars = media.detect(detector=ScriptedDetector(per_frame), labels=["car"]).track()

    events: list[int] = []
    cars.on_dwell(ZONE, lambda e: events.append(e.track.id), min_frames=3)
    media.run()
    assert events == []


def test_dwell_fires_once_per_stay():
    per_frame = [IN, IN, IN, IN]  # 4 frames inside, min_frames=2
    media = Media([object(), object(), object(), object()])
    cars = media.detect(detector=ScriptedDetector(per_frame), labels=["car"]).track()

    events: list[str] = []
    cars.on_dwell(ZONE, lambda e: events.append("dwell"), min_frames=2)
    media.run()
    assert events == ["dwell"]


def test_vanishing_dwelling_track_counts_as_leave():
    per_frame = [IN, IN, IN, []]
    media = Media([object(), object(), object(), object()])
    cars = media.detect(
        detector=ScriptedDetector(per_frame), labels=["car"]
    ).track(tracker=CentroidTracker(max_missed=0))

    watcher = DwellWatcher(ZONE, min_frames=2)
    cars.add_producer(watcher)
    events: list[str] = []
    watcher.on("dwell", lambda e: events.append(e.name))
    watcher.on("leave", lambda e: events.append(e.name))
    media.run()
    assert events == ["dwell", "leave"]


def test_reentry_dwells_again():
    # Inside long enough to dwell, out for a frame, inside again: two dwells.
    per_frame = [IN, IN, OUT, IN, IN]
    media = Media([object(), object(), object(), object(), object()])
    cars = media.detect(
        detector=ScriptedDetector(per_frame), labels=["car"]
    ).track(tracker=CentroidTracker(max_missed=0))

    watcher = DwellWatcher(ZONE, min_frames=2)
    cars.add_producer(watcher)
    events: list[str] = []
    watcher.on("dwell", lambda e: events.append(e.name))
    watcher.on("leave", lambda e: events.append(e.name))
    media.run()
    assert events == ["dwell", "leave", "dwell"]


def test_dwellwatcher_validates_input():
    with pytest.raises(ValueError):
        DwellWatcher(ZONE, 0)
    with pytest.raises(ValueError):
        DwellWatcher(ZONE, 2).on("enter", lambda e: None)
