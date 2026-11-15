import pytest

from tracekit import Detection, LifecycleWatcher, Media
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


def test_appear_and_lost_fire_on_lifecycle():
    # Track 1 appears, drops (max_missed=0), track 2 appears far away.
    per_frame = [
        [Detection(bbox=(10, 10, 60, 60), confidence=0.9, label="car")],
        [Detection(bbox=(15, 10, 65, 60), confidence=0.9, label="car")],
        [],  # gone → lost
        [Detection(bbox=(300, 10, 350, 60), confidence=0.9, label="car")],  # new id
    ]
    media = Media([object(), object(), object(), object()])
    cars = media.detect(
        detector=ScriptedDetector(per_frame), labels=["car"]
    ).track(tracker=CentroidTracker(max_missed=0))

    events: list[tuple[str, int]] = []
    cars.on_appear(lambda e: events.append(("appear", e.track.id)))
    cars.on_lost(lambda e: events.append(("lost", e.track.id)))
    media.run()
    assert events == [("appear", 1), ("lost", 1), ("appear", 2)]


def test_missed_track_is_not_lost_while_within_grace():
    # Default tracker keeps a track alive through missed frames, so an
    # undetected-but-tracked object must not fire lost.
    per_frame = [
        [Detection(bbox=(10, 10, 60, 60), confidence=0.9, label="car")],
        [],  # missed but still tracked (max_missed=15)
        [],
    ]
    media = Media([object(), object(), object()])
    cars = media.detect(detector=ScriptedDetector(per_frame), labels=["car"]).track()

    events: list[str] = []
    cars.on_lost(lambda e: events.append("lost"))
    media.run()
    assert events == []


def test_lifecyclewatcher_rejects_unknown_events():
    watcher = LifecycleWatcher()
    with pytest.raises(ValueError):
        watcher.on("update", lambda e: None)
