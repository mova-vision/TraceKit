import pytest

from tracekit import Detection, Media, Zone
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


def test_zone_contains_and_coverage():
    assert ZONE.contains((50, 50))
    assert not ZONE.contains((150, 50))
    assert ZONE.coverage((10, 10, 60, 60)) == 1.0
    assert ZONE.coverage((80, 80, 120, 120)) == 0.25
    assert ZONE.coverage((150, 150, 160, 160)) == 0.0


def test_zone_validates_input():
    with pytest.raises(ValueError):
        Zone(points=[(0, 0), (10, 10)])
    with pytest.raises(ValueError):
        Zone(points=[(0, 0), (10, 10), (20, 20)], min_coverage=1.5)


def test_enter_and_exit_events_fire_on_transitions():
    # Car starts just outside the zone, moves inside, then back out —
    # small steps so the centroid tracker keeps the same ID.
    per_frame = [
        [Detection(bbox=(120, 10, 160, 60), confidence=0.9, label="car")],
        [Detection(bbox=(60, 10, 110, 60), confidence=0.9, label="car")],
        [Detection(bbox=(110, 10, 160, 60), confidence=0.9, label="car")],
    ]
    media = Media([object(), object(), object()])
    cars = media.detect(detector=ScriptedDetector(per_frame), labels=["car"]).track()

    events: list[str] = []
    cars.on_enter(ZONE, lambda e: events.append(("enter", e.track.id, round(e.coverage, 2))))
    cars.on_exit(ZONE, lambda e: events.append(("exit", e.track.id)))

    media.run()

    assert ("enter", 1, 0.8) in events
    assert ("exit", 1) in events
    assert events.index(("enter", 1, 0.8)) < events.index(("exit", 1))


def test_min_coverage_gates_the_enter_event():
    # Bbox only ~10% inside: with min_coverage=0.5 it must NOT count as enter.
    per_frame = [
        [Detection(bbox=(95, 20, 140, 60), confidence=0.9, label="car")],
        [Detection(bbox=(95, 20, 140, 60), confidence=0.9, label="car")],
    ]
    media = Media([object(), object()])
    cars = media.detect(detector=ScriptedDetector(per_frame), labels=["car"]).track()

    entered: list[int] = []
    cars.on_enter(ZONE, lambda e: entered.append(e.track.id))
    media.run()
    assert entered == []

    # Same geometry, min_coverage=0.0 (touching counts):
    touch_zone = Zone(points=ZONE.points, min_coverage=0.0)
    media2 = Media([object(), object()])
    cars2 = media2.detect(detector=ScriptedDetector(per_frame), labels=["car"]).track()
    entered2: list[int] = []
    cars2.on_enter(touch_zone, lambda e: entered2.append(e.track.id))
    media2.run()
    assert entered2 == [1]


def test_disappearing_track_counts_as_exit():
    per_frame = [
        [Detection(bbox=(10, 10, 60, 60), confidence=0.9, label="car")],
        [],  # object gone
    ]
    media = Media([object(), object()])
    cars = media.detect(
        detector=ScriptedDetector(per_frame), labels=["car"]
    ).track(tracker=CentroidTracker(max_missed=0))

    exits: list[int] = []
    cars.on_exit(ZONE, lambda e: exits.append(e.track.id))
    media.run()
    assert exits == [1]


def test_multiple_objects_share_one_decode_and_run_independently():
    """Two .detect() chains on one Media: shared frames, separate state."""
    car_det = ScriptedDetector([
        [Detection(bbox=(10, 10, 60, 60), confidence=0.9, label="car")],
        [Detection(bbox=(15, 10, 65, 60), confidence=0.9, label="car")],
    ])
    person_det = ScriptedDetector([
        [Detection(bbox=(300, 10, 320, 80), confidence=0.9, label="person")],
        [Detection(bbox=(305, 10, 325, 80), confidence=0.9, label="person")],
    ])

    media = Media([object(), object()])
    cars = media.detect(detector=car_det, labels=["car"]).track()
    people = media.detect(detector=person_det, labels=["person"]).track()

    car_events: list = []
    person_events: list = []
    cars.on("update", lambda e: car_events.append(list(e.tracks)))
    people.on("update", lambda e: person_events.append(list(e.tracks)))

    media.run()

    assert len(car_events) == 2
    assert len(person_events) == 2
    # Each chain keeps its own state: labels never leak across objects.
    assert all(t.label == "car" for t in car_events[-1])
    assert all(t.label == "person" for t in person_events[-1])
    assert car_events[-1][0].path != person_events[-1][0].path
