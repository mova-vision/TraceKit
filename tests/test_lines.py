import pytest

from tracekit import Detection, Line, Media
from tracekit.spatial import LineWatcher
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


LINE = Line(a=(100, 0), b=(100, 100), name="tripwire")


def test_line_side_and_crosses():
    assert LINE.side((35, 65)) == -1   # left of a→b
    assert LINE.side((130, 65)) == 1   # right of a→b
    assert LINE.side((100, 50)) == 0   # exactly on the line
    assert LINE.crosses((35, 65), (130, 65))
    assert not LINE.crosses((35, 65), (75, 65))
    assert not LINE.crosses((35, 65), (35, 90))
    # Works on diagonal lines too.
    diag = Line(a=(0, 0), b=(200, 200))
    assert diag.side((10, 0)) != diag.side((0, 10))
    assert diag.crosses((10, 0), (0, 10))


def test_cross_beyond_segment_endpoints_does_not_fire():
    # Car crosses the infinite extension of the line BELOW its endpoints:
    # segment-bounded behavior means this is not a cross.
    per_frame = [
        [Detection(bbox=(10, 140, 60, 190), confidence=0.9, label="car")],   # (35,165)
        [Detection(bbox=(50, 140, 100, 190), confidence=0.9, label="car")],  # (75,165)
        [Detection(bbox=(105, 140, 155, 190), confidence=0.9, label="car")], # (130,165) past x=100
    ]
    media = Media([object(), object(), object()])
    cars = media.detect(detector=ScriptedDetector(per_frame), labels=["car"]).track()

    events: list[int] = []
    cars.on_cross(LINE, lambda e: events.append(e.track.id))
    media.run()
    assert events == []


def test_cross_within_segment_still_fires_far_from_center():
    # Same geometry as above but crossing at y=80, inside the segment
    # (y 0..100) — must fire.
    per_frame = [
        [Detection(bbox=(10, 55, 60, 105), confidence=0.9, label="car")],   # (35,80)
        [Detection(bbox=(50, 55, 100, 105), confidence=0.9, label="car")],  # (75,80)
        [Detection(bbox=(105, 55, 155, 105), confidence=0.9, label="car")], # (130,80) cross +
    ]
    media = Media([object(), object(), object()])
    cars = media.detect(detector=ScriptedDetector(per_frame), labels=["car"]).track()

    events: list[tuple[int, int]] = []
    cars.on_cross(LINE, lambda e: events.append((e.track.id, e.direction)))
    media.run()
    assert events == [(1, 1)]


def test_line_validates_input():
    with pytest.raises(ValueError):
        Line(a=(50, 50), b=(50, 50))


def test_cross_fires_on_side_changes_with_direction():
    # Car moves left → right across the vertical line, then back.
    # Small steps (< max_distance) so the centroid tracker keeps one ID.
    per_frame = [
        [Detection(bbox=(10, 40, 60, 90), confidence=0.9, label="car")],   # center x=35
        [Detection(bbox=(50, 40, 100, 90), confidence=0.9, label="car")],  # x=75
        [Detection(bbox=(90, 40, 140, 90), confidence=0.9, label="car")],  # x=115 → cross +
        [Detection(bbox=(50, 40, 100, 90), confidence=0.9, label="car")],  # x=75 → cross -
        [Detection(bbox=(10, 40, 60, 90), confidence=0.9, label="car")],   # x=35
    ]
    media = Media([object(), object(), object(), object(), object()])
    cars = media.detect(detector=ScriptedDetector(per_frame), labels=["car"]).track()

    events: list[tuple[str, int, int]] = []
    cars.on_cross(LINE, lambda e: events.append((e.track.id, e.direction, e.line.name)))

    media.run()

    assert events == [(1, 1, "tripwire"), (1, -1, "tripwire")]


def test_no_cross_without_side_change():
    per_frame = [
        [Detection(bbox=(10, 40, 60, 90), confidence=0.9, label="car")],
        [Detection(bbox=(30, 40, 80, 90), confidence=0.9, label="car")],
        [Detection(bbox=(50, 40, 100, 90), confidence=0.9, label="car")],  # x=75, still left
    ]
    media = Media([object(), object(), object()])
    cars = media.detect(detector=ScriptedDetector(per_frame), labels=["car"]).track()

    events: list[int] = []
    cars.on_cross(LINE, lambda e: events.append(e.track.id))
    media.run()
    assert events == []


def test_vanished_and_returning_track_does_not_refire():
    # Track sits on the right side, disappears, comes back on the same
    # side: no spurious cross.
    per_frame = [
        [Detection(bbox=(110, 40, 160, 90), confidence=0.9, label="car")],
        [],  # gone
        [Detection(bbox=(120, 40, 170, 90), confidence=0.9, label="car")],  # same side
    ]
    media = Media([object(), object(), object()])
    cars = media.detect(
        detector=ScriptedDetector(per_frame), labels=["car"]
    ).track(tracker=CentroidTracker(max_missed=0))

    events: list[int] = []
    cars.on_cross(LINE, lambda e: events.append(e.track.id))
    media.run()
    assert events == []


def test_reacquired_track_crossing_fires():
    # Track persists through a missed frame (default tracker), then
    # re-acquires on the other side — the position jump is a real crossing.
    # Re-acquisition stays within max_distance=80 so the ID survives.
    per_frame = [
        [Detection(bbox=(110, 40, 160, 90), confidence=0.9, label="car")],  # x=135, side +
        [],                                                                 # missed
        [Detection(bbox=(40, 40, 90, 90), confidence=0.9, label="car")],    # x=65, side -
    ]
    media = Media([object(), object(), object()])
    cars = media.detect(detector=ScriptedDetector(per_frame), labels=["car"]).track()

    events: list[tuple[int, int]] = []
    cars.on_cross(LINE, lambda e: events.append((e.track.id, e.direction)))
    media.run()
    assert events == [(1, -1)]


def test_new_track_appearing_on_other_side_is_not_a_cross():
    # A brand-new track (old one fully dropped) has no prior side, so
    # appearing beyond the line is not a crossing — lifecycle ≠ geometry.
    per_frame = [
        [Detection(bbox=(110, 40, 160, 90), confidence=0.9, label="car")],
        [],
        [Detection(bbox=(20, 40, 70, 90), confidence=0.9, label="car")],
    ]
    media = Media([object(), object(), object()])
    cars = media.detect(
        detector=ScriptedDetector(per_frame), labels=["car"]
    ).track(tracker=CentroidTracker(max_missed=0))

    events: list[int] = []
    cars.on_cross(LINE, lambda e: events.append(e.track.id))
    media.run()
    assert events == []


def test_linewatcher_rejects_unknown_events():
    watcher = LineWatcher(LINE)
    with pytest.raises(ValueError):
        watcher.on("enter", lambda e: None)
