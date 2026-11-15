import pytest

from tracekit import Detection, Media
from tracekit.core.detector import Detector


class ScriptedDetector(Detector):
    """Returns pre-scripted detections, regardless of frame."""

    def __init__(self, per_frame: list[list[Detection]]) -> None:
        self._per_frame = per_frame
        self._i = 0

    def detect(self, frame):
        detections = self._per_frame[min(self._i, len(self._per_frame) - 1)]
        self._i += 1
        return detections


def bbox_at(x):
    """80-wide bbox centered at (x, 50) → position (x, 50)."""
    return (x - 40, 10, x + 40, 90)


# Track 1 parks at x=50; track 2 approaches from the right. Consecutive
# positions stay within the tracker's max_distance=80 (keeps one ID) and
# farther than 80 from track 1 until the intended approach.
# Centers: 260 → 190 → 120 (close, 70) → 120 → 195 (apart) → 125 (close, 75).
SCENARIO = [
    [Detection(bbox=bbox_at(50), confidence=0.9, label="car"),
     Detection(bbox=bbox_at(260), confidence=0.9, label="car")],
    [Detection(bbox=bbox_at(50), confidence=0.9, label="car"),
     Detection(bbox=bbox_at(190), confidence=0.9, label="car")],
    [Detection(bbox=bbox_at(50), confidence=0.9, label="car"),
     Detection(bbox=bbox_at(120), confidence=0.9, label="car")],  # distance 70 ≤ 80
    [Detection(bbox=bbox_at(50), confidence=0.9, label="car"),
     Detection(bbox=bbox_at(120), confidence=0.9, label="car")],
    [Detection(bbox=bbox_at(50), confidence=0.9, label="car"),
     Detection(bbox=bbox_at(195), confidence=0.9, label="car")],  # apart again
    [Detection(bbox=bbox_at(50), confidence=0.9, label="car"),
     Detection(bbox=bbox_at(125), confidence=0.9, label="car")],  # close again
]


def run(max_distance=80.0):
    media = Media([object() for _ in SCENARIO])
    cars = media.detect(detector=ScriptedDetector(list(SCENARIO)), labels=["car"]).track()
    events: list[tuple[int, int, float]] = []
    cars.on_proximity(
        max_distance,
        lambda e: events.append((e.track.id, e.other.id, round(e.distance, 1))),
    )
    media.run()
    return events


def test_proximity_fires_on_becoming_close():
    events = run()
    # Pair (1, 2): fires at distance 70, not again while still close, then
    # once more after separating and closing back in.
    assert events == [(1, 2, 70.0), (1, 2, 75.0)]


def test_proximity_pair_ids_are_sorted():
    events = run()
    for id_a, id_b, _ in events:
        assert id_a < id_b


def test_proximity_never_fires_when_too_far():
    events = run(max_distance=40.0)
    assert events == []


def test_proximity_validates_distance():
    with pytest.raises(ValueError):
        Media([object()]).detect(detector=ScriptedDetector([])).track().on_proximity(
            0.0, lambda e: None
        )
