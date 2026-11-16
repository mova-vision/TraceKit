"""Tests for the ByteTrack-style tracker (IDs must survive gaps, not leak)."""

import numpy as np
import pytest

from tracekit import ByteTracker, ByteTrackerConfig
from tracekit.core.detection import Detection


def det(cx: float, cy: float, conf: float = 0.9, label: str = "car",
        w: float = 40, h: float = 30) -> Detection:
    return Detection(bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                     confidence=conf, label=label)


def moving_line(tracker: ByteTracker, start: float, n: int, step: float = 4.0,
                conf: float = 0.9) -> list[int]:
    """Run n frames of one object moving right; return ids of frames that
    emitted a track (probation frames emit none)."""
    ids: list[int] = []
    for i in range(n):
        outs = tracker.update([det(start + i * step, 100.0, conf=conf)])
        if outs:
            ids.append(outs[0].id)
    return ids


def test_steady_object_keeps_one_id() -> None:
    tracker = ByteTracker(confirm_hits=2)
    ids = moving_line(tracker, 50, 20)
    assert len(set(ids)) == 1, "no ID switches on clean motion"


def test_id_survives_short_gap() -> None:
    tracker = ByteTracker(confirm_hits=2, max_age=10)
    ids = moving_line(tracker, 50, 6)
    gap_ids = []
    for _ in range(4):          # object fully missed for 4 frames
        outs = tracker.update([])
        gap_ids.append(outs[0].id if outs else None)
    after = moving_line(tracker, 50 + 6 * 4 + 16, 4)
    all_ids = [i for i in ids + gap_ids + after if i is not None]
    assert all_ids, "track must appear at some point"
    assert len(set(all_ids)) == 1, "lost track must recover with the SAME id"


def test_new_object_does_not_inherit_old_id() -> None:
    """The original bug: car A leaves, car B behind it reuses A's number."""
    tracker = ByteTracker(confirm_hits=2, max_age=5)
    a_ids = moving_line(tracker, 50, 8)
    # A exits the frame: no detections for longer than max_age.
    for _ in range(8):
        tracker.update([])
    b_ids = moving_line(tracker, 60, 8)
    assert not set(b_ids) & set(a_ids), "different cars must never share an id"


def test_two_parallel_objects_never_swap() -> None:
    tracker = ByteTracker(confirm_hits=2)
    ids_a, ids_b = [], []
    for i in range(15):
        outs = tracker.update([det(50 + i * 4, 100.0), det(50 + i * 3.5, 300.0)])
        by_y = {t.id: t.position[1] for t in outs}
        # after confirmation both are live; match by lane position
        top = [t.id for t in outs if t.position[1] < 200]
        bottom = [t.id for t in outs if t.position[1] > 200]
        if top:
            ids_a.append(top[0])
        if bottom:
            ids_b.append(bottom[0])
    assert len(set(ids_a)) == 1 and len(set(ids_b)) == 1
    assert ids_a[-1] != ids_b[-1]


def test_tentative_false_positive_never_surface() -> None:
    tracker = ByteTracker(ByteTrackerConfig(confirm_hits=3, new_track_thresh=0.6))
    outs = tracker.update([det(100, 100, conf=0.9)])
    assert outs == [], "single-frame blip stays on probation (no public id)"
    tracker.update([det(104, 100, conf=0.9)])
    outs = tracker.update([det(108, 100, conf=0.9)])
    assert len(outs) == 1, "confirmed after confirm_hits frames"
    assert outs[0].path and len(outs[0].path) == 3  # path accumulated


def test_low_confidence_rescue() -> None:
    tracker = ByteTracker(confirm_hits=2)
    moving_line(tracker, 50, 5)
    # blur: confidence drops below hi_thresh but above lo_thresh
    outs = tracker.update([det(70, 100, conf=0.2)])
    ids = [t.id for t in outs]
    assert len(ids) == 1, "weak box still updates the existing track"


def test_weak_detection_never_spawns_track() -> None:
    tracker = ByteTracker(ByteTrackerConfig(confirm_hits=2, new_track_thresh=0.6))
    assert tracker.update([det(100, 100, conf=0.2)]) == []
    assert tracker.update([det(100, 100, conf=0.3)]) == []


def test_class_aware_matching() -> None:
    tracker = ByteTracker(confirm_hits=1)
    ids_car, ids_truck = [], []
    for i in range(6):
        outs = tracker.update([det(50 + i * 4, 100.0, label="car"),
                               det(50 + i * 3, 300.0, label="truck")])
        for t in outs:
            (ids_car if t.label == "car" else ids_truck).append(t.id)
    assert ids_car and ids_truck
    assert set(ids_car).isdisjoint(ids_truck), "labels never share a track"


def test_reset_clears_state() -> None:
    tracker = ByteTracker(confirm_hits=1)
    moving_line(tracker, 50, 3)
    assert len(tracker._tracks) == 1
    tracker.reset()
    assert tracker._tracks == [] and tracker._next_id == 1
    assert moving_line(tracker, 50, 3), "tracks again after reset"


def test_duplicate_spawn_suppressed() -> None:
    tracker = ByteTracker(confirm_hits=1)
    tracker.update([det(100, 100, w=80, h=60)])
    tracker.update([det(110, 105, conf=0.95), det(400, 400, conf=0.95)])
    # confirm the far object on the next frame
    outs = tracker.update([det(400, 400, conf=0.95)])
    assert len(outs) == 2, "heavy-overlap duplicate ignored; distant one spawns"


def test_kalman_prediction_close_to_truth() -> None:
    tracker = ByteTracker(confirm_hits=1)
    for i in range(5):
        outs = tracker.update([det(50 + i * 10, 100.0)])
    t = outs[0]
    expected = 50 + 5 * 10  # next position if velocity is learned
    gap = tracker.update([])  # predict one frame ahead
    assert gap and abs(gap[0].position[0] - expected) < 12


def test_hungarian_assignment_optimal() -> None:
    from tracekit.tracking._hungarian import assign

    cost = np.array([
        [0.9, 0.1, 0.9],
        [0.2, 0.9, 0.9],
        [0.9, 0.9, 0.1],
    ])
    pairs = assign(cost)
    total = sum(cost[i, j] for i, j in pairs)
    assert total == pytest.approx(0.4), "0.1+0.2+0.1 beats the 1.1 diagonal"
    assert sorted(pairs) == [(0, 1), (1, 0), (2, 2)]
