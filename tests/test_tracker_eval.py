"""Simulated-detector evaluation of ByteTracker: dropout, false positives,
jitter — counting IDs with the (patched) IdSwitchEvaluator."""

import numpy as np

from tracekit import ByteTracker, IdSwitchEvaluator
from tracekit.core.detection import Detection


def det(cx: float, cy: float, conf: float = 0.9, label: str = "car",
        w: float = 40, h: float = 30) -> Detection:
    return Detection(bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
                     confidence=conf, label=label)


def run_messy(frames: int = 600, n_objects: int = 6, seed: int = 0):
    """n_objects cars move right at constant speed on separate lanes.

    Detector imperfections: 12% dropout, 0.15/false-positive rate, jitter.
    Returns the evaluator.
    """
    rng = np.random.default_rng(seed)
    tracker = ByteTracker()
    ev = IdSwitchEvaluator()
    lane_y = [80.0 + 60.0 * i for i in range(n_objects)]
    for f in range(frames):
        gt, dets = [], []
        for k in range(n_objects):
            x = 30.0 + f * 2.0
            if x > 640:  # exited
                continue
            gt.append((k, (x - 20, lane_y[k] - 15, x + 20, lane_y[k] + 15)))
            # dropout
            if rng.random() < 0.12:
                continue
            jx = x + rng.normal(0, 3.0)   # jitter
            jy = lane_y[k] + rng.normal(0, 2.0)
            dets.append(det(jx, jy, conf=0.85))
        # false positives
        if rng.random() < 0.15:
            dets.append(det(rng.uniform(0, 640), rng.uniform(0, 480), conf=0.7))
        outs = tracker.update(dets)
        ev.update(gt, outs)
    return ev


def test_clean_motion_zero_switches() -> None:
    tracker = ByteTracker()
    ev = IdSwitchEvaluator()
    for f in range(200):
        outs = tracker.update([det(30 + f * 2.0, 100.0)])
        ev.update([(0, (30 + f * 2.0 - 20, 85, 30 + f * 2.0 + 20, 115))], outs)
    assert ev.idsw == 0, f"clean run must be perfect, got frag={ev.fragmentations} merge={ev.merges}"
    assert ev.fn <= 2, "only probation frames may be uncovered"


def test_dropout_does_not_fragment_id() -> None:
    tracker = ByteTracker()
    ev = IdSwitchEvaluator()
    f_step = 2.0
    f = 0
    for i in range(100):
        x = 30 + f * f_step
        if i % 3 == 0:              # every third frame the detector misses
            f += 1
            outs = tracker.update([])
        else:
            outs = tracker.update([det(x, 100.0)])
            f += 1
        ev.update([(0, (x - 20, 85, x + 20, 115))], outs)
    assert ev.fragmentations == 0, "dropout must not rebirth the same car"


def test_false_positive_never_becomes_track_or_merge() -> None:
    tracker = ByteTracker()
    ev = IdSwitchEvaluator()
    for f in range(150):
        dets = [det(30 + f * 2.0, 100.0)]
        if f % 7 == 0:  # recurring blip far from the real car
            dets.append(det(600.0, 400.0, conf=0.75))
        outs = tracker.update(dets)
        ev.update([(0, (30 + f * 2.0 - 20, 85, 30 + f * 2.0 + 20, 115))], outs)
    assert ev.merges == 0 and ev.fragmentations == 0


def test_messy_simulation_switches_are_low() -> None:
    """Full messy run: 12% dropout, jitter, false positives — 6 lanes.

    Claude's patched tracker measured 7 fragmentations / 5 merges on a
    nastier simulator; we assert ours stays in the same order (well below
    the object count).
    """
    ev = run_messy(seed=3)
    assert ev.fragmentations <= 6, f"too fragmented: {ev.fragmentations}"
    assert ev.merges <= 6, f"too many swaps: {ev.merges}"
    assert ev.fn <= 20, f"too many uncovered frames: {ev.fn}"


def test_merges_detect_true_id_swap() -> None:
    """One ID carrying two different GT objects over time is a merge."""
    tracker = ByteTracker()
    ev = IdSwitchEvaluator()
    for f in range(80):
        x = 30 + f * 2.0
        outs = tracker.update([det(x, 100.0)])
        ev.update([(0, (x - 20, 85, x + 20, 115))], outs)
    # GT object 0 leaves; GT object 1 appears at the same continuing spot.
    # The tracker keeps id 1 (id_gts: id1 -> {0, 1}) → one merge.
    for f in range(80, 160):
        x = 30 + f * 2.0
        outs = tracker.update([det(x, 100.0)])
        ev.update([(1, (x - 20, 85, x + 20, 115))], outs)
    assert ev.merges == 1, f"ID hijack must count as a merge, got {ev.merges}"
