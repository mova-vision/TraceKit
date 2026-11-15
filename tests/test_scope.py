import numpy as np
import pytest

from tracekit import Detection, Media, Scope, Zone
from tracekit.core.detector import Detector


class SpyDetector(Detector):
    """Records the frames it receives; returns detections in *crop* coords."""

    def __init__(self, per_call: list[list[Detection]] | None = None) -> None:
        self.frames: list = []
        self._per_call = per_call or []

    def detect(self, frame):
        self.frames.append(frame)
        i = min(len(self.frames) - 1, len(self._per_call) - 1)
        return list(self._per_call[i]) if self._per_call else []


def det(x1, y1, x2, y2, label="car", conf=0.9):
    return Detection(bbox=(float(x1), float(y1), float(x2), float(y2)),
                     confidence=conf, label=label)


# 200x100 frame; zone covering the left half (x 10..90).
ZONE = Zone(points=[(10, 10), (90, 10), (90, 90), (10, 90)], name="road")


def test_model_sees_only_the_crop():
    """The inner detector must receive the cropped frame, not the full one."""
    spy = SpyDetector()
    np_frame = np.zeros((100, 200, 3), dtype=np.uint8)
    Scope(spy, ZONE, margin=10).detect(np_frame)

    assert len(spy.frames) == 1
    seen = spy.frames[0]
    # bbox (10,10)-(90,90) + margin 10, clipped to frame: (0,0)-(100,100)
    assert seen.shape == (100, 100, 3)
    assert np_frame[0:100, 0:100].tobytes() == seen.tobytes()  # the right crop


def test_detections_are_remapped_to_full_frame_coords():
    """A box at crop position (5,5) must land at full-frame (0+?, ...) —
    crop origin (0,0) here; use a shifted zone to prove the offset."""
    zone = Zone(points=[(110, 10), (190, 10), (190, 90), (110, 90)])
    # crop rect = (100,0)-(200,100); detection at crop (5,5)-(45,25)
    spy = SpyDetector([[det(5, 5, 45, 25)]])
    out = Scope(spy, zone, margin=10).detect(np.zeros((100, 200, 3)))

    assert len(out) == 1
    assert out[0].bbox == (105.0, 5.0, 145.0, 25.0)  # +100 x-offset


def test_detections_outside_zone_are_dropped():
    """Inner detector returned a box entirely outside the polygon — dropped."""
    # crop rect = (0,0)-(100,100); box at crop (92,5)-(99,25) → full
    # (92,5)-(99,25), right of the zone's x=90 edge → no overlap.
    spy = SpyDetector([[det(92, 5, 99, 25), det(0, 20, 40, 40)]])
    out = Scope(spy, ZONE, margin=10).detect(np.zeros((100, 200, 3)))

    assert len(out) == 1
    assert out[0].bbox == (0.0, 20.0, 40.0, 40.0)  # only the inside one



def test_margin_is_clipped_to_frame_bounds():
    """Zone hugging the frame edge: margin must not push the crop negative."""
    zone = Zone(points=[(0, 0), (30, 0), (30, 30), (0, 30)])
    spy = SpyDetector()
    Scope(spy, zone, margin=50).detect(np.zeros((100, 200, 3)))
    assert spy.frames[0].shape == (80, 80, 3)  # 0..30 + margin 50, clipped


def test_empty_zone_area_returns_nothing():
    zone = Zone(points=[(500, 500), (600, 500), (600, 600), (500, 600)])
    spy = SpyDetector([[det(0, 0, 10, 10)]])
    out = Scope(spy, zone).detect(np.zeros((100, 200, 3)))
    assert out == []


def test_min_coverage_is_respected():
    """min_coverage=1.0 requires the whole box inside the polygon."""
    zone = Zone(points=[(10, 10), (90, 10), (90, 90), (10, 90)], min_coverage=1.0)
    # Box (0,20)-(40,40): only half inside → dropped; (20,20)-(60,40): inside.
    spy = SpyDetector([[det(0, 20, 40, 40), det(20, 20, 60, 40)]])
    out = Scope(spy, zone, margin=10).detect(np.zeros((100, 200, 3)))
    assert [d.bbox for d in out] == [(20.0, 20.0, 60.0, 40.0)]


def test_works_through_media_pipeline():
    """End-to-end: tracks carry full-frame positions even though the model
    only ever saw crops."""
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    # One "object" moving right, detections given in crop coords (0..100).
    per_frame = [
        [det(10, 40, 50, 60)],
        [det(14, 40, 54, 60)],
        [det(18, 40, 58, 60)],
    ]
    spy = SpyDetector(per_frame)
    media = Media([frame.copy() for _ in range(3)])
    cars = media.detect(
        detector=Scope(spy, ZONE, margin=10), labels=["car"]
    ).track()

    positions: list[tuple] = []
    cars.on_update(lambda e: positions.append(
        tuple(t.position for t in e.tracks)))
    media.run()

    # Full-frame x = crop x + 0 (crop origin x=0) — but positions must be
    # sane full-frame coords, in order.
    assert len(positions) == 3
    xs = [p[0][0] for p in positions]
    assert xs == sorted(xs) and all(0 <= x < 200 for x in xs)
    assert spy.frames[0].shape == (100, 100, 3)  # model never saw full frame


def test_validates_margin():
    with pytest.raises(ValueError):
        Scope(SpyDetector(), ZONE, margin=-1)


def test_multiple_zones_act_as_union():
    """Two disjoint zones: the model runs once on the combined crop, and a
    detection inside either zone is kept."""
    zone_a = Zone(points=[(10, 10), (90, 10), (90, 90), (10, 90)])
    zone_b = Zone(points=[(130, 10), (190, 10), (190, 90), (130, 90)])
    # Combined crop = (0,0)-(200,100); detections in crop coords land in
    # zone_a and zone_b respectively after remapping.
    spy = SpyDetector([[det(30, 20, 70, 40), det(140, 20, 180, 40),
                        det(95, 20, 125, 40)]])  # middle one in neither
    out = Scope(spy, [zone_a, zone_b], margin=10).detect(
        np.zeros((100, 200, 3)))

    assert spy.frames[0].shape == (100, 200, 3)  # one run, combined crop
    assert [d.bbox for d in out] == [
        (30.0, 20.0, 70.0, 40.0),
        (140.0, 20.0, 180.0, 40.0),
    ]


def test_empty_zone_list_rejected():
    with pytest.raises(ValueError):
        Scope(SpyDetector(), [])
