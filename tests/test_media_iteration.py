"""Media iteration: for frame in video, stop(), fps and scale knobs."""

import pytest

from tracekit import Detection, Media
from tracekit.core.detector import Detector


class CounterDetector(Detector):
    """Counts frames seen; returns one detection per frame."""

    def __init__(self) -> None:
        self.seen = 0

    def detect(self, frame):
        self.seen += 1
        return [Detection(bbox=(10, 10, 60, 60), confidence=0.9, label="car")]


FRAMES = [f"frame{i}" for i in range(10)]


def test_media_is_iterable_and_auto_stops():
    media = Media(FRAMES)
    collected = list(media)
    assert collected == FRAMES  # yields every frame, stops at source end


def test_stop_breaks_iteration_mid_source():
    media = Media(FRAMES)
    seen = []
    for frame in media:
        seen.append(frame)
        if len(seen) == 3:
            media.stop()
    assert seen == FRAMES[:3]


def test_fps_decimates_processing_for_file_sources(tmp_path):
    # Probe a real file: car_highway.mp4 is ~30fps; fps=10 → ~1/3 of frames.
    path = tmp_path / "v.mp4"
    path.write_bytes(b"")  # placeholder; real decode test uses samples below


def test_detector_sees_decimated_frames_via_run():
    media = Media("./samples/car_highway.mp4", fps=10)
    det = CounterDetector()
    media.detect(detector=det).track()
    media.run()

    import cv2

    cap = cv2.VideoCapture("./samples/car_highway.mp4")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    assert 0 < det.seen < total  # decimated: fewer frames than the source
    assert abs(det.seen - total / 3) <= 2  # ~1/3 at fps=10 on ~30fps source


def test_scale_resizes_frames():
    import numpy as np

    frames = [np.zeros((360, 640, 3), dtype=np.uint8)]
    media = Media(frames, scale=0.5)
    out = list(media)[0]
    assert out.shape[:2] == (180, 320)


def test_run_and_iteration_are_independent():
    media = Media(FRAMES)
    det = CounterDetector()
    media.detect(detector=det).track()
    media.run()          # consumes the source once
    media.run()          # and can run again (fresh iteration)
    assert det.seen == len(FRAMES) * 2
