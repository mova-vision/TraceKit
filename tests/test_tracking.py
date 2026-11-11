from tracekit import Detection, Media
from tracekit.core.detector import Detector
from tracekit.tracking import CentroidTracker


class ScriptedDetector(Detector):
    def __init__(self, per_frame: list[list[Detection]]) -> None:
        self._per_frame = per_frame
        self._i = 0

    def detect(self, frame):
        detections = self._per_frame[self._i]
        self._i += 1
        return detections


def test_centroid_tracker_keeps_id_and_grows_path():
    tracker = CentroidTracker(max_distance=50)
    d1 = Detection(bbox=(100, 100, 120, 120), confidence=0.9, label="obj")
    d2 = Detection(bbox=(105, 102, 125, 122), confidence=0.9, label="obj")

    tracks1 = tracker.update([d1])
    assert len(tracks1) == 1
    assert tracks1[0].id == 1
    assert tracks1[0].path == [d1.position]

    tracks2 = tracker.update([d2])
    assert len(tracks2) == 1
    assert tracks2[0].id == 1
    assert tracks2[0].path == [d1.position, d2.position]
    assert tracks2[0].position == d2.position


def test_media_pipeline_emits_update_with_tracks():
    frames = [object(), object(), object()]
    per_frame = [
        [Detection(bbox=(10, 10, 20, 20), confidence=1.0, label="a")],
        [Detection(bbox=(12, 11, 22, 21), confidence=1.0, label="a")],
        [Detection(bbox=(14, 12, 24, 22), confidence=1.0, label="a")],
    ]
    media = Media(frames)
    updates: list[list] = []

    objects = media.detect(detector=ScriptedDetector(per_frame)).track()
    objects.on("update", lambda event: updates.append(list(event.tracks)))

    media.run()

    assert len(updates) == 3
    assert updates[0][0].id == updates[1][0].id == updates[2][0].id
    assert len(updates[-1][0].path) == 3
