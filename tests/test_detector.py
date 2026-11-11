from tracekit import Detection, Detector


class FakeDetector(Detector):
    def __init__(self, detections: list[Detection] | None = None) -> None:
        self._detections = detections or []

    def detect(self, frame):
        return list(self._detections)


def test_detection_defaults_position_to_bbox_center():
    det = Detection(bbox=(0, 0, 10, 20), confidence=0.9, label="ball")
    assert det.position == (5.0, 10.0)


def test_detection_keeps_explicit_position():
    det = Detection(
        bbox=(0, 0, 10, 20),
        confidence=0.9,
        label="ball",
        position=(1.0, 2.0),
    )
    assert det.position == (1.0, 2.0)


def test_custom_detector_returns_detections():
    expected = [Detection(bbox=(1, 2, 3, 4), confidence=0.5, label="x")]
    detector = FakeDetector(expected)
    assert detector.detect(frame=None) == expected
