"""User-defined event producers — extending TraceKit without core changes."""

from tracekit import Detection, Event, EventProducer, Media, Zone
from tracekit.core.detector import Detector


class ScriptedDetector(Detector):
    def __init__(self, per_frame: list[list[Detection]]) -> None:
        self._per_frame = per_frame
        self._i = 0

    def detect(self, frame):
        detections = self._per_frame[min(self._i, len(self._per_frame) - 1)]
        self._i += 1
        return detections


class RightMoverWatcher(EventProducer):
    """Custom event: fires when a track moves to the right this frame.

    Demonstrates user logic mixing track paths — no core involvement.
    """

    def update(self, tracks) -> None:
        for track in tracks:
            if len(track.path) >= 2 and track.path[-1][0] > track.path[-2][0]:
                self.emit("moved_right", Event(name="moved_right", track=track))


def test_custom_producer_fires_user_events():
    per_frame = [
        [Detection(bbox=(10, 10, 60, 60), confidence=0.9, label="car")],  # path 1 pt
        [Detection(bbox=(20, 10, 70, 60), confidence=0.9, label="car")],  # moved right
        [Detection(bbox=(25, 10, 75, 60), confidence=0.9, label="car")],  # moved right
    ]
    media = Media([object(), object(), object()])
    cars = media.detect(detector=ScriptedDetector(per_frame)).track()

    watcher = RightMoverWatcher()
    cars.add_producer(watcher)

    fired: list[int] = []
    watcher.on("moved_right", lambda e: fired.append(e.track.id))
    media.run()

    assert fired == [1, 1]


def test_zone_watcher_still_validates_events():
    zone = Zone(points=[(0, 0), (10, 0), (10, 10)])
    media = Media([object()])
    objects = media.detect(
        detector=ScriptedDetector([[Detection(bbox=(1, 1, 5, 5), confidence=0.9, label="car")]])
    ).track()

    from tracekit.spatial.zone import ZoneWatcher

    watcher = ZoneWatcher(zone)
    try:
        watcher.on("bogus", lambda e: None)
        assert False, "should have raised"
    except ValueError:
        pass
