"""Events demo — synthetic 'video', no files, no cv2, no YOLO.

Car A cruises left→right: it enters a zone, dwells there, crosses a
checkpoint line, then exits the zone. Car B enters late from the right and
closes in on A, firing proximity.

Run:  python examples/events_demo.py
"""

from tracekit import Detection, Detector, Line, Media, Zone
from tracekit.spatial import DwellWatcher


class DemoScene(Detector):
    """Scripted scene: car A cruises, car B enters at frame 8 and closes in."""

    def __init__(self) -> None:
        self._i = 0

    def detect(self, frame) -> list[Detection]:
        i = self._i
        self._i += 1
        ax = 20 + 40 * i                       # car A: 20, 60, 100, ... 380
        dets = [Detection(bbox=(ax - 20, 30, ax + 20, 70), confidence=0.9, label="car")]
        if i >= 7:                             # car B: 480, 440, 400
            bx = 480 - 40 * (i - 7)
            dets.append(Detection(
                bbox=(bx - 20, 30, bx + 20, 70), confidence=0.9, label="car"))
        return dets


media = Media([object() for _ in range(10)])
cars = media.detect(detector=DemoScene(), labels=["car"]).track()

ZONE = Zone(points=[(100, 0), (250, 0), (250, 200), (100, 200)],
            name="waiting area", min_coverage=0.5)
CHECKPOINT = Line(a=(150, 0), b=(150, 200), name="checkpoint")

cars.on_enter(ZONE, lambda e: print(
    f"[enter]      car {e.track.id} in {e.zone.name!r} (coverage {e.coverage:.2f})"))
cars.on_exit(ZONE, lambda e: print(
    f"[exit]       car {e.track.id} left {e.zone.name!r}"))
cars.on_cross(CHECKPOINT, lambda e: print(
    f"[cross]      car {e.track.id} crossed {e.line.name!r} "
    f"(side {e.direction:+d})"))
cars.on_dwell(ZONE, lambda e: print(
    f"[dwell]      car {e.track.id} waiting in {e.zone.name!r}"), min_frames=3)
cars.on_appear(lambda e: print(f"[appear]     car {e.track.id} first seen"))
cars.on_lost(lambda e: print(f"[lost]       car {e.track.id} dropped by tracker"))
cars.on_proximity(100.0, lambda e: print(
    f"[proximity]  cars {e.track.id} & {e.other.id} within "
    f"{e.distance:.0f} px"))

# 'leave' has no convenience shortcut — the producer pattern covers it:
dwell = DwellWatcher(ZONE, min_frames=3)
cars.add_producer(dwell)
dwell.on("leave", lambda e: print(
    f"[leave]      car {e.track.id} stopped dwelling in {e.zone.name!r}"))

media.run()
