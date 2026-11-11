"""Traffic zone counting demo.

Counts cars entering and leaving two zones on the highway: +1 on enter,
-1 on exit, one counter per zone. ``min_coverage`` controls how much of the
car must be inside before it counts (0.1 = touching counts, 0.5 = half in).

Adjust the zone points to match your video's geometry.

Usage:
    python examples/traffic_zone.py path/to/traffic.mp4
"""

from __future__ import annotations

import sys

from render import VideoSink

from tracekit import Media, YOLOAdapter, Zone

source = sys.argv[1] if len(sys.argv) > 1 else "traffic.mp4"

# -- configuration ------------------------------------------------------------
ZONE1 = Zone(
    name="zone1",
    points=[(333, 145), (468, 144), (534, 231), (322, 237), (331, 146)],
    min_coverage=0.5,
)
ZONE2 = Zone(
    name="zone2",
    points=[(546, 62), (499, 71), (565, 108), (607, 82), (547, 61)],
    min_coverage=0.5,
)
# ------------------------------------------------------------------------------

detector = YOLOAdapter(weights="models/yolov8n.pt", imgsz=480)
video = Media(source)

cars = video.detect(detector=detector, confidence=0.4, labels=["car"]).track()

counts = {ZONE1.name: 0, ZONE2.name: 0}


def make_on_enter(zone: Zone):
    def on_enter(e) -> None:
        counts[zone.name] += 1
        print(f"enter: car {e.track.id} ({e.coverage:.0%} in) -> {counts}")
    return on_enter


def make_on_exit(zone: Zone):
    def on_exit(e) -> None:
        counts[zone.name] -= 1
        print(f"exit: car {e.track.id} -> {counts}")
    return on_exit


for zone in (ZONE1, ZONE2):
    cars.on_enter(zone, make_on_enter(zone))
    cars.on_exit(zone, make_on_exit(zone))

# Annotated output video: zone outlines + live counters
sink = VideoSink(source=source, zones=[ZONE1, ZONE2],
                 overlay=lambda: f"z1: {counts['zone1']}  z2: {counts['zone2']}")
cars.on_update(lambda e: sink.write(e.tracks, e.frame))

video.run()
sink.close()
print(f"\nFinal counts: {counts}")
