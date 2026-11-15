"""Highway events demo — every shipped event on one real video.

Uses samples/car_highway.mp4 (640x360, ~30s). The road runs vertically:
traffic flows top → bottom and bunches up at the bottom edge. So:

- a HORIZONTAL line across the road is a checkpoint every car must cross
- a zone at the bottom catches cars that stop in the queue (dwell/leave)
- bunched cars fire proximity
- appear / lost fire as cars enter and leave the frame

Run:
    python examples/highway_events.py [path/to/video.mp4]

Terminal shows the events; an annotated copy (boxes, paths, queue zone)
is saved next to the input.
"""

from __future__ import annotations

import sys

import cv2

from render import VideoSink

from tracekit import Line, Media, YOLOAdapter, Zone
from tracekit.spatial import DwellWatcher

source = sys.argv[1] if len(sys.argv) > 1 else "./samples/car_highway.mp4"

# Geometry measured against samples/car_highway.mp4 (traffic flows downward).
CHECKPOINT = Line(a=(40, 180), b=(600, 180), name="checkpoint")
QUEUE = Zone(points=[(330, 290), (470, 290), (470, 356), (330, 356)],
             name="queue", min_coverage=0.3)

video = Media(source)
cars = video.detect(
    detector=YOLOAdapter(weights="models/yolov8n.pt", imgsz=480),
    confidence=0.4,
    labels=["car"],
).track()

crosses = {"down": 0, "up": 0}


def on_cross(e) -> None:
    key = "down" if e.direction < 0 else "up"
    crosses[key] += 1
    print(f"[cross]      car {e.track.id} crossed {e.line.name!r} "
          f"({key}, total {crosses['down'] + crosses['up']})")


cars.on_cross(CHECKPOINT, on_cross)

cars.on_appear(lambda e: print(f"[appear]     car {e.track.id} entered the frame"))
cars.on_lost(lambda e: print(f"[lost]       car {e.track.id} dropped by tracker"))
cars.on_proximity(90.0, lambda e: print(
    f"[proximity]  cars {e.track.id} & {e.other.id} within {e.distance:.0f} px"))

# 'leave' has no convenience shortcut — the custom-producer pattern covers it.
dwell = DwellWatcher(QUEUE, min_frames=20)  # ~0.7 s at 30 fps
cars.add_producer(dwell)
dwell.on("dwell", lambda e: print(
    f"[dwell]      car {e.track.id} waiting in {e.zone.name!r}"))
dwell.on("leave", lambda e: print(
    f"[leave]      car {e.track.id} moved on from {e.zone.name!r}"))

# Annotated video: boxes + paths + queue zone outline + live cross counter.
# VideoSink doesn't draw lines (that's app-side drawing), so the checkpoint
# line is drawn here on each frame before the sink writes it.

def draw_and_write(e) -> None:
    cv2.line(e.frame, CHECKPOINT.a, CHECKPOINT.b, (0, 200, 255), 2)  # orange
    sink.write(e.tracks, e.frame)

sink = VideoSink(
    source,
    out=source.replace(".mp4", "_events_output.mp4"),
    zones=[QUEUE],
    overlay=lambda: f"checkpoint crosses: {crosses['down']} down / {crosses['up']} up",
)
cars.on_update(draw_and_write)

video.run()
sink.close()
print(f"summary: {crosses['down']} down / {crosses['up']} up crossings")
