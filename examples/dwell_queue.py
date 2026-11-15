"""Dwell demo — detect cars STOPPING in a zone. Nothing else.

A car that stays inside the queue zone for ~0.7 s fires 'dwell' (once).
When it finally moves on (or vanishes), it fires 'leave'.

    python examples/dwell_queue.py
"""

from render import VideoSink
from tracekit import Media, YOLOAdapter, Zone
from tracekit.spatial import DwellWatcher

SRC = "./samples/car_highway.mp4"

# Zone picked by user with zone_picker.py on this video (mid-road trapezoid).
# Measured on this video: cars crossing it streak 13-18 frames, cars outside
# it 0 — so min_frames=15 isolates the slow/stalled ones.
QUEUE = Zone(points=[(214, 124), (443, 123), (568, 264), (104, 267)],
             name="queue", min_coverage=0.5)

video = Media(SRC)
cars = video.detect(
    detector=YOLOAdapter(weights="models/yolov8n.pt", imgsz=480),
    confidence=0.4,
    labels=["car"],
).track()

# Threshold measured from this video: 13-18 frame traversal streaks inside
# the zone, 0 outside. 15 = slow/stalled cars only.
watcher = DwellWatcher(QUEUE, min_frames=15)
cars.add_producer(watcher)
watcher.on("dwell", lambda e: print(f"car {e.track.id} is WAITING in the queue"))
watcher.on("leave", lambda e: print(f"car {e.track.id} moved on"))

# Annotated output: the zone outline + live dwell count on screen.
count = {"waiting": 0}
watcher.on("dwell", lambda e: count.__setitem__("waiting", count["waiting"] + 1))
watcher.on("leave", lambda e: count.__setitem__("waiting", count["waiting"] - 1))
sink = VideoSink(SRC, out="./samples/car_highway_dwell_output.mp4", zones=[QUEUE], overlay=lambda: f"waiting: {count['waiting']}")
cars.on_update(lambda e: sink.write(e.tracks, e.frame))

video.run()
sink.close()
