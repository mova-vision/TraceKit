"""Two counting lines — one per carriageway, each with its own counter.

    python examples/two_lines.py
"""

import cv2

from render import VideoSink
from tracekit import Line, Media, YOLOAdapter

SRC = "./samples/car_highway.mp4"

LANE1 = Line(a=(193, 150), b=(476, 150), name="lane1")  # main carriageway
LANE2 = Line(a=(529, 80), b=(596, 70), name="lane2")    # opposite carriageway

counts = {LANE1.name: 0, LANE2.name: 0}

video = Media(SRC)
cars = video.detect(
    detector=YOLOAdapter(weights="models/yolov8n.pt", imgsz=480),
    confidence=0.4,
    labels=["car"],
).track()


def make_on_cross(line: Line):
    def on_cross(e):
        counts[line.name] += 1
        d = "down" if e.direction == -1 else "up"
        print(f"car {e.track.id} crossed {line.name} going {d} "
              f"({e.direction:+d}) -> {counts}")
    return on_cross


cars.on_cross(LANE1, make_on_cross(LANE1))
cars.on_cross(LANE2, make_on_cross(LANE2))

# Annotated output: both lines drawn + live per-lane counters.
sink = VideoSink(
    SRC,
    out="./samples/car_highway_two_lines_output.mp4",
    overlay=lambda: f"lane1: {counts['lane1']}  lane2: {counts['lane2']}",
)


def draw_and_write(e):
    cv2.line(e.frame, LANE1.a, LANE1.b, (0, 200, 255), 2)  # orange
    cv2.line(e.frame, LANE2.a, LANE2.b, (255, 200, 0), 2)  # blue
    sink.write(e.tracks, e.frame)


cars.on_update(draw_and_write)

video.run()
sink.close()
print(f"final: {counts}")
