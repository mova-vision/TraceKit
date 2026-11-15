"""Count cars crossing one line — terminal count + annotated output video.

    python examples/count_cars.py
"""

import cv2

from render import VideoSink
from tracekit import Line, Media, YOLOAdapter

LINE = Line(a=(40, 180), b=(600, 180), name="checkpoint")  # horizontal, across the road

count = 0

SRC = "./samples/car_highway.mp4"

video = Media(SRC)
cars = video.detect(
    detector=YOLOAdapter(weights="models/yolov8n.pt", imgsz=480),
    confidence=0.4,
    labels=["car"],
).track()


def on_cross(e):
    global count
    count += 1
    print(f"car {e.track.id} crossed -> count = {count}")


cars.on_cross(LINE, on_cross)

# Annotated output: boxes + paths + the line + live counter on screen.
sink = VideoSink(
    SRC,
    out="./samples/car_highway_count_output.mp4",
    overlay=lambda: f"cars crossed: {count}",
)


def draw_and_write(e):
    cv2.line(e.frame, LINE.a, LINE.b, (0, 200, 255), 2)  # orange line
    sink.write(e.tracks, e.frame)


cars.on_update(draw_and_write)

video.run()
sink.close()
print(f"total: {count} cars")
