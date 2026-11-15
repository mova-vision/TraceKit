"""Proximity demo — flag cars that get too close to each other. Nothing else.

A pair fires once when it BECOMES close (no repeat while it stays close;
fires again only after separating and closing back in).

    python examples/proximity_alert.py
"""

import cv2

from render import VideoSink
from tracekit import Media, YOLOAdapter

SRC = "./samples/car_highway.mp4"
TOO_CLOSE = 80  # px between two car centers = alert

video = Media(SRC)
cars = video.detect(
    detector=YOLOAdapter(weights="models/yolov8n.pt", imgsz=480),
    confidence=0.4,
    labels=["car"],
).track()

close_pairs: set[tuple[int, int]] = set()


def on_proximity(e) -> None:
    close_pairs.add((e.track.id, e.other.id))
    print(f"cars {e.track.id} & {e.other.id} too close: {e.distance:.0f} px")


cars.on_proximity(TOO_CLOSE, on_proximity)

# Annotated output: red link drawn between each close pair + counter overlay.
sink = VideoSink(SRC, out="./samples/car_highway_proximity_output.mp4",
                 overlay=lambda: f"too close threshold: {TOO_CLOSE} px")


def draw_and_write(e) -> None:
    by_id = {t.id: t for t in e.tracks}
    for a, b in list(close_pairs):
        if a in by_id and b in by_id:      # both still on screen
            pa, pb = by_id[a].position, by_id[b].position
            cv2.line(e.frame, (int(pa[0]), int(pa[1])),
                     (int(pb[0]), int(pb[1])), (0, 0, 255), 2)
        else:
            close_pairs.discard((a, b))    # someone left the frame
    sink.write(e.tracks, e.frame)


cars.on_update(draw_and_write)

video.run()
sink.close()
