"""File-based tracking demo.

Runs detection → tracking on a video file and writes an annotated copy
(<input>_output.mp4) next to it. All drawing lives in render.py.

Usage:
    pip install -e ".[yolo,opencv]"
    python examples/basic_tracking.py path/to/video.mp4 [person,car]
"""

from __future__ import annotations

import sys

from render import VideoSink

from tracekit import Media, YOLOAdapter

source = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
labels = sys.argv[2].split(",") if len(sys.argv) > 2 else None

detector = YOLOAdapter(weights="models/yolov8n.pt", imgsz=480)
video = Media(source)

people = video.detect(detector=detector, confidence=0.4, labels=labels).track()

sink = VideoSink(source=source)


def on_update(e) -> None:
    if e.tracks:
        print(f"id={e.tracks[0].id} pos={e.tracks[0].position}")
    sink.write(e.tracks, e.frame)


people.on_update(on_update)
video.run()
sink.close()
