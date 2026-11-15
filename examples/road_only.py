"""Run the model ONLY inside a zone — it never sees the rest of the frame.

The full frame is still displayed; boxes are mapped back to full-frame
coordinates, so tracking/visuals are unchanged.

    python examples/road_only.py
"""

from tracekit import Media, Scope, YOLOAdapter, Zone

from render import VideoSink

SRC = "./samples/car_highway.mp4"

# Main carriageway only (left of the guardrail) — the opposite road on the
# right is never examined by the model.
ROAD = Zone(points=[(20, 30), (545, 30), (545, 356), (20, 356)], name="road")

detector = Scope(
    YOLOAdapter(weights="models/yolov8n.pt", imgsz=480),
    ROAD,
    margin=20,
)

video = Media(SRC)
cars = video.detect(detector=detector, confidence=0.4, labels=["car"]).track()

# Annotated output: full frame, region outlined, tracks drawn as usual.
sink = VideoSink(
    SRC,
    zones=[ROAD],
    out="./samples/car_highway_region_output.mp4",
    overlay=lambda: f"model sees {ROAD.name} only",
)
cars.on_update(lambda e: sink.write(e.tracks, e.frame))

video.run()
sink.close()
