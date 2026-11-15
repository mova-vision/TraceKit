"""Scope demo — two videos from one run:

  1. FULL view:    whole frame, hatched "model doesn't look here" area
  2. REGION view:  everything outside the scope blacked out + hatch lines

The gray angled parallel lines mark what the model never sees; the white
border is the scope itself. Drawing is app-side — TraceKit only supplies
tracks + frames.

    python examples/scope_two_views.py
"""

import cv2
import numpy as np

from render import VideoSink
from tracekit import Media, Scope, YOLOAdapter, Zone

SRC = "./samples/car_highway.mp4"

# Zone picked by the user on this video — irregular 7-point polygon
# (traced around the road; the closing repeat point is dropped).
ROAD = Zone(
    points=[(332, 84), (326, 203), (158, 198), (64, 316),
            (315, 332), (608, 326), (414, 73)],
    name="road",
)

detector = Scope(
    YOLOAdapter(weights="models/yolov8n.pt", imgsz=480),
    ROAD,
    margin=20,
)

video = Media(SRC)
cars = video.detect(detector=detector, confidence=0.4, labels=["car"]).track()

sink_full = VideoSink(SRC, out="./samples/car_highway_scope_full.mp4")
sink_region = VideoSink(SRC, out="./samples/car_highway_scope_region.mp4")

# --- app-side visuals --------------------------------------------------------
PTS = np.array(ROAD.points, np.int32)
GAP, GRAY = 16, (110, 110, 110)
_cache: dict = {}


def _layers(shape):
    """Inside/outside masks + hatch layer, computed once for the frame size."""
    if "layers" not in _cache and _cache.get("shape") != shape:
        h, w = shape[:2]
        inside = np.zeros((h, w), np.uint8)
        cv2.fillPoly(inside, [PTS], 255)
        hatch = np.zeros((h, w, 3), np.uint8)
        for x in range(-h, w + h, GAP):           # angled parallel lines
            cv2.line(hatch, (x, h), (x + h, 0), GRAY, 1)
        outside = cv2.bitwise_not(inside)
        _cache["layers"] = (
            inside,
            outside,
            cv2.bitwise_and(hatch, hatch, mask=outside),  # hatch only outside
        )
        _cache["shape"] = shape
    return _cache["layers"]


def full_view(frame):
    """Whole frame + hatched excluded area + scope border."""
    inside, outside, hatch = _layers(frame.shape)
    view = frame.copy()
    cv2.add(view, hatch, dst=view)
    cv2.polylines(view, [PTS], True, (255, 255, 255), 2)
    return view


def region_view(frame):
    """Outside blacked out; hatch kept so the boundary stays readable."""
    inside, outside, hatch = _layers(frame.shape)
    view = cv2.bitwise_and(frame, frame, mask=inside)     # outside -> black
    cv2.add(view, hatch, dst=view)                        # hatch over black
    cv2.polylines(view, [PTS], True, (255, 255, 255), 2)
    return view


def _label(view, text):
    """Label with a black backing plate, readable over hatch or road."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 1)
    x, y = 20, 80
    cv2.rectangle(view, (x - 10, y - th - base - 10),
                  (x + tw + 10, y + 10), (0, 0, 0), -1)
    cv2.putText(view, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 1, cv2.LINE_AA)


def draw_and_write(e):
    full = full_view(e.frame)
    region = region_view(e.frame)
    _label(full, "full view - hatched = model blind spot")
    _label(region, "region view - what the model sees")
    sink_full.write(e.tracks, full)
    sink_region.write(e.tracks, region)


cars.on_update(draw_and_write)

video.run()
sink_full.close()
sink_region.close()
