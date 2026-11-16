"""Count oranges on a conveyor — three counting lines, one counter each.

    python examples/conveyor_count.py

Video goes at samples/conveyor_orange.mp4 (image/video source of the belt).
COCO class name is "orange" — apples use labels=["apple"] the same way.
"""

import cv2
import numpy as np

from render import VideoSink
from tracekit import Line, Media, Scope, YOLOAdapter, Zone
from tracekit.spatial import LineWatcher
from tracekit.tracking.tracker import CentroidTracker

SRC = "./samples/conveyor_orange.mp4"

# Scope: model only sees the belt (user-picked 5-point polygon). The crop is
# the polygon's bounding box; the polygon itself decides what counts.
BELT = Zone(points=[(4, 263), (1, 652), (1072, 1556), (1076, 506), (338, 196)],
            name="belt")

# Three counting lines, one per conveyor row (user-picked, clear of corners).
LINES = [
    Line(a=(131, 728), b=(315, 539), name="row1"),
    Line(a=(347, 534), b=(523, 420), name="row2"),
    # row3 now crosses the busy top-left lane (the stream under the
    # brushes) — the old line sat over an empty lane, which is why it
    # never counted.
    Line(a=(260, 350), b=(130, 500), name="row3"),
]

counts = {ln.name: 0 for ln in LINES}
seen: set[tuple[str, int]] = set()   # (line name, track id) — count once each

# Belt flow is +1 on these lines (verified on a first run: jitter made some
# oranges bounce back across a line; counting both directions double-counts
# them). Only forward crossings count.
FLOW = +1

video = Media(SRC)
apples = video.detect(
    detector=Scope(YOLOAdapter(weights="models/yolov8m.pt", imgsz=640),
                   BELT, margin=20),
    confidence=0.2,   # low: blurry oranges score weak; tracker absorbs misses
    labels=["orange"],
).track(
    # Best measured config on this footage (10/6/2 with max_missed=30, and
    # 150 px distance merged adjacent oranges): defaults win. Remaining
    # misses are detection-side (blur), not tracking-side.
    tracker=CentroidTracker(max_distance=80, max_missed=15),
)


def make_on_cross(line: Line):
    def on_cross(e):
        key = (line.name, e.track.id)
        if e.direction != FLOW:
            print(f"orange {e.track.id} bounced back on {line.name} "
                  f"({e.direction:+d}) — ignored")
            return
        if key in seen:
            print(f"orange {e.track.id} re-crossed {line.name} — ignored")
            return
        seen.add(key)
        counts[line.name] += 1
        print(f"orange {e.track.id} crossed {line.name} "
              f"({e.direction:+d}) -> {counts}")
    return on_cross


for ln in LINES:
    # edge_margin=0.1: the outer 10% of each line at both ends is dead —
    # a wobbling orange that only clips the line's tip doesn't count.
    watcher = LineWatcher(ln, edge_margin=0.1)
    apples.add_producer(watcher)
    watcher.on("cross", make_on_cross(ln))

sink = VideoSink(SRC, out="./samples/conveyor_orange_count_output.mp4",
                 overlay=lambda: "  ".join(f"{k}: {v}" for k, v in counts.items())
                                 + f"  total: {sum(counts.values())}")

# --- app-side visuals: hatched blind spot outside the belt, like the
# highway scope demo ---------------------------------------------------------
PTS = np.array(BELT.points, np.int32)
GAP, GRAY = 16, (110, 110, 110)
_cache: dict = {}


def _hatch_outside(shape):
    """Angled parallel gray lines covering everything outside the belt."""
    if _cache.get("shape") != shape:
        h, w = shape[:2]
        inside = np.zeros((h, w), np.uint8)
        cv2.fillPoly(inside, [PTS], 255)
        hatch = np.zeros((h, w, 3), np.uint8)
        for x in range(-h, w + h, GAP):
            cv2.line(hatch, (x, h), (x + h, 0), GRAY, 1)
        outside = cv2.bitwise_not(inside)
        _cache["shape"] = shape
        _cache["layer"] = cv2.bitwise_and(hatch, hatch, mask=outside)
    return _cache["layer"]


def draw_and_write(e):
    view = e.frame.copy()
    cv2.add(view, _hatch_outside(view.shape), dst=view)
    cv2.polylines(view, [PTS], True, (255, 255, 255), 2)
    for ln in LINES:
        cv2.line(view, ln.a, ln.b, (0, 200, 255), 2)
        mx, my = int((ln.a[0] + ln.b[0]) / 2), int((ln.a[1] + ln.b[1]) / 2)
        text = f"{ln.name}: {counts[ln.name]}"
        cv2.putText(view, text, (mx - 50, my - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (mx - 50, my - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    sink.write(e.tracks, view)


apples.on_update(draw_and_write)

video.run()
sink.close()
print(f"final: {counts}")
