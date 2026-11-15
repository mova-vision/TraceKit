# TraceKit Usage

How to use TraceKit. No theory — just what to call and what you get.

## Install

```bash
pip install -e ".[all]"     # opencv + yolo + everything
```

## Basic loop

```python
from tracekit import Media, YOLOAdapter

video = Media("video.mp4")
objects = video.detect(
    detector=YOLOAdapter(weights="yolov8n.pt"),
    confidence=0.4,
    labels=["car"],            # filter by class name (optional)
).track()

objects.on_update(lambda e: print(e.tracks))   # runs every frame
video.run()
```

Every detected object becomes a `Track` with a persistent ID:

```python
for t in objects.tracks:      # all objects, right now
    t.id         # 3 — same car = same id, every frame
    t.position   # (x, y) center
    t.path       # [(x, y), ...] full history
    t.bbox       # (x1, y1, x2, y2)
    t.label      # "car"
```

## Events

All handlers look the same: `def handler(e): ...`. Register with one line.

### Cross a line — counting, direction

```python
from tracekit import Line

gate = Line(a=(40, 180), b=(600, 180), name="gate")

def on_cross(e):
    print(f"car {e.track.id} crossed going {'down' if e.direction == -1 else 'up'}")

objects.on_cross(gate, on_cross)
```

- `e.direction` is `+1` or `-1` — the side the object moved onto, relative to the line's a→b direction
- A crossing only counts if it happens **on the drawn segment** — cars passing the line's extension past its endpoints don't trigger it

### Enter / exit a zone — occupancy

```python
from tracekit import Zone

gate = Zone(points=[(209, 128), (459, 123), (604, 308), (81, 292)],
            name="gate", min_coverage=0.5)

objects.on_enter(gate, lambda e: print("in ", e.track.id, e.coverage))
objects.on_exit(gate, lambda e: print("out", e.track.id))
```

- `min_coverage` = fraction of the object's box that must be inside: `0.0` first touch, `0.5` half, `1.0` whole
- Counting occupancy = increment on `enter`, decrement on `exit`

### Dwell — staying too long

```python
from tracekit.spatial import DwellWatcher

watcher = DwellWatcher(gate, min_frames=30)   # ~1 s at 30 fps
objects.add_producer(watcher)
watcher.on("dwell", lambda e: print(e.track.id, "is lingering"))
watcher.on("leave", lambda e: print(e.track.id, "moved on"))
```

- Fires once after `min_frames` consecutive frames inside; `leave` when it exits or vanishes
- A duration *gate* (yes/no), not a duration meter — for exact times, record `enter`/`exit` frames yourself

### Proximity — two objects too close

```python
def on_proximity(e):
    print(f"cars {e.track.id} & {e.other.id}: {e.distance:.0f} px")

objects.on_proximity(80, on_proximity)   # 80 px threshold
```

- Fires **once** when a pair becomes close; fires again only after they separate and close back in

### Appear / lost — arrivals and departures

```python
objects.on_appear(lambda e: print(f"car {e.track.id} entered the frame"))
objects.on_lost(lambda e: print(f"car {e.track.id} dropped by tracker"))
```

## Scope — model looks only inside a zone

```python
from tracekit import Media, Scope, YOLOAdapter, Zone

road = Zone(points=[(329, 105), (318, 308), (576, 292), (435, 96)])

detector = Scope(YOLOAdapter(weights="yolov8n.pt"), road, margin=20)

video = Media("video.mp4")
objects = video.detect(detector=detector, confidence=0.4, labels=["car"]).track()
```

- The frame is **cropped** to the zone (+ `margin` px) *before* the model runs — less compute, nothing outside is ever detected
- Detections are mapped back to **full-frame coordinates** — tracking, events, and your rendering keep working on the full image, unchanged
- `road.min_coverage` also filters detections (default `0.0` = any overlap with the zone counts)
- Pass a **list** of zones to watch several regions with a single model run: `Scope(detector, [zone_a, zone_b])` — a detection inside any zone is kept

## Rendering

TraceKit never draws. Draw anything you want in an `on_update` handler:

```python
import cv2

def draw(e):
    for t in e.tracks:
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        cv2.rectangle(e.frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

objects.on_update(draw)
```

Runnable, minimal examples for every feature live in [`examples/`](examples/).
