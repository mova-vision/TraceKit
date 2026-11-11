# TraceKit v1 Foundation Design

**Date:** 2026-11-11  
**Status:** Approved for implementation (user directed: build minimal working prototype now)

## Goal

Ship a minimal, working TraceKit foundation: detector abstraction → normalized detections → tracking with persistent IDs and path → `"update"` events → runnable pipeline. Prove it with an OpenCV/YOLO path and a tiny browser-camera demo (WSL-friendly).

## Non-goals (v1)

- Collision / cross / enter / exit event logic
- Perspective / court mapping
- Multiple media backends in core
- Generic visualization subsystem in core
- Styling, colors, fancy overlays (application concern)

## Architecture

```text
Media source (file | browser frames via example server)
        ↓
Detector (user / YOLOAdapter)
        ↓
Detection[]  (normalized)
        ↓
Tracker (internal)
        ↓
Track[]  (id, position, path)
        ↓
Event "update"
        ↓
Application callbacks / browser draw
```

TraceKit owns infrastructure. The application owns what detections mean and how to render.

## Public API

```python
from tracekit import Media, Detector, Detection, Track, YOLOAdapter

detector = YOLOAdapter(weights="yolov8n.pt")  # or custom Detector

video = Media("input.mp4")  # OpenCV-backed file source in v1

objects = (
    video
    .detect(detector=detector, confidence=0.4)
    .track()
)

objects.on("update", handler)
video.run()
```

Browser demo uses the same core pipeline over frames posted from the page; it is an **example**, not a core Media type.

## Package layout

```text
tracekit/
├── pyproject.toml
├── README.md
├── AGENTS.md
├── LICENSE
├── src/tracekit/
│   ├── __init__.py
│   ├── core/
│   │   ├── media.py
│   │   ├── detector.py
│   │   ├── detection.py
│   │   ├── track.py
│   │   └── events.py
│   ├── tracking/
│   │   └── tracker.py
│   ├── adapters/
│   │   ├── yolo.py
│   │   └── opencv_media.py
│   └── _internal/
│       └── pipeline.py
├── examples/
│   ├── basic_tracking.py
│   └── browser/
│       ├── server.py
│       └── static/index.html
└── tests/
```

## Core types

### Detection

Normalized model output. Fields: `bbox` (x1,y1,x2,y2), `confidence`, `label`, `position` (cx, cy). No framework-specific objects leak past the adapter.

### Detector

ABC with `detect(frame) -> list[Detection]`. Frame is opaque enough for adapters (v1: numpy ndarray BGR/RGB as used by OpenCV/YOLO). Core does not import ultralytics.

### Track

`id: int`, `position: tuple[float, float]`, `path: list[tuple[float, float]]`, plus current `detection` / label as convenient. Path accumulates positions each update.

### Events

Minimal emitter on the tracked-objects handle. v1 supports `"update"` only. Payload includes track snapshot(s) for the frame. Extensible string-named events later without changing core dispatch shape.

### Media

Owns source + pipeline config. Fluent: `.detect(...)` → `.track()` → returns / mutates a pipeline handle that supports `.on(...)`. `.run()` iterates frames, runs detect→track→dispatch.

OpenCV `VideoCapture` lives in `adapters/opencv_media.py`, not as the Media public contract.

## Tracking

Simple nearest-centroid (or IoU) matcher with max distance threshold and new-ID allocation. Internal; swappable later. Independent of detector implementation.

## Adapters

- `YOLOAdapter(Detector)` — ultralytics optional dependency
- OpenCV video file reader used by `Media(path)` when path is a file

## Browser demo

Local server + single HTML page:

1. `getUserMedia` for camera (works in Windows browser while Python runs in WSL if bound to `0.0.0.0` / localhost forwarding)
2. Capture frames → send to server (JPEG/base64 or binary over HTTP/WebSocket)
3. Server: decode → YOLOAdapter → track → return track JSON (`id`, `position`, `bbox`, `path` tail)
4. Page draws minimal markers (circles/boxes) — **application** drawing, not TraceKit

## Testing

Unit tests with a fake `Detector` and synthetic frames (no YOLO required in CI). Cover detection normalization, ID persistence, path accumulation, `"update"` firing.

## Dependencies

- Core: numpy (frames/coords)
- Optional: opencv-python (file media + examples), ultralytics (YOLO), fastapi/uvicorn (browser example)

## Design principles

1. Normalize at the detector boundary
2. Core never depends on YOLO/OpenCV/browser APIs
3. Tracks and paths are framework-owned state
4. Events notify; apps decide behavior
5. Rendering is out of core
6. Keep internals internal
