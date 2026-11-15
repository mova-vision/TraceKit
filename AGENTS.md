# TraceKit — Agent / Contributor Reference

Keep this file updated as the architecture evolves.

## What TraceKit is

TraceKit is an open-source Python framework for media + computer vision automation.

It is **not** another YOLO wrapper or OpenCV utility library. It provides the reusable infrastructure **between** a user’s model and their application logic:

```text
Media → Detector (user-supplied) → TraceKit Detection → Tracking
  → Track / Position / Path → Spatial reasoning → Events → Application logic
```

TraceKit owns infrastructure. The application owns business behavior and rendering.

## Long-term architecture

```text
Media sources (file, image, webcam, browser camera, stream, …)
        ↓
Detector adapter (YOLO / PyTorch / MediaPipe / ONNX / API / custom)
        ↓
Normalized Detection
        ↓
Tracking (persistent IDs, path accumulation)
        ↓
Spatial reasoning (zones, transforms, distances, collisions, …)
        ↓
Events (update, enter, exit, cross, collision, …)
        ↓
Optional visualization backends (OpenCV, Canvas, …)
        ↓
User application callbacks
```

## Core design principles

1. **Normalize at the detector boundary.** Core code never consumes YOLO/MediaPipe/OpenCV result objects.
2. **BYO detector.** `Detector` is a public ABC; any model stack can adapt into `Detection`.
3. **Tracking is detector-independent.** IDs, positions, and paths are TraceKit state.
4. **Events notify; apps decide.** TraceKit detects events; applications choose reactions.
5. **Rendering is not core.** Emit geometry/state (points, boxes, paths). Backends/apps draw.
6. **Media is abstracted.** Do not couple the core to `cv2.VideoCapture` alone.
7. **Keep internals internal.** Trackers, registries, pipelines stay replaceable under `_internal/` / `tracking/`.

## Detector abstraction

```python
from tracekit import Detector, Detection

class MyDetector(Detector):
    def detect(self, frame) -> list[Detection]:
        ...
```

Boundary:

```text
User model output → Detector / adapter → Detection → TraceKit core
```

`YOLOAdapter` is an adapter only. YOLO must never define the framework architecture.

## Media abstraction

`Media` is the fluent entrypoint:

```python
video = Media(source)
objects = video.detect(detector=...).track()
objects.on("update", handler)
video.run()
```

v1 sources: filesystem video path (OpenCV adapter) and iterable frames (tests / streaming). Browser camera is demonstrated via an **example server**, not a core Media backend yet.

## Tracking responsibilities

TraceKit should:

- assign persistent track IDs
- maintain current `position`
- accumulate `path` over time
- remain independent of detector implementation

Application code should not manually append positions each frame.

## Track / path model

```python
track.id
track.position   # (x, y)
track.path       # [(x, y), ...]
```

Optional convenience fields in v1: `label`, `confidence`, `bbox`, `detection`.

## Event architecture

String-named emitter. Handlers receive a single `Event` object
(`event.tracks` / `event.track`, `event.other`, `event.frame`, `event.zone`,
`event.line`, `event.coverage`, `event.direction`, `event.distance`).

Shipped producers (each validates its event names in `on()`):

| Producer (module) | Events | Payload highlights |
|---|---|---|
| `ZoneWatcher` (`spatial/zone.py`) | `enter` / `exit` | `track`, `zone`, `coverage` |
| `DwellWatcher` (`spatial/zone.py`) | `dwell` / `leave` | `zone` + `min_frames` stay gate |
| `LineWatcher` (`spatial/line.py`) | `cross` | `line`, `direction` (+1/-1 side moved onto) |
| `ProximityWatcher` (`spatial/proximity.py`) | `proximity` | `track` + `other` pair, `distance`, transition-based |
| `LifecycleWatcher` (`core/lifecycle.py`) | `appear` / `lost` | track first seen / dropped by tracker |

Fluent conveniences on `TrackedObjects`: `on_enter` / `on_exit` / `on_dwell` /
`on_cross` / `on_proximity` / `on_appear` / `on_lost` — each find-or-creates
one producer per geometry (see `_producer`). Crossings require observed
movement of a *tracked* id; a brand-new track appearing beyond a line is not
a cross. Proximity is O(n²) pairs; everything else is O(1) per track.

Users can add their **own events** without core changes: subclass
`EventProducer` (implement `update(tracks)`, fire with `self.emit(...)`),
register via `objects.add_producer(...)`, attach handlers on the producer.
Producers may mix any state (zones, paths, speeds, other producers).

Zones come from `tracekit.spatial` (`Zone` polygon + `min_coverage`, the
fraction of a track's bbox that must be inside before it counts).

Principle: TraceKit produces the event; the application decides meaning.

## Multi-object Media

One `Media` can host several `detect(...).track()` chains (e.g. people and
cars). Frames are decoded once and each detector instance runs once per
frame; every chain keeps its own filtering, tracker, IDs and events.

## Rendering philosophy

Do **not** make TraceKit depend on `cv2.circle` / `cv2.rectangle` in core.

Future: generic primitives (points, paths, boxes, lines, polygons, text) with pluggable backends (OpenCV, HTML Canvas, etc.).

Until then: examples draw; core returns data.

## What belongs in TraceKit vs application code

| TraceKit | Application |
|----------|-------------|
| Media iteration | Business rules |
| Detection normalization | Model training / weights choice |
| Tracking + path state | What a collision *means* |
| Event emission | Handlers / side effects |
| Spatial transforms (future) | Domain geometry (e.g. tennis scoring) |
| Optional render adapters (future) | Colors, strokes, UI chrome |

## Current v1 scope

- Package layout under `src/tracekit/`
- Public: `Media`, `Detector`, `Detection`, `Track`, `YOLOAdapter`, `Zone`,
  `Line`, `LifecycleWatcher` (+ watchers in `tracekit.spatial`)
- Centroid tracker with path accumulation
- Event producers: `update`, `enter`/`exit`, `dwell`/`leave`, `cross`,
  `proximity`, `appear`/`lost`
- OpenCV file reader adapter
- Examples: `examples/basic_tracking.py`, `examples/traffic_zone.py`,
  `examples/events_demo.py`, `examples/browser/`
- Unit tests with fake detectors (no YOLO required)

## Planned / future capabilities

- collision (bbox overlap) events, speed / stationary events
- perspective / coordinate mapping
- zone occupancy counts
- additional media backends (webcam, browser-native Media, streams)
- visualization primitive layer + backends
- stronger / swappable trackers

## Public API conventions

- Prefer a small fluent surface: `detect` → `track` → `on` → `run`
- Export only stable types from `tracekit`
- Adapters live in `tracekit.adapters`
- Anything under `_internal/` may change without notice

## Important architectural decisions

- Normalization happens inside detectors/adapters, not in trackers or events
- OpenCV and ultralytics are **optional** dependencies
- Browser demo talks HTTP to a local server that runs the same core pipeline
- Trackers are implementation details; public object is `Track`

## Things future contributors / agents should NOT do

- Do not make YOLO the center of the architecture
- Do not put application business logic in the library
- Do not bake OpenCV drawing into core
- Do not expose internal tracker/pipeline types as the primary public API
- Do not hardcode tennis / sport-specific geometry into core
- Do not expand event types by special-casing the emitter — add producers, keep dispatch generic
- Do not treat v1 as final architecture; extend at the boundaries
