# TraceKit v1 Foundation Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Minimal working TraceKit: Detector → Detection → Track (id/position/path) → `"update"` events, plus OpenCV/YOLO example and browser-camera demo.

**Architecture:** Fluent `Media.detect().track()` pipeline; YOLO/OpenCV only in adapters/examples; core stays framework-agnostic.

**Tech Stack:** Python 3.10+, numpy, pytest; optional opencv-python, ultralytics, fastapi, uvicorn.

## Global Constraints

- Public API: Media, Detector, Detection, Track, events, YOLOAdapter
- Core must not import cv2 or ultralytics
- Rendering stays in examples/application code
- v1 events: `"update"` only
- Package layout under `src/tracekit/` as in design spec

---

### Task 1: Package scaffold + core types

**Files:**
- Create: `pyproject.toml`, `LICENSE`, `README.md`, `AGENTS.md`
- Create: `src/tracekit/__init__.py`
- Create: `src/tracekit/core/__init__.py`, `detection.py`, `detector.py`, `track.py`, `events.py`
- Test: `tests/test_detector.py`, `tests/test_events.py`

**Interfaces:**
- Produces: `Detection(bbox, confidence, label, position)`, `Detector.detect(frame) -> list[Detection]`, `Track(id, position, path)`, `EventEmitter.on/emit`

- [ ] Write tests for Detection position default from bbox center, Detector ABC, EventEmitter update
- [ ] Implement types + package metadata
- [ ] `pytest tests/test_detector.py tests/test_events.py -v` PASS

---

### Task 2: Tracker + Track path accumulation

**Files:**
- Create: `src/tracekit/tracking/__init__.py`, `tracker.py`
- Test: `tests/test_tracking.py`

**Interfaces:**
- Consumes: Detection, Track
- Produces: `CentroidTracker.update(detections) -> list[Track]` with stable ids and growing `path`

- [ ] Write failing tests: same object keeps id across frames; path length grows
- [ ] Implement nearest-centroid matcher with max distance + new ids
- [ ] `pytest tests/test_tracking.py -v` PASS

---

### Task 3: Media pipeline + OpenCV adapter

**Files:**
- Create: `src/tracekit/core/media.py`, `src/tracekit/_internal/pipeline.py`
- Create: `src/tracekit/adapters/__init__.py`, `opencv_media.py`
- Modify: `src/tracekit/__init__.py` exports

**Interfaces:**
- Produces: `Media(source).detect(detector, confidence=...).track()` → handle with `.on("update", cb)`; `Media.run()` drives frames

- [ ] Synthetic Media source for tests (list of frames) so CI needs no OpenCV video
- [ ] OpenCV file reader behind Media when source is a path string
- [ ] Wire detect → track → emit update each frame

---

### Task 4: YOLOAdapter

**Files:**
- Create: `src/tracekit/adapters/yolo.py`

**Interfaces:**
- Produces: `YOLOAdapter(weights, device=None)` implementing Detector; maps ultralytics boxes to Detection

- [ ] Implement adapter; lazy-import ultralytics
- [ ] Optional conf/label filter kwargs on detect path via Media.detect

---

### Task 5: Examples — file + browser

**Files:**
- Create: `examples/basic_tracking.py`
- Create: `examples/browser/server.py`, `examples/browser/static/index.html`

- [ ] basic_tracking: Media + YOLO + print track id/position/path len on update
- [ ] browser: FastAPI serves HTML; POST/WebSocket frame → pipeline → JSON tracks; page draws circles at positions

---

### Task 6: Docs polish + verify

- [ ] AGENTS.md matches architecture
- [ ] README run instructions for file + browser demos
- [ ] Full `pytest` green
