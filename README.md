# TraceKit

Infrastructure between your vision model and application logic.

Bring your own detector. TraceKit normalizes detections, tracks objects, accumulates paths, and emits events. Your app decides what those events mean and how to render.

```python
from tracekit import Media, YOLOAdapter

detector = YOLOAdapter(weights="yolov8n.pt")
video = Media("tennis.mp4")

objects = (
    video
    .detect(detector=detector, confidence=0.4, labels=["sports ball"])
    .track()
)

objects.on("update", lambda tracks, frame: print(tracks[0].id, tracks[0].path[-1]))
video.run()
```

## Install

```bash
pip install -e ".[dev]"          # core + tests
pip install -e ".[all]"          # opencv + yolo + browser demo + tests
```

## Run tests

```bash
pytest -v
```

## File video demo

```bash
pip install -e ".[all]"
python examples/basic_tracking.py path/to/video.mp4
# optional label filter (COCO names):
python examples/basic_tracking.py path/to/video.mp4 "person,sports ball"
```

## Browser camera demo (WSL2-friendly)

Python runs in WSL; the camera lives in your Windows browser.

```bash
pip install -e ".[all]"
python examples/browser/server.py
# optional:
python examples/browser/server.py --labels "person,sports ball" --confidence 0.35
```

Open **http://127.0.0.1:8000** in your Windows browser, allow camera access, and move objects in view. Markers and short paths are drawn by the example page from TraceKit track state.

## Layout

See `AGENTS.md` for architecture, design principles, and what belongs in TraceKit vs application code.
