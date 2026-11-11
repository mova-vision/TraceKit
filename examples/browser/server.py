"""Minimal browser-camera demo for TraceKit.

Open http://127.0.0.1:8000 after starting this server.
The page uses the browser camera (handy on WSL2) and draws simple markers
from TraceKit track state. Drawing is application code, not TraceKit.

Usage:
    pip install -e ".[all]"
    python examples/browser/server.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from tracekit import Media, YOLOAdapter
from tracekit.adapters.opencv_media import decode_image_bytes

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="TraceKit browser demo")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_media: Media | None = None


def build_pipeline(
    weights: str,
    confidence: float,
    labels: list[str] | None,
    imgsz: int = 320,
) -> Media:
    media = Media([])  # streaming frames arrive via HTTP
    detector = YOLOAdapter(weights=weights, imgsz=imgsz)
    media.detect(detector=detector, confidence=confidence, labels=labels).track()
    # Warm up: first real inference is slow (kernel autotuning), so pay it now
    import numpy as np

    detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    return media


def tracks_to_json(tracks) -> list[dict]:
    payload = []
    for track in tracks:
        payload.append(
            {
                "id": track.id,
                "label": track.label,
                "confidence": track.confidence,
                "position": track.position,
                "bbox": track.bbox,
                "path": track.path[-32:],  # keep payload small
            }
        )
    return payload


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/frame")
async def process_frame(file: UploadFile = File(...)) -> dict:
    if _media is None:
        return {"tracks": [], "error": "pipeline not ready"}
    data = await file.read()
    frame = decode_image_bytes(data)
    tracks = _media.process_frame(frame)
    return {"tracks": tracks_to_json(tracks)}


def main() -> None:
    global _media

    import torch

    torch.set_num_threads(min(8, __import__("os").cpu_count() or 4))

    parser = argparse.ArgumentParser(description="TraceKit browser camera demo")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=320,
                        help="YOLO input size; smaller = faster (320 ≈ 60 FPS CPU)")
    parser.add_argument("--labels", default="cell phone",
                        help="Comma-separated COCO labels (empty = all classes)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    labels = [s.strip() for s in args.labels.split(",") if s.strip()] or None
    _media = build_pipeline(args.weights, args.confidence, labels, args.imgsz)

    print(f"Open http://127.0.0.1:{args.port} in your Windows browser")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
