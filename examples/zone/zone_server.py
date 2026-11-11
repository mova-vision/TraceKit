"""Zone testing server — upload a video, pick zones in the browser,
get back an annotated video with enter/exit counting.

Run:
    python examples/zone/zone_server.py
Open:
    http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/ → render.py
from render import VideoSink  # noqa: E402

from tracekit import Media, YOLOAdapter, Zone  # noqa: E402

app = FastAPI(title="TraceKit zone tester")
HERE = Path(__file__).resolve().parent


def transcode_for_browser(src: Path) -> Path:
    """OpenCV writes mpeg4, which browsers can't decode in <video>.

    Re-encode to H.264/yuv420p with ffmpeg when available; else serve as-is.
    """
    if shutil.which("ffmpeg") is None:
        return src
    dst = src.with_name("browser.mp4")
    result = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
         "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(dst)],
        capture_output=True,
    )
    return dst if result.returncode == 0 and dst.exists() else src


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HERE / "testing.html")


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    zones: str = Form(...),
    labels: str = Form("car"),
    confidence: float = Form(0.4),
    min_coverage: float = Form(0.5),
    imgsz: int = Form(480),
    model: str = Form("yolov8n"),
) -> FileResponse:
    data = await file.read()
    tmp = Path(tempfile.mkdtemp())
    src_path = tmp / "input.mp4"
    out_path = tmp / "output.mp4"
    src_path.write_bytes(data)

    zone_list = [
        Zone(
            name=z.get("name") or f"zone{i + 1}",
            points=[(float(x), float(y)) for x, y in z["points"]],
            min_coverage=float(z.get("min_coverage", min_coverage)),
        )
        for i, z in enumerate(json.loads(zones))
    ]

    cap = cv2.VideoCapture(str(src_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    weights = f"models/{model}.pt"
    detector = YOLOAdapter(weights=weights, imgsz=int(imgsz))
    video = Media(str(src_path))
    label_list = [s.strip() for s in labels.split(",") if s.strip()] or None
    objects = video.detect(
        detector=detector, confidence=confidence, labels=label_list
    ).track()

    counts: dict[str, dict[str, int]] = {
        z.name: {"in": 0, "out": 0} for z in zone_list
    }

    def make_on_enter(zone: Zone):
        def handler(e) -> None:
            counts[zone.name]["in"] += 1
        return handler

    def make_on_exit(zone: Zone):
        def handler(e) -> None:
            counts[zone.name]["out"] += 1
        return handler

    for zone in zone_list:
        objects.on_enter(zone, make_on_enter(zone))
        objects.on_exit(zone, make_on_exit(zone))

    def overlay() -> str:
        return "  ".join(
            f"{name}: in {s['in']} out {s['out']}" for name, s in counts.items()
        )

    sink = VideoSink(out=str(out_path), fps=fps, zones=zone_list, overlay=overlay)
    objects.on_update(lambda e: sink.write(e.tracks, e.frame))
    video.run()
    sink.close()

    final_counts = {
        name: {**s, "current": s["in"] - s["out"]} for name, s in counts.items()
    }

    final_path = transcode_for_browser(out_path)

    return FileResponse(
        final_path,
        media_type="video/mp4",
        filename="annotated.mp4",
        headers={"X-Zone-Counts": json.dumps(final_counts)},
        background=BackgroundTask(lambda: shutil.rmtree(tmp, ignore_errors=True)),
    )


if __name__ == "__main__":
    print("Open http://127.0.0.1:8000 in your browser")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
