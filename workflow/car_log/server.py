"""Car log workflow: live browser video → zone → snapshot every car that
fully enters the zone (>=90% of its box inside), shown in a sidebar.

Run:
    python workflow/car_log/server.py
Open:  http://127.0.0.1:8090

The browser sends JPEG frames over HTTP (works for an uploaded video or
the webcam — same path). The server runs the TraceKit pipeline and returns
tracks + any new car snapshots (crops encoded as base64 JPEG).

Processing only starts once the client sends a zone: Scope crops the frame
to the zone (cars outside it are never even detected), and the zone's
``min_coverage=0.9`` enforces "fully inside".
"""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from tracekit import ByteTracker, ByteTrackerConfig, Media, Scope, YOLOAdapter
from tracekit.adapters.opencv_media import decode_image_bytes
from tracekit.core.track import Track
from tracekit.spatial.zone import Zone

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="TraceKit car log")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# One detector instance, built and warmed once; rebuilt pipelines reuse it
# (no weight reload when the zone changes).
base_detector = YOLOAdapter(weights="models/yolov8n.pt", imgsz=480)
base_detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))  # warm up

COVERAGE = 0.9  # fraction of the car's box that must be inside to be LOGGED
DETECT_COVERAGE = 0.5  # boxes show from half-in — logging still waits for 90%

media: Media | None = None  # built when the client sends a zone
zone: Zone | None = None
captured: dict[int, dict] = {}  # track id → snapshot
_pending: dict[int, dict] = {}  # track id → sharpest-crop search state
SHARPEN_FRAMES = 8  # frames to watch a car and keep its sharpest crop


def sharpness(crop: np.ndarray) -> float:
    """Laplacian variance — the standard blur metric (higher = sharper)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def build_media(log_zone: Zone) -> Media:
    m = Media([])
    # Detection zone: same polygon, looser gate — tracking starts while the
    # car is still entering, so fast cars can't slip through untracked.
    det_zone = Zone(points=log_zone.points, name="detect",
                    min_coverage=DETECT_COVERAGE)
    m.detect(
        # Scope: model sees ONLY the zone (+margin) — cars far outside are
        # never detected; entering cars get boxes early (blue in the UI).
        detector=Scope(base_detector, det_zone, margin=40),
        confidence=0.10,  # low floor: weak boxes reach the tracker's rescue stage
        labels=["car"],
    ).track(
        # ByteTrack-style tracker: Kalman velocity prediction + two-stage
        # association — a car missing for a few frames keeps ITS id, and the
        # car behind never inherits it (the centroid tracker's old bug).
        tracker=ByteTracker(ByteTrackerConfig(
            confirm_hits=2, max_age=30, hi_thresh=0.5, lo_thresh=0.1,
        )),
    )
    m.process_frame(np.zeros((480, 640, 3), dtype=np.uint8))  # warm up
    return m


def fresh_tracks(tracks: list[Track]) -> list[Track]:
    """Drop coasting tracks (no detection this frame) — no stale boxes."""
    return [t for t in tracks if t.bbox is not None and t.missed <= 2]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/zone")
def set_zone(points: list[tuple[int, int]]) -> dict[str, object]:
    global media, captured, zone
    captured = {}
    _pending.clear()
    if len(points) >= 3:
        zone = Zone(points=points, name="target", min_coverage=COVERAGE)
        media = build_media(zone)
        return {"active": True, "points": len(points), "coverage": COVERAGE}
    media = None  # empty zone = processing off (client can draw again)
    return {"active": False}


@app.post("/frame")
async def frame(request: Request) -> dict:
    if media is None:  # no zone yet — nothing is processed
        return {"tracks": [], "cars": []}

    rgb = decode_image_bytes(await request.body())
    tracks = fresh_tracks(media.process_frame(rgb))

    new_cars = []
    seen = set()
    for t in tracks:
        if t.bbox is None or t.id in captured:
            continue
        # Logging rule: only cars ≥COVERAGE inside the LOG zone enter the
        # snapshot search (detections start earlier, logging stays strict).
        assert zone is not None
        if zone.coverage(t.bbox) < COVERAGE:
            continue
        x1, y1, x2, y2 = (int(v) for v in t.bbox)
        crop = rgb[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue

        # Snapshot timing: watch each qualifying car for a few frames and
        # keep its SHARPEST crop — the first frame is often motion-blurred.
        score = sharpness(crop)
        state = _pending.get(t.id)
        if state is None:
            _pending[t.id] = {"best": crop, "score": score, "age": 0,
                              "miss": 0}
            continue
        state["age"] += 1
        state["miss"] = 0
        if score > state["score"]:
            state["best"], state["score"] = crop, score
        if state["age"] < SHARPEN_FRAMES:
            continue

        ok, buf = cv2.imencode(
            ".jpg", state["best"], [cv2.IMWRITE_JPEG_QUALITY, 85])
        _pending.pop(t.id, None)
        if ok and buf is not None:
            captured[t.id] = {
                "jpeg": base64.b64encode(buf).decode(),
                "id": t.id,
                "label": t.label or "car",
            }
            new_cars.append(captured[t.id])
        seen.add(t.id)

    # Cars that left the zone mid-search: flush their best-so-far crop.
    for tid in [i for i, s in _pending.items() if i not in seen]:
        s = _pending[tid]
        s["miss"] += 1
        if s["miss"] > 2:
            _pending.pop(tid, None)
            if tid not in captured and s["best"].size:
                ok, buf = cv2.imencode(
                    ".jpg", s["best"], [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok and buf is not None:
                    captured[tid] = {
                        "jpeg": base64.b64encode(buf).decode(),
                        "id": tid,
                        "label": "car",
                    }
                    new_cars.append(captured[tid])

    return {
        "tracks": [
            {
                "id": t.id,
                "bbox": t.bbox,
                "label": t.label,
                # honest status: in-zone = ≥90% of box inside the polygon;
                # logged = snapshot saved. Colors in the browser use this.
                "in_zone": zone is not None and t.bbox is not None
                    and zone.coverage(t.bbox) >= COVERAGE,
                "logged": t.id in captured,
            }
            for t in tracks
        ],
        "cars": new_cars,
    }


if __name__ == "__main__":
    print("http://127.0.0.1:8090")
    uvicorn.run(app, host="127.0.0.1", port=8090)
