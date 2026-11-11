"""Demo: both ways of driving Media — run() and the manual loop.

Uses samples/car_highway.mp4. Watch the terminal:

    python examples/media_loop_demo.py
"""

from __future__ import annotations

from tracekit import Media, YOLOAdapter, Zone

SRC = "./samples/car_highway.mp4"
det = YOLOAdapter(weights="models/yolov8n.pt", imgsz=480)

gate = Zone(name="gate", points=[(100, 150), (540, 150), (540, 300), (100, 300)],
            min_coverage=0.3)

# ── Door 1: run() — library drives the loop ────────────────────────────────
print("=== 1. run(): plain ===")
video = Media(SRC)
cars = video.detect(detector=det, confidence=0.4, labels=["car"]).track()
n = 0
cars.on_update(lambda e: None)
video.run()
print(f"processed all frames, final tracks: {len(cars.tracks)}")

# ── Door 1b: run() with knobs — fps decimation + scale ─────────────────────
print("\n=== 1b. run(fps=10, scale=0.5): same source, ~1/3 frames ===")
video = Media(SRC, fps=10, scale=0.5)
cars = video.detect(detector=det, confidence=0.4, labels=["car"]).track()
cars.on_enter(gate, lambda e: print(f"  enter car {e.track.id}"))
video.run()
print("done (3x faster than door 1)")

# ── Door 2: manual loop — YOU drive, auto-stop + manual stop ───────────────
print("\n=== 2. for frame in video: full control, stops after 20 frames ===")
video = Media(SRC)
cars = video.detect(detector=det, confidence=0.4, labels=["car"]).track()

i = 0
for frame in video:                      # decoded frames; auto-stops at file end
    tracks = video.process_frame(frame)  # feeds ALL chains + events
    i += 1
    print(f"  frame {i}: {len(tracks)} car(s)")
    if i >= 20:
        video.stop()                     # manual stop mid-source
print(f"stopped early after {i} frames")

print("\nAll three paths work. Same pipeline, same events, same zones.")
