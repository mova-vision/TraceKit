"""Interactive zone picker.

Shows a frame from the video; click to add polygon points, and the final
point list is printed in copy-paste-ready Python form for traffic_zone.py.

Keys:
    left-click   add point
    u            undo last point
    c            clear all points
    n            next frame (fine-tune where the zone should sit)
    q / ESC      quit (prints the points if any)

Usage:
    python examples/zone_picker.py path/to/video.mp4 [--frame 100]
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

parser = argparse.ArgumentParser(description="Pick a zone polygon from a video frame")
parser.add_argument("video", help="Path to the video file")
parser.add_argument("--frame", type=int, default=0, help="Frame number to start from")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if args.frame:
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)

ok, frame = cap.read()
if not ok:
    raise SystemExit("Could not read a frame from the video")

points: list[tuple[int, int]] = []


def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))


cv2.namedWindow("zone picker", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("zone picker", on_mouse)

while True:
    view = frame.copy()
    for p in points:
        cv2.circle(view, p, 4, (0, 255, 0), -1)
    if len(points) > 1:
        cv2.polylines(view, [np.array(points, dtype=np.int32).reshape(-1, 1, 2)],
                      False, (255, 255, 255), 2)
    cv2.putText(view, f"frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{total}  "
                      f"points: {len(points)}  [u]ndo [c]lear [n]ext-frame [q]uit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("zone picker", view)
    key = cv2.waitKey(30) & 0xFF
    if key == ord("u"):
        points.pop()
    elif key == ord("c"):
        points.clear()
    elif key == ord("n"):
        ok, frame = cap.read()
        if not ok:
            print("End of video")
            break
    elif key in (ord("q"), 27):
        break

cv2.destroyAllWindows()
cap.release()

if points:
    print("\nPaste into traffic_zone.py:\n")
    print("ZONE = Zone(")
    print('    name="gate",')
    print(f"    points={points},")
    print("    min_coverage=0.5,")
    print(")")
