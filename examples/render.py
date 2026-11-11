"""Application-side rendering helpers for TraceKit examples.

TraceKit core never draws (per design). These helpers keep the example
scripts minimal by hiding OpenCV drawing, VideoWriter plumbing, and zone
overlay logic here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from tracekit import Track, Zone

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)


class VideoSink:
    """Draw tracks (+ optional zones/counter) and write an annotated MP4.

    Output defaults to ``<source>_output.mp4`` next to the input file.
    """

    def __init__(
        self,
        source: str | None = None,
        out: str | None = None,
        *,
        fps: float | None = None,
        zones: list[Zone] | None = None,
        overlay: Callable[[], str] | None = None,
    ) -> None:
        self._zones = zones or []
        self._overlay = overlay
        self._fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._out_path = out or self._default_out(source)

    @staticmethod
    def _default_out(source: str | None) -> str:
        if source:
            p = Path(source)
            return str(p.with_name(f"{p.stem}_output.mp4"))
        return "output.mp4"

    def _ensure_writer(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            return
        height, width = frame.shape[:2]
        self._writer = cv2.VideoWriter(
            self._out_path, cv2.VideoWriter_fourcc(*"mp4v"), self._fps or 30.0, (width, height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open output writer for {self._out_path}")

    def write(self, tracks: list[Track], frame: Any) -> None:
        """Draw one frame's tracks and append it to the output video."""
        self._ensure_writer(frame)
        draw_tracks(frame, tracks)
        for zone in self._zones:
            draw_zone(frame, zone)
        if self._overlay is not None:
            text = self._overlay()
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, GREEN, 1, cv2.LINE_AA)
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            print(f"Saved annotated video: {self._out_path}")


def draw_tracks(frame: np.ndarray, tracks: list[Track]) -> None:
    """Boxes, IDs, centroid dots and motion paths (fresh tracks only)."""
    for track in tracks:
        if track.missed > 0:
            continue  # stale: box would trail behind the moving object
        x, y = track.position
        cv2.circle(frame, (int(x), int(y)), 4, GREEN, -1)

        if track.bbox:
            x1, y1, x2, y2 = (int(v) for v in track.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
            label = f"{track.id} {track.label}"
            cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1, cv2.LINE_AA)

        if len(track.path) > 1:
            pts = np.array(track.path, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, YELLOW, 1)


def draw_zone(frame: np.ndarray, zone: Zone) -> None:
    """Outline a zone polygon with its name."""
    pts = np.array(zone.points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], True, WHITE, 2)
    zx, zy = zone.points[0]
    cv2.putText(frame, zone.name, (int(zx), int(zy) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)
