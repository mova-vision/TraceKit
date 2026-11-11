"""OpenCV-backed frame sources (optional dependency)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any


def iter_video_frames(path: str | Path) -> Iterator[Any]:
    """Yield BGR frames from a video file via OpenCV."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenCV is required for file/webcam Media sources. "
            "Install with: pip install tracekit[opencv]"
        ) from exc

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"Could not open media source: {path}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame
    finally:
        capture.release()


def video_fps(source: str | Path) -> float | None:
    """Source frame rate, or None if it cannot be determined."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenCV is required to probe video fps. "
            "Install with: pip install tracekit[opencv]"
        ) from exc

    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return fps if fps and fps > 0 else None


def resize_frame(frame: Any, scale: float) -> Any:
    """Downscale a frame by ``scale`` (0.5 = half width/height)."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenCV is required for scale. "
            "Install with: pip install tracekit[opencv]"
        ) from exc

    height, width = frame.shape[:2]
    return cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))))


def decode_image_bytes(data: bytes) -> Any:
    """Decode an encoded image (JPEG/PNG) into a BGR ndarray."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenCV and numpy are required to decode images. "
            "Install with: pip install tracekit[opencv]"
        ) from exc

    array = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image bytes")
    return frame
