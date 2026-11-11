"""Normalized detection representation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Detection:
    """Framework-normalized detection produced by a Detector adapter.

    Downstream TraceKit code (tracking, events, spatial) should consume this
    type only — never raw YOLO/MediaPipe/OpenCV result objects.
    """

    bbox: tuple[float, float, float, float]
    """Axis-aligned box as (x1, y1, x2, y2) in image coordinates."""

    confidence: float
    label: str
    position: tuple[float, float] | None = None
    """Centroid (cx, cy). Defaults to bbox center when omitted."""

    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.position is None:
            x1, y1, x2, y2 = self.bbox
            self.position = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
