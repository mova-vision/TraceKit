"""Tracked object state owned by TraceKit."""

from __future__ import annotations

from dataclasses import dataclass, field

from tracekit.core.detection import Detection


@dataclass(slots=True)
class Track:
    """Persistent tracked object with accumulated path."""

    id: int
    position: tuple[float, float]
    path: list[tuple[float, float]] = field(default_factory=list)
    label: str = ""
    confidence: float = 0.0
    bbox: tuple[float, float, float, float] | None = None
    detection: Detection | None = None
    missed: int = 0
    """Frames since this track last matched a detection (0 = fresh)."""

    def update_from_detection(self, detection: Detection) -> None:
        """Refresh current state and append position to the path."""
        assert detection.position is not None
        self.position = detection.position
        self.path.append(detection.position)
        self.label = detection.label
        self.confidence = detection.confidence
        self.bbox = detection.bbox
        self.detection = detection
