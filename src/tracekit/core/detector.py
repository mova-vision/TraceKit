"""Public Detector abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tracekit.core.detection import Detection


class Detector(ABC):
    """Bring-your-own model boundary.

    Implementations convert framework-specific model output into TraceKit
    ``Detection`` objects. Core TraceKit never depends on the underlying
    model stack (YOLO, PyTorch, MediaPipe, ONNX, remote APIs, etc.).
    """

    @abstractmethod
    def detect(self, frame: Any) -> list[Detection]:
        """Run detection on a single frame and return normalized detections."""
