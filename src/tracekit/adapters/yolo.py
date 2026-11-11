"""Ultralytics YOLO → TraceKit Detection adapter."""

from __future__ import annotations

from typing import Any

from tracekit.core.detection import Detection
from tracekit.core.detector import Detector


class YOLOAdapter(Detector):
    """Adapter only — YOLO is not part of TraceKit's architecture."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        *,
        device: str | None = None,
        imgsz: int = 640,
        verbose: bool = False,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for YOLOAdapter. "
                "Install with: pip install tracekit[yolo]"
            ) from exc

        self._model = YOLO(weights)
        self._device = device
        self._imgsz = imgsz
        self._verbose = verbose

    def detect(self, frame: Any) -> list[Detection]:
        results = self._model.predict(
            source=frame,
            imgsz=self._imgsz,
            device=self._device,
            verbose=self._verbose,
        )
        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names or {}
        boxes = result.boxes
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().tolist()
        confs = boxes.conf.cpu().tolist()
        clss = boxes.cls.cpu().tolist()

        for box, conf, cls_id in zip(xyxy, confs, clss):
            label = names.get(int(cls_id), str(int(cls_id)))
            x1, y1, x2, y2 = (float(v) for v in box)
            detections.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    label=str(label),
                )
            )
        return detections
