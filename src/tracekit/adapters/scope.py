"""Scoped detector wrapper: run the model only inside a Zone.

The frame is cropped to the zone's bounding box (plus a margin) *before*
the inner detector sees it — less compute and fewer false positives from
areas the application doesn't care about. Detections are translated back
into full-frame coordinates, so trackers, events, and rendering continue
to work on the original image without knowing cropping happened.
"""

from __future__ import annotations

from typing import Any

from tracekit.core.detection import Detection
from tracekit.core.detector import Detector
from tracekit.spatial.zone import Zone


class Scope(Detector):
    """Wrap any :class:`Detector` so it only looks inside one or more Zones.

    The detector's scope = the region(s) it is allowed to see. Zones are
    reused as-is for the geometry (polygon, ``min_coverage``); no separate
    ROI class is needed. Each frame:

    1. crop the frame to the zones' combined bounding box expanded by
       ``margin`` (the margin lets objects straddling the boundary be
       seen whole)
    2. run the inner detector on the crop only — once, regardless of how
       many zones were given
    3. translate detected boxes back to full-frame coordinates
    4. drop detections that belong to no zone per ``zone.is_inside``
       (``min_coverage=0.0`` = any overlap counts)

    Multiple zones act as a **union**: a detection inside any zone is kept.
    """

    def __init__(self, detector: Detector,
                 zone: Zone | list[Zone], margin: int = 20) -> None:
        if margin < 0:
            raise ValueError("margin must be non-negative")
        self._inner = detector
        self.zones = [zone] if isinstance(zone, Zone) else list(zone)
        if not self.zones:
            raise ValueError("Scope needs at least one zone")
        self.margin = margin

    def detect(self, frame: Any) -> list[Detection]:
        x0, y0, x1, y1 = self._crop_rect(frame)
        if x1 <= x0 or y1 <= y0:
            return []

        crop = frame[y0:y1, x0:x1]
        offset_dets = self._inner.detect(crop)

        shape = getattr(frame, "shape", None)
        h, w = (shape[0], shape[1]) if shape is not None else (None, None)

        out: list[Detection] = []
        for det in offset_dets:
            bx1, by1, bx2, by2 = det.bbox
            bx1, by1, bx2, by2 = bx1 + x0, by1 + y0, bx2 + x0, by2 + y0
            if h is not None:  # clip to the real frame edges
                bx1, by1 = max(0.0, bx1), max(0.0, by1)
                bx2, by2 = min(float(w), bx2), min(float(h), by2)
            if bx2 <= bx1 or by2 <= by1:
                continue
            center = ((bx1 + bx2) / 2.0, (by1 + by2) / 2.0)
            if not any(z.is_inside((bx1, by1, bx2, by2), center)
                       for z in self.zones):
                continue
            out.append(Detection(
                bbox=(bx1, by1, bx2, by2),
                confidence=det.confidence,
                label=det.label,
                metadata=det.metadata,
            ))
        return out

    def _crop_rect(self, frame: Any) -> tuple[int, int, int, int]:
        """Combined zones bounding box expanded by margin, clipped to frame."""
        pts = [p for z in self.zones for p in z.points]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x0 = int(min(xs)) - self.margin
        y0 = int(min(ys)) - self.margin
        x1 = int(max(xs)) + self.margin
        y1 = int(max(ys)) + self.margin

        shape = getattr(frame, "shape", None)
        if shape is not None and len(shape) >= 2:
            x0 = max(0, x0)
            y0 = max(0, y0)
            x1 = min(int(shape[1]), x1)
            y1 = min(int(shape[0]), y1)
        return x0, y0, x1, y1
