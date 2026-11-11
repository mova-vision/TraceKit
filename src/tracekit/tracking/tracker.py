"""Minimal centroid-based multi-object tracker."""

from __future__ import annotations

import math

from tracekit.core.detection import Detection
from tracekit.core.track import Track


class CentroidTracker:
    """Assign persistent IDs by nearest-centroid association.

    Internal implementation detail — replaceable without changing the public API.
    """

    def __init__(self, max_distance: float = 80.0, max_missed: int = 15) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, Track] = {}
        self._missed: dict[int, int] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()
        self._missed.clear()

    def update(self, detections: list[Detection]) -> list[Track]:
        if not self._tracks:
            for detection in detections:
                self._spawn(detection)
            return list(self._tracks.values())

        track_ids = list(self._tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(detections)))
        assignments: list[tuple[int, int]] = []

        # Greedy nearest matching
        pairs: list[tuple[float, int, int]] = []
        for tid in track_ids:
            tx, ty = self._tracks[tid].position
            for di, detection in enumerate(detections):
                assert detection.position is not None
                dx, dy = detection.position
                dist = math.hypot(tx - dx, ty - dy)
                if dist <= self.max_distance:
                    pairs.append((dist, tid, di))
        pairs.sort(key=lambda item: item[0])

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, tid, di in pairs:
            if tid in used_tracks or di in used_dets:
                continue
            assignments.append((tid, di))
            used_tracks.add(tid)
            used_dets.add(di)

        unmatched_tracks -= used_tracks
        unmatched_dets -= used_dets

        for tid, di in assignments:
            self._tracks[tid].update_from_detection(detections[di])
            self._tracks[tid].missed = 0
            self._missed[tid] = 0

        for tid in unmatched_tracks:
            self._missed[tid] = self._missed.get(tid, 0) + 1
            self._tracks[tid].missed = self._missed[tid]
            if self._missed[tid] > self.max_missed:
                del self._tracks[tid]
                del self._missed[tid]

        for di in unmatched_dets:
            self._spawn(detections[di])

        return list(self._tracks.values())

    def _spawn(self, detection: Detection) -> Track:
        assert detection.position is not None
        track = Track(
            id=self._next_id,
            position=detection.position,
            path=[detection.position],
            label=detection.label,
            confidence=detection.confidence,
            bbox=detection.bbox,
            detection=detection,
        )
        self._tracks[self._next_id] = track
        self._missed[self._next_id] = 0
        self._next_id += 1
        return track
