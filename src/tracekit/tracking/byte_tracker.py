"""ByteTrack-style multi-object tracker: Kalman prediction + two-stage
Hungarian association with probation and lost-track recovery.

Why not the centroid tracker: greedy nearest-centroid matching re-issues IDs
whenever a detection is missed for a few frames — the new car behind picks up
the dead track's ID. This tracker instead:

1. *Predicts* every track forward with a size-scaled Kalman filter (constant
   velocity), so a missed detection does not teleport identity to whatever
   drives into the empty spot.
2. Matches with an optimal (Hungarian) fused cost: IoU + centre distance,
   gated by a Mahalanobis test on the predicted position and by class.
3. Runs a second stage that matches weak detections (blur / partial occlusion)
   to still-unmatched tracks — "byte" association from the ByteTrack paper.
4. Puts new tracks on probation: no public ID until ``confirm_hits`` frames,
   killing false-positive spawn churn.
5. Keeps lost tracks coasting up to ``max_age`` frames; a track that matches
   again keeps its ID (recovery, no switch).

Detector-agnostic: consumes normalized :class:`Detection` objects and returns
:class:`Track` objects with path accumulation, exactly like
:class:`CentroidTracker` — drop-in via ``.track(tracker=ByteTracker())``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from tracekit.core.detection import Detection
from tracekit.core.track import Track
from tracekit.tracking._hungarian import assign

_BIG: Final = 1e6


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU for boxes given as (cx, cy, w, h) rows."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
    ax2, ay2 = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx1, by1 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    bx2, by2 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    iw = np.clip(
        np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None]),
        0.0, None,
    )
    ih = np.clip(
        np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None]),
        0.0, None,
    )
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


class _Kalman:
    """Constant-velocity Kalman on state ``[cx, cy, w, h, vx, vy]``.

    All noise scales with object size, so one config fits 20 px pedestrians
    and 400 px trucks alike.
    """

    def __init__(self, box: np.ndarray, cfg: ByteTrackerConfig) -> None:
        self.x = np.array([*box[:4], 0.0, 0.0], dtype=float)
        s = max(box[2], box[3], 1.0)
        self.P = np.diag([
            (cfg.meas_pos * s) ** 2, (cfg.meas_pos * s) ** 2,
            (cfg.meas_size * s) ** 2, (cfg.meas_size * s) ** 2,
            (0.5 * s) ** 2, (0.2 * s) ** 2,
        ])
        self._cfg = cfg

    def predict(self, dt: float = 1.0) -> None:
        F = np.eye(6)
        F[0, 4] = dt
        F[1, 5] = dt
        s = max(self.x[2], self.x[3], 1.0)
        c = self._cfg
        self.x = F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        Q = np.diag([
            (c.std_pos * s * dt) ** 2, (c.std_pos * s * dt) ** 2,
            (c.std_size * s) ** 2, (c.std_size * s) ** 2,
            (c.std_vel * s) ** 2, (c.std_vel * s) ** 2,
        ])
        self.P = F @ self.P @ F.T + Q

    def _R(self, scale: float = 1.0) -> np.ndarray:
        s = max(self.x[2], self.x[3], 1.0)
        c = self._cfg
        return np.diag([
            (c.meas_pos * s * scale) ** 2, (c.meas_pos * s * scale) ** 2,
            (c.meas_size * s * scale) ** 2, (c.meas_size * s * scale) ** 2,
        ])

    def gate(self, center: np.ndarray) -> float:
        """Squared Mahalanobis distance of a detection centre (2 dof)."""
        S = (self.P[:2, :2] + self._R()[:2, :2])
        y = center - self.x[:2]
        return float(y @ np.linalg.solve(S, y))

    def update(self, box: np.ndarray, scale: float = 1.0) -> None:
        R = self._R(scale)
        S = self.P[:4, :4] + R              # H P H^T + R with H = [I 0]
        K = np.linalg.solve(S.T, self.P[:, :4].T).T   # (6,4) gain
        y = box - self.x[:4]
        self.x = self.x + K @ y             # full state update incl. velocity
        I_KH = np.eye(6)
        I_KH[:, :4] -= K                    # I - K H with H = [I 0]
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T  # Joseph form


@dataclass
class ByteTrackerConfig:
    """Tuning knobs. All noise values are fractions of object size."""

    hi_thresh: float = 0.5
    """Stage-A association pool (confidences at/above this)."""
    lo_thresh: float = 0.1
    """Stage-B rescue pool floor — run the detector with a low conf floor."""
    new_track_thresh: float = 0.6
    """Only detections at/above this confidence may start a track."""
    confirm_hits: int = 3
    """Frames on probation before a track gets its public ID."""
    max_age: int = 30
    """Frames a lost track keeps coasting (should exceed the longest gap)."""
    gate_chi2: float = 6.63
    """Mahalanobis gate on centre distance (99% chi-square, 2 dof — tighter
    gate than 13.82, keeps matches from jumping between near neighbours)."""
    w_iou: float = 0.4
    w_dist: float = 0.6
    match_thresh: float = 0.8
    """Stage-A cost cutoff; stage B uses ``match_thresh + 0.05``."""
    class_aware: bool = True
    """Never match detections across different labels."""
    dup_iou: float = 0.7
    """Do not spawn a track overlapping an existing one this much."""
    # Kalman noise (fractions of object size). Defaults suit steady traffic
    # (near-constant velocity); raise std_vel to 0.01–0.02 for objects that
    # accelerate / turn a lot.
    std_pos: float = 0.10
    std_vel: float = 0.001
    std_size: float = 0.03
    meas_pos: float = 0.12
    """Detector box-centre jitter as a fraction of object size — set to your
    detector's real jitter: lower (~0.05) for a stable detector, raise if IDs
    break up on barely-moving objects."""
    meas_size: float = 0.05
    low_conf_scale: float = 3.0
    """Weak detections nudge instead of yank (larger measurement noise)."""


class _Internal:
    """Tracker-side state wrapped around the public :class:`Track` object."""

    def __init__(self, track: Track, box: np.ndarray, cfg: ByteTrackerConfig) -> None:
        self.track = track
        self.kf = _Kalman(box, cfg)
        self.label = track.label
        self.hits = 1
        self.time_since_update = 0
        self.state = "tentative"  # tentative | confirmed | lost
        self.counted = False

    @property
    def box(self) -> np.ndarray:
        """Filtered box as (cx, cy, w, h)."""
        return self.kf.x[:4]

    def predict(self, dt: float) -> None:
        self.kf.predict(dt)
        self.time_since_update += 1

    def refresh(self, detection: Detection, low: bool, cfg: ByteTrackerConfig) -> None:
        cx, cy = detection.position  # type: ignore[misc]
        x1, y1, x2, y2 = detection.bbox
        z = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        self.kf.update(z, cfg.low_conf_scale if low else 1.0)
        self.hits += 1
        self.time_since_update = 0
        self.track.update_from_detection(detection)


class ByteTracker:
    """Two-stage Hungarian + Kalman tracker implementing the ``Tracker`` protocol.

    Pass to ``.track(tracker=ByteTracker())``. Tune ``max_age`` to at least the
    longest number of frames an object is expected to be missed, and run your
    detector with a low confidence floor (``lo_thresh``) so weak boxes survive
    into the rescue stage.
    """

    def __init__(
        self,
        config: ByteTrackerConfig | None = None,
        *,
        max_age: int | None = None,
        confirm_hits: int | None = None,
    ) -> None:
        self.cfg = config or ByteTrackerConfig()
        if max_age is not None:
            self.cfg.max_age = max_age
        if confirm_hits is not None:
            self.cfg.confirm_hits = confirm_hits
        self._tracks: list[_Internal] = []
        self._next_id = 1
        self._frame = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._frame = 0

    # ------------------------------------------------------------------ API
    def update(self, detections: list[Detection]) -> list[Track]:
        c = self.cfg
        self._frame += 1

        high = [d for d in detections if d.confidence >= c.hi_thresh]
        low = [d for d in detections if c.lo_thresh <= d.confidence < c.hi_thresh]

        for t in self._tracks:
            t.predict(1.0)

        pool = [t for t in self._tracks if t.state in ("confirmed", "lost")]
        tentative = [t for t in self._tracks if t.state == "tentative"]
        matched: set[int] = set()

        # Stage A: confirmed + lost tracks vs high-confidence detections.
        pairs, ut, ud = self._associate(pool, high, c.match_thresh)
        for ti, di in pairs:
            t = pool[ti]
            t.refresh(high[di], low=False, cfg=c)
            if t.state == "lost":
                t.state = "confirmed"
            matched.add(id(t))

        # Stage B: unmatched tracks vs weak detections (blur / occlusion rescue).
        rem_tracks = [pool[i] for i in ut]
        rem_high = [high[i] for i in ud]
        pairs, ut, low_ud = self._associate(rem_tracks, low, c.match_thresh + 0.05)
        for ti, di in pairs:
            t = rem_tracks[ti]
            t.refresh(low[di], low=True, cfg=c)
            if t.state == "lost":
                t.state = "confirmed"
            matched.add(id(t))

        # Stage C: tentative tracks vs leftover high detections.
        rem_high2 = rem_high
        pairs, _, ud2 = self._associate(tentative, rem_high2, c.match_thresh)
        for ti, di in pairs:
            t = tentative[ti]
            t.refresh(rem_high2[di], low=False, cfg=c)
            matched.add(id(t))
        leftovers = [rem_high2[i] for i in ud2]

        # Lifecycle: promote, lose, kill.
        survivors: list[_Internal] = []
        for t in self._tracks:
            if id(t) in matched:
                if t.state == "tentative" and t.hits >= c.confirm_hits:
                    t.state = "confirmed"
                survivors.append(t)
            else:
                if t.state == "tentative":
                    continue  # false positive dies silently, never had a public life
                if t.state == "confirmed":
                    t.state = "lost"
                if t.time_since_update > c.max_age:
                    continue
                survivors.append(t)
        self._tracks = survivors

        # Spawn tentative tracks from strong, non-duplicate leftovers.
        for d in leftovers:
            if d.confidence < c.new_track_thresh or self._duplicate(d):
                continue
            self._spawn(d)

        # Mirror state onto public Track objects.
        for t in self._tracks:
            t.track.missed = t.time_since_update
            if t.state == "lost":
                cx, cy, w, h = t.box
                # Coast: keep drawing at the predicted spot, but the path only
                # records observed detections.
                t.track.position = (float(cx), float(cy))
                t.track.bbox = (float(cx - w / 2), float(cy - h / 2),
                                float(cx + w / 2), float(cy + h / 2))
                t.track.confidence = 0.0

        return [t.track for t in self._tracks if t.state != "tentative"]

    # ------------------------------------------------------------ internals
    def _spawn(self, detection: Detection) -> None:
        x1, y1, x2, y2 = detection.bbox
        cx, cy = detection.position  # type: ignore[misc]
        track = Track(
            id=self._next_id,
            position=(cx, cy),
            path=[(cx, cy)],
            label=detection.label,
            confidence=detection.confidence,
            bbox=detection.bbox,
            detection=detection,
        )
        self._next_id += 1
        box = np.array([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        self._tracks.append(_Internal(track, box, self.cfg))

    def _duplicate(self, detection: Detection) -> bool:
        if not self._tracks:
            return False
        x1, y1, x2, y2 = detection.bbox
        db = np.array([[(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]])
        pb = np.array([t.box for t in self._tracks])
        return bool((_iou_matrix(pb, db) > self.cfg.dup_iou).any())

    def _associate(
        self, tracks: list[_Internal], detections: list[Detection], thresh: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Fused-cost assignment. Returns (pairs, unmatched_tracks, unmatched_dets)."""
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))
        c = self.cfg
        pb = np.array([t.box for t in tracks])
        db = np.array([
            [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]] for d in detections
        ])
        dbox = np.stack([
            (db[:, 0] + db[:, 2]) / 2, (db[:, 1] + db[:, 3]) / 2,
            db[:, 2] - db[:, 0], db[:, 3] - db[:, 1],
        ], axis=1)
        iou = _iou_matrix(pb, dbox)
        dist = np.linalg.norm(pb[:, None, :2] - dbox[None, :, :2], axis=2)
        scale = np.maximum(pb[:, None, 2:4].max(axis=2), dbox[None, :, 2:4].max(axis=2)) * 2.0
        dist_n = np.clip(dist / np.maximum(scale, 1e-9), 0.0, 1.0)
        cost = (c.w_iou * (1.0 - iou) + c.w_dist * dist_n) / (c.w_iou + c.w_dist)

        veto = np.zeros(cost.shape, dtype=bool)
        if c.class_aware:
            labels = [t.label for t in tracks]
            dlabels = [d.label for d in detections]
            veto |= np.array([[tl != dl for dl in dlabels] for tl in labels])
        for i, t in enumerate(tracks):
            for j in range(len(detections)):
                if t.kf.gate(dbox[j, :2]) > c.gate_chi2:
                    veto[i, j] = True
        cost[veto] = _BIG

        pairs = []
        for i, j in assign(cost):
            if cost[i, j] < _BIG and cost[i, j] <= thresh:
                pairs.append((i, j))
        used_t = {i for i, _ in pairs}
        used_d = {j for _, j in pairs}
        return (
            pairs,
            [i for i in range(len(tracks)) if i not in used_t],
            [j for j in range(len(detections)) if j not in used_d],
        )
