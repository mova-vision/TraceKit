"""Track-identity evaluation: ID switches between tracker output and ground truth.

Per frame, each ground-truth object is matched to the tracker output one-to-one
(Hungarian on centre distance, matched if closer than ``dist_thr`` object
sizes). Counting then happens at the end:

- ``fragmentations``: extra track IDs used for one GT object (lost and
  re-born with a new ID)
- ``merges``: extra GT objects carried by one track ID (a true ID swap)
- ``idsw``: fragmentations + merges (single headline number; 0 is perfect)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from tracekit.tracking._hungarian import assign

if TYPE_CHECKING:  # pragma: no cover
    from tracekit.core.track import Track


class IdSwitchEvaluator:
    """Accumulate per-frame ground-truth/output matches; read ``.idsw`` at the end."""

    def __init__(self, dist_thr: float = 0.5) -> None:
        self.dist_thr = dist_thr
        self._gt_ids: dict[int, set[int]] = {}
        self._id_gts: dict[int, set[int]] = {}
        self.fn = 0
        self.gt_total = 0

    def update(
        self,
        gt: Sequence[tuple[int, tuple[float, float, float, float]]],
        outs: Sequence[Track],
    ) -> None:
        """Feed one frame: ``gt`` = (object_id, (x1, y1, x2, y2)) boxes."""
        if not gt:
            return
        self.gt_total += len(gt)
        if not outs:
            self.fn += len(gt)
            return
        g = np.array([b for _, b in gt], dtype=float)
        o = np.array([t.bbox for t in outs if t.bbox is not None], dtype=float)
        if len(o) != len(outs):  # some outputs have no bbox yet
            self.fn += len(gt)
            return
        centers_g = g[:, :2] + g[:, 2:4] / 2
        centers_o = o[:, :2] + o[:, 2:4] / 2
        D = np.linalg.norm(centers_g[:, None] - centers_o[None], axis=2)
        D = D / np.maximum(g[:, None, 2:4].max(axis=2), 1.0)  # in object sizes
        pairs = assign(D)
        hit = 0
        for i, j in pairs:
            if D[i, j] > self.dist_thr:
                continue
            hit += 1
            gid = gt[i][0]
            self._gt_ids.setdefault(gid, set()).add(outs[j].id)
            self._id_gts.setdefault(outs[j].id, set()).add(gid)
        self.fn += len(gt) - hit

    @property
    def fragmentations(self) -> int:
        """Extra track IDs used for one GT object (lost → re-born)."""
        return sum(len(v) - 1 for v in self._gt_ids.values() if v)

    @property
    def merges(self) -> int:
        """Extra GT objects carried by one track ID (a true ID swap)."""
        return sum(len(v) - 1 for v in self._id_gts.values() if v)

    @property
    def idsw(self) -> int:
        """Fragmentations + merges — 0 is perfect."""
        return self.fragmentations + self.merges
