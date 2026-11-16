"""Small O(n³) Hungarian (Kuhn-Munkres) assignment on a cost matrix.

Pure numpy/python — avoids a scipy dependency in the core package. Solves
min-cost one-to-one assignment for ``cost`` of shape (n, m); n may exceed m.
"""
from __future__ import annotations

import numpy as np


def assign(cost: np.ndarray) -> list[tuple[int, int]]:
    """Return the optimal (row, col) assignment pairs minimizing total cost.

    Handles rectangular matrices in either direction. Partial: rows/cols left
    unmatched are simply absent from the result.
    """
    if cost.size == 0:
        return []
    a = np.asarray(cost, dtype=float)
    transposed = a.shape[0] > a.shape[1]
    if transposed:
        a = a.T
    n, m = a.shape
    cols = _hungarian(a)
    pairs = [(c, r) if transposed else (r, c) for r, c in enumerate(cols) if c >= 0]
    return pairs


def _hungarian(a: np.ndarray) -> list[int]:
    """Row -> column assignment (-1 when unassigned) for an (n, m) matrix, n <= m."""
    n, m = a.shape
    big = float(np.max(np.abs(a)) * (n + m) + 1.0) if a.size else 0.0
    cost = np.where(np.isfinite(a), a, big)
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)          # p[j] = row matched to column j (1-based)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            row = cost[i0 - 1]
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = float(row[j - 1]) - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if j1 == 0:  # should not happen with finite costs; bail safely
                break
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    cols = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            cols[p[j] - 1] = j - 1
    return cols
