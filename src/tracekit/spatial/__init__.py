"""Spatial reasoning: zones, lines, geometry (public surface)."""

from tracekit.spatial.line import Line, LineWatcher
from tracekit.spatial.proximity import ProximityWatcher
from tracekit.spatial.zone import DwellWatcher, Zone, ZoneWatcher

__all__ = [
    "Zone",
    "ZoneWatcher",
    "Line",
    "LineWatcher",
    "DwellWatcher",
    "ProximityWatcher",
]
