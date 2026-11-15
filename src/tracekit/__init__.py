"""TraceKit — infrastructure between your vision model and application logic."""

from tracekit.adapters.scope import Scope
from tracekit.adapters.yolo import YOLOAdapter
from tracekit.core.detection import Detection
from tracekit.core.detector import Detector
from tracekit.core.events import Event, EventProducer
from tracekit.core.lifecycle import LifecycleWatcher
from tracekit.core.media import Media
from tracekit.core.track import Track
from tracekit.spatial.line import Line
from tracekit.spatial.zone import Zone

__all__ = [
    "Detection",
    "Detector",
    "Event",
    "EventProducer",
    "LifecycleWatcher",
    "Line",
    "Media",
    "Scope",
    "Track",
    "YOLOAdapter",
    "Zone",
]

__version__ = "0.1.0"
