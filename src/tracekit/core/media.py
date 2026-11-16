"""Fluent media entrypoint: detect → track → on → run."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tracekit._internal.pipeline import DetectionStage, Pipeline
from tracekit.core.detection import Detection
from tracekit.core.detector import Detector

if TYPE_CHECKING:  # pragma: no cover
    from tracekit.core.track import Track


class Media:
    """Video/image source driving one or more pipelines.

    Multiple ``detect(...).track()`` chains can share one Media: frames are
    decoded once and each detector instance runs once per frame, while every
    chain keeps its own filtering, tracking, and event state.

    Media is also iterable — ``for frame in media`` yields decoded frames
    (auto-stops at end of source); pair with ``process_frame()`` for full
    per-frame control. ``stop()`` breaks out mid-iteration, ``fps`` decimates
    processing rate, ``scale`` downsizes frames before anything sees them.
    """

    def __init__(
        self,
        source: str | Path | Iterable[Any],
        *,
        fps: float | None = None,
        scale: float | None = None,
    ) -> None:
        self._source = source
        self._fps = fps
        self._scale = scale
        self._stopped = False
        self._pipelines: list[Pipeline] = []

    def detect(
        self,
        detector: Detector,
        *,
        confidence: float | None = None,
        labels: list[str] | None = None,
    ) -> DetectionStage:
        return DetectionStage(
            media=self,
            detector=detector,
            confidence=confidence,
            labels=labels,
        )

    def run(self) -> None:
        """Iterate the source once, feeding every registered pipeline."""
        if not self._pipelines:
            raise RuntimeError(
                "Nothing to run. Call .detect(...).track() before video.run()."
            )
        self._stopped = False
        for frame in self:
            self._feed(frame)

    def __iter__(self) -> Iterator[Any]:
        """Yield decoded frames (fps-decimated, scaled) until source ends."""
        self._stopped = False
        step = self._decimation_step()
        for index, frame in enumerate(self._iter_frames()):
            if self._stopped:
                return
            if step > 1 and index % step:
                continue
            if self._scale is not None:
                frame = self._resized(frame)
            yield frame

    def _decimation_step(self) -> int:
        """How many source frames to skip per processed frame (fps knob)."""
        if self._fps is None:
            return 1
        if isinstance(self._source, (str, Path)):
            from tracekit.adapters.opencv_media import video_fps

            source_fps = video_fps(self._source) or 30.0
            return max(1, round(source_fps / self._fps))
        raise ValueError(
            "fps decimation requires a file source; "
            "pace iterable sources yourself."
        )

    def _resized(self, frame: Any) -> Any:
        if self._scale is None or not hasattr(frame, "shape"):
            return frame
        from tracekit.adapters.opencv_media import resize_frame

        return resize_frame(frame, self._scale)

    def stop(self) -> None:
        """Stop iteration/run mid-source (checked before each frame)."""
        self._stopped = True

    def process_frame(self, frame: Any) -> Any:
        """Process one frame through all pipelines (streaming demos).

        Returns the single pipeline's tracks when only one is registered,
        otherwise a list of track lists in registration order.
        """
        if not self._pipelines:
            raise RuntimeError(
                "Nothing to process. Call .detect(...).track() first."
            )
        results = self._feed(frame)
        if len(results) == 1:
            return results[0]
        return results

    def _feed(self, frame: Any) -> list[list[Track]]:
        """Run each unique detector once, then ingest into every pipeline."""
        detections: dict[int, list[Detection]] = {}
        results: list[list[Track]] = []
        for pipeline in self._pipelines:
            key = id(pipeline._detector)
            if key not in detections:
                detections[key] = pipeline._detector.detect(frame)
            results.append(pipeline.ingest(frame, detections[key]))
        return results

    def _add_pipeline(self, pipeline: Pipeline) -> None:
        self._pipelines.append(pipeline)

    def _iter_frames(self) -> Iterator[Any]:
        source = self._source
        if isinstance(source, (str, Path)):
            from tracekit.adapters.opencv_media import iter_video_frames

            yield from iter_video_frames(source)
            return

        yield from source
