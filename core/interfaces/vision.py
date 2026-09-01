"""Detection / tracking interfaces. See architecture.md sections 14 and 15."""

from __future__ import annotations

from typing import Protocol

from core.domain.models import Detection, FramePacket, TrackObservation


class ObjectDetector(Protocol):
    @property
    def model_name(self) -> str: ...

    def detect(self, frame: FramePacket) -> list[Detection]: ...


class MultiObjectTracker(Protocol):
    def update(
        self,
        frame: FramePacket,
        detections: list[Detection],
    ) -> list[TrackObservation]: ...

    def reset(self) -> None: ...
