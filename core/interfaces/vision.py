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


class PoseEstimator(Protocol):
    """Body keypoints for players the detector already found (M2.2).

    Enriches existing detections; it never decides who is a player, so a pose
    model can be swapped (YOLO-pose, RTMPose, ViTPose) without touching
    identity, tracking or the offside geometry that consumes the output.
    Returns one entry per person detection, including players whose pose
    failed — see offside/body_keypoints/estimator.py for why.
    """

    @property
    def model_name(self) -> str: ...

    def load(self) -> None: ...

    def warmup(self) -> None: ...

    def estimate(
        self,
        frame: FramePacket,
        detections: list[Detection],
    ) -> list:  # list[PlayerPose]; loose to keep core/ free of offside imports
        ...

    def estimate_batch(
        self,
        frames: list[FramePacket],
        detections_per_frame: list[list[Detection]],
    ) -> list[list]: ...
