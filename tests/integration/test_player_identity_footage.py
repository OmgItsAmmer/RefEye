"""Player identity against the real models and real broadcast footage.

Skips cleanly when either checkpoint is absent (both gitignored — see
models/pose/README.md).

What only real footage answers: whether identities actually survive a
broadcast's motion blur, overlapping players and camera movement, or whether
the tracker fragments into a new id every few frames. Fragmentation is the
quiet failure mode here — it produces no error, just a pipeline that never
accumulates enough evidence about anybody to trust them.

The reference clip is a development fixture, not a design target (M2_Plan
section 4), so the assertions stay behavioural and avoid pinning numbers to
this one video.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.domain.models import FramePacket
from offside.player_identity import IdentityState, IdentityTracker

POSE_CHECKPOINT = Path("models/pose/yolo11n-pose.pt")
DETECTOR_CHECKPOINT = Path("models/detector/yolo11n.pt")
CLIP = Path("data/videos/client_m2_test_video.mp4")

RUN_START = 90
RUN_LENGTH = 25


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


@pytest.fixture(scope="module")
def frames() -> list[FramePacket]:
    if not CLIP.exists():
        pytest.skip(f"{CLIP} not present — the client reference clip is not in git")

    import cv2

    capture = cv2.VideoCapture(str(CLIP))
    packets: list[FramePacket] = []
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, RUN_START)
        for offset in range(RUN_LENGTH):
            ok, image = capture.read()
            if not ok:
                break
            frame_id = RUN_START + offset
            height, width = image.shape[:2]
            packets.append(
                FramePacket(
                    frame_id=frame_id,
                    pts=frame_id,
                    timestamp_ms=int(frame_id * 1000 / 30),
                    capture_timestamp_ms=int(frame_id * 1000 / 30),
                    width=width,
                    height=height,
                    source_id="client_m2_test_video",
                    image=image,
                )
            )
    finally:
        capture.release()

    if len(packets) < 10:
        pytest.skip("could not decode a run of frames from the reference clip")
    return packets


@pytest.fixture(scope="module")
def poses(frames):
    if not (POSE_CHECKPOINT.exists() and DETECTOR_CHECKPOINT.exists()):
        pytest.skip("model checkpoints not present — see models/pose/README.md")

    from offside.body_keypoints.estimator import YoloPoseEstimator
    from vision.detection.yolo_detector import YoloObjectDetector

    detector = YoloObjectDetector(
        checkpoint=str(DETECTOR_CHECKPOINT),
        confidence_threshold=0.35,
        ball_confidence_threshold=0.15,
        device=_device(),
        imgsz=1280,
    )
    detector.load()
    estimator = YoloPoseEstimator(checkpoint=str(POSE_CHECKPOINT), device=_device())
    estimator.load()
    return estimator.estimate_batch(frames, detector.detect_batch(frames))


@pytest.fixture(scope="module")
def results(frames, poses):
    tracker = IdentityTracker()
    return [
        tracker.update(frame, frame_poses)
        for frame, frame_poses in zip(frames, poses)
    ]


def test_players_are_actually_followed_on_this_footage(results):
    """Fixture guard: if identity collapsed to nothing, everything below
    would pass vacuously."""
    assert all(result.identities for result in results)
    assert len(results[-1].identities) >= 5


def test_identities_do_not_fragment_every_few_frames(results):
    """The quiet failure: a tracker that mints a new id constantly produces no
    error at all, it just starves every later stage of evidence."""
    seen: set[str] = set()
    for result in results:
        seen.update(identity.track_id for identity in result.identities)

    players_per_frame = max(len(result.identities) for result in results)
    # Some churn is expected at the edges of frame as play moves; an id count
    # several times the player count would mean the tracking is not working.
    assert len(seen) < players_per_frame * 3, (
        f"{len(seen)} identities for at most {players_per_frame} players on screen"
    )


def test_most_players_become_confirmed_as_evidence_accumulates(results):
    first, last = results[0], results[-1]

    assert all(i.state is IdentityState.TENTATIVE for i in first.identities)
    confirmed = [i for i in last.identities if i.state is IdentityState.CONFIRMED]
    assert len(confirmed) >= len(last.identities) * 0.5


def test_every_pose_gets_an_identity_stamped_on_it(frames, poses, results):
    """Identity belongs to the tracker, and M2.3 pools its colour votes
    against exactly these ids — a pose without one silently loses its
    accumulated evidence."""
    for frame_poses in poses:
        assert all(pose.track_id for pose in frame_poses)


def test_contested_identities_are_reported_rather_than_resolved_silently(results):
    for result in results:
        for identity in result.contested():
            assert identity.needs_confirmation
            assert identity.confidence < 0.5
        if result.contested():
            assert result.warnings


def test_the_same_footage_gives_the_same_identities_twice(frames, poses):
    """A verdict that changes between two runs of the same clip is not
    something an operator can be asked to trust."""

    def run() -> list[str]:
        tracker = IdentityTracker()
        ids: list[str] = []
        for frame, frame_poses in zip(frames, poses):
            result = tracker.update(frame, frame_poses)
            ids.extend(identity.track_id for identity in result.identities)
        return ids

    assert run() == run()
