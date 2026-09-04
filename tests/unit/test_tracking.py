"""Tracking: player association and ball trajectory continuity."""

from __future__ import annotations

from core.domain.models import Detection, FramePacket
from vision.tracking.ball_tracker import BallTracker
from vision.tracking.iou_tracker import ByteTracker, iou


def frame(frame_id: int) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        pts=frame_id,
        timestamp_ms=frame_id * 40,
        capture_timestamp_ms=frame_id * 40,
        width=960,
        height=540,
        source_id="test",
    )


def detection(frame_id, box, cls="player", conf=0.9) -> Detection:
    return Detection(
        frame_id=frame_id,
        class_name=cls,
        confidence=conf,
        bbox_xyxy=box,
        source_model="test",
    )


class TestIoU:
    def test_identical_boxes(self):
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0

    def test_disjoint_boxes(self):
        assert iou((0, 0, 10, 10), (50, 50, 60, 60)) == 0.0

    def test_partial_overlap(self):
        assert 0.0 < iou((0, 0, 10, 10), (5, 0, 15, 10)) < 1.0


class TestByteTracker:
    def test_maintains_identity_across_frames(self):
        tracker = ByteTracker(min_hits=2)
        ids = []
        for i in range(6):
            x = 100 + i * 5
            obs = tracker.update(frame(i), [detection(i, (x, 100, x + 30, 180))])
            if obs:
                ids.append(obs[0].track_id)

        assert ids, "tracker produced no confirmed observations"
        assert len(set(ids)) == 1, "identity should not change for a steadily moving player"

    def test_ignores_single_frame_flicker(self):
        """min_hits suppresses one-frame false positives."""
        tracker = ByteTracker(min_hits=2)
        obs = tracker.update(frame(0), [detection(0, (10, 10, 40, 90))])
        assert obs == []

    def test_tracks_multiple_players_separately(self):
        tracker = ByteTracker(min_hits=2)
        for i in range(4):
            observations = tracker.update(
                frame(i),
                [
                    detection(i, (100 + i, 100, 130 + i, 180)),
                    detection(i, (500 + i, 200, 530 + i, 280)),
                ],
            )
        assert len({o.track_id for o in observations}) == 2

    def test_low_confidence_detections_extend_but_never_create(self):
        """The ByteTrack second-association pass, which rescues occlusions."""
        tracker = ByteTracker(high_threshold=0.5, low_threshold=0.2, min_hits=2)
        for i in range(3):
            tracker.update(frame(i), [detection(i, (100, 100, 130, 180), conf=0.9)])

        # A weak detection keeps the existing track alive...
        observations = tracker.update(
            frame(3), [detection(3, (101, 100, 131, 180), conf=0.3)]
        )
        assert len(observations) == 1

        # ...but a weak detection somewhere new starts nothing.
        fresh = ByteTracker(high_threshold=0.5, low_threshold=0.2, min_hits=1)
        assert fresh.update(frame(0), [detection(0, (10, 10, 40, 90), conf=0.3)]) == []

    def test_camera_cut_drops_all_identities(self):
        """Identity cannot survive a cut (architecture.md section 26)."""
        tracker = ByteTracker(min_hits=1)
        for i in range(3):
            tracker.update(frame(i), [detection(i, (100, 100, 130, 180))])
        assert tracker.active_track_count > 0

        tracker.start_new_segment()
        assert tracker.active_track_count == 0
        assert tracker.segment == 1

    def test_track_is_dropped_after_sustained_misses(self):
        tracker = ByteTracker(min_hits=1, max_misses=3)
        for i in range(3):
            tracker.update(frame(i), [detection(i, (100, 100, 130, 180))])

        for i in range(3, 12):
            tracker.update(frame(i), [])

        assert tracker.active_track_count == 0


class TestBallTracker:
    def test_computes_velocity_between_observations(self):
        tracker = BallTracker()
        tracker.update(0, 0, [detection(0, (100, 100, 110, 110), cls="ball")])
        state = tracker.update(1, 40, [detection(1, (120, 100, 130, 110), cls="ball")])

        assert state.velocity is not None
        assert state.velocity[0] > 0
        assert state.speed > 0

    def test_bridges_a_short_gap_by_extrapolating(self):
        """Missing ball detections are normal (architecture.md section 25)."""
        tracker = BallTracker(max_gap_frames=5)
        tracker.update(0, 0, [detection(0, (100, 100, 110, 110), cls="ball")])
        tracker.update(1, 40, [detection(1, (120, 100, 130, 110), cls="ball")])

        gap = tracker.update(2, 80, [])
        assert gap.center is not None
        assert gap.is_interpolated
        assert gap.center[0] > 125  # continued along the previous velocity

    def test_interpolated_confidence_decays(self):
        tracker = BallTracker(max_gap_frames=5)
        tracker.update(0, 0, [detection(0, (100, 100, 110, 110), cls="ball", conf=0.9)])
        tracker.update(1, 40, [detection(1, (120, 100, 130, 110), cls="ball", conf=0.9)])

        first_gap = tracker.update(2, 80, [])
        second_gap = tracker.update(3, 120, [])
        assert second_gap.confidence < first_gap.confidence

    def test_gives_up_after_a_long_gap(self):
        tracker = BallTracker(max_gap_frames=2)
        tracker.update(0, 0, [detection(0, (100, 100, 110, 110), cls="ball")])
        tracker.update(1, 40, [detection(1, (120, 100, 130, 110), cls="ball")])

        for i in range(2, 8):
            state = tracker.update(i, i * 40, [])

        assert state.center is None
        assert not state.is_interpolated

    def test_prefers_the_detection_nearest_the_prediction(self):
        """The most confident blob is often a pitch marking, not the ball."""
        tracker = BallTracker()
        tracker.update(0, 0, [detection(0, (100, 100, 110, 110), cls="ball", conf=0.8)])
        tracker.update(1, 40, [detection(1, (120, 100, 130, 110), cls="ball", conf=0.8)])

        state = tracker.update(
            2,
            80,
            [
                detection(2, (600, 400, 610, 410), cls="ball", conf=0.95),  # far, confident
                detection(2, (140, 100, 150, 110), cls="ball", conf=0.5),   # near, weaker
            ],
        )
        assert state.center[0] < 300, "should follow the trajectory, not the loudest blob"

    def test_coasts_instead_of_snapping_to_a_distant_confident_false_positive(self):
        """When nothing near the predicted position was detected, prefer
        extrapolating over the previous fallback of grabbing the loudest
        blob anywhere on the pitch (a sock, an ad board, a face)."""
        tracker = BallTracker()
        tracker.update(0, 0, [detection(0, (100, 100, 110, 110), cls="ball", conf=0.8)])
        tracker.update(1, 40, [detection(1, (120, 100, 130, 110), cls="ball", conf=0.8)])

        state = tracker.update(
            2,
            80,
            [detection(2, (900, 800, 910, 810), cls="ball", conf=0.99)],  # far, very confident
        )
        assert state.is_interpolated
        assert state.center[0] < 300, "must not snap to the distant false positive"

    def test_camera_cut_clears_velocity(self):
        """Never compute motion across a cut (architecture.md section 26)."""
        tracker = BallTracker()
        tracker.update(0, 0, [detection(0, (100, 100, 110, 110), cls="ball")])
        tracker.update(1, 40, [detection(1, (120, 100, 130, 110), cls="ball")])

        tracker.start_new_segment()
        state = tracker.update(2, 80, [detection(2, (800, 400, 810, 410), cls="ball")])

        assert state.velocity is None, "velocity must not carry across a cut"
        assert state.segment == 1
