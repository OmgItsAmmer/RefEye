"""T-DEED adapter: loads the real checkpoint and produces real candidates.

Skips cleanly if the checkpoint is absent (it is a ~50MB binary asset that
does not belong in git — see models/tdeed/README or the operator setup docs).
When present, this proves the vendored architecture genuinely matches the
checkpoint (a strict state_dict load) rather than trusting the wiring alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai.action_spotting.tdeed.adapter import TDeedActionSpotter
from core.domain.models import FramePacket
from core.interfaces.ai import AnalysisClip

CHECKPOINT = Path("models/tdeed/checkpoint_best.pt")


def make_frame(frame_id: int):
    import numpy as np

    return FramePacket(
        frame_id=frame_id,
        pts=frame_id,
        timestamp_ms=frame_id * 40,
        capture_timestamp_ms=frame_id * 40,
        width=960,
        height=540,
        source_id="test",
        image=np.zeros((540, 960, 3), dtype=np.uint8),
    )


@pytest.fixture(scope="module")
def spotter():
    if not CHECKPOINT.exists():
        pytest.skip(
            "models/tdeed/checkpoint_best.pt not present — download "
            "SoccerNetBall_challenge1 from the T-DEED repo to run this test"
        )

    instance = TDeedActionSpotter(checkpoint=str(CHECKPOINT), device="cpu")
    instance.load()
    return instance


def test_checkpoint_loads_strictly(spotter):
    """A strict state_dict load already happened in load(); reaching here
    without a ModelLoadError proves the vendored architecture matches."""
    assert spotter.model_version == "SoccerNetBall_challenge1"


def test_supported_actions_are_football_actions(spotter):
    actions = spotter.supported_actions
    assert "pass" in actions
    assert "shot" in actions
    # Never diving/gymnastics/skating classes from another T-DEED dataset.
    assert "twist" not in actions

def test_warmup_does_not_raise(spotter):
    spotter.warmup()


def test_infer_returns_football_domain_candidates(spotter):
    frames = [make_frame(i) for i in range(100)]
    clip = AnalysisClip(
        request_id="test",
        start_frame_id=0,
        end_frame_id=99,
        start_timestamp_ms=0,
        end_timestamp_ms=99 * 40,
        frames=frames,
    )

    candidates = spotter.infer(clip)

    for candidate in candidates:
        assert candidate.source_model == "tdeed:SoccerNetBall_challenge1"
        assert candidate.action_type in spotter.supported_actions
        assert 0.0 <= candidate.model_score <= 1.0


def test_infer_returns_too_few_frames_gracefully(spotter):
    """A window shorter than the model's clip_len must not crash."""
    frames = [make_frame(i) for i in range(5)]
    clip = AnalysisClip(
        request_id="test", start_frame_id=0, end_frame_id=4,
        start_timestamp_ms=0, end_timestamp_ms=200, frames=frames,
    )
    assert spotter.infer(clip) == []


def test_padded_windows_align_scores_to_the_correct_real_frame(spotter):
    """Regression: a clip shorter than clip_len is zero-padded at the FRONT
    for the model's sliding window (matching T-DEED's own dataset loader).

    The bug this guards against is subtler than an out-of-range frame_id: a
    missing pad-offset does not push candidates outside the valid range (every
    strided_frames[i] is always a real frame) — it silently attributes a score
    that came from an all-zero PADDING frame to a real frame near the start of
    the clip instead. That misattribution is invisible to a "frame_id is
    valid" check, so this test verifies the actual index arithmetic directly:
    the returned scores array must have exactly one row per real subsampled
    frame, not one row per model-input frame (which includes padding).
    """
    clip_len = spotter._args.clip_len
    stride = 2  # _STRIDE_SNB
    # Short enough that stride-2 subsampling drops well below clip_len,
    # forcing padding — the exact condition that triggered the bug.
    real_frame_count = 65
    frames = [make_frame(i) for i in range(real_frame_count * stride)]

    scores = spotter._run_sliding_windows(frames)

    assert scores.shape[0] == real_frame_count, (
        f"expected one score row per real subsampled frame ({real_frame_count}), "
        f"got {scores.shape[0]} — padding rows were not removed before return, "
        f"which misattributes padded-input scores to real frames downstream"
    )


def test_candidate_frame_ids_stay_within_the_source_clip(spotter):
    """Every candidate's frame_id must trace back to a real source frame."""
    frames = [make_frame(i) for i in range(1000, 1130)]
    clip = AnalysisClip(
        request_id="test",
        start_frame_id=frames[0].frame_id,
        end_frame_id=frames[-1].frame_id,
        start_timestamp_ms=frames[0].timestamp_ms,
        end_timestamp_ms=frames[-1].timestamp_ms,
        frames=frames,
    )

    candidates = spotter.infer(clip)

    valid_ids = {f.frame_id for f in frames}
    for candidate in candidates:
        assert candidate.anchor_frame_id in valid_ids, (
            f"candidate anchored at frame {candidate.anchor_frame_id}, "
            f"outside the source clip's real frame range "
            f"[{frames[0].frame_id}, {frames[-1].frame_id}]"
        )
