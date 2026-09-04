import pytest

from video.timestamps.normalizer import TimestampNormalizer

# 1/1000 time base: PTS is milliseconds, which keeps the arithmetic obvious.
MS_TIME_BASE = (1, 1000)


def test_frame_ids_are_monotonic_and_gapless():
    n = TimestampNormalizer(MS_TIME_BASE)
    ids = [n.normalize(pts).frame_id for pts in range(0, 500, 40)]
    assert ids == list(range(len(ids)))


def test_first_frame_anchors_timeline_at_zero():
    """A source whose PTS starts at an arbitrary offset still starts at t=0."""
    n = TimestampNormalizer(MS_TIME_BASE)
    first = n.normalize(900_000)
    assert first.timestamp_ms == 0

    second = n.normalize(900_040)
    assert second.timestamp_ms == 40


def test_timestamps_track_pts_within_a_segment():
    n = TimestampNormalizer(MS_TIME_BASE)
    for pts in (0, 40, 80, 120):
        result = n.normalize(pts)
    assert result.timestamp_ms == 120
    assert not result.is_discontinuity


def test_backwards_pts_jump_is_flagged_as_discontinuity():
    n = TimestampNormalizer(MS_TIME_BASE, discontinuity_threshold_ms=2000)
    for pts in range(0, 5000, 1000):
        n.normalize(pts)

    # File loops back to the start.
    wrapped = n.normalize(0)
    assert wrapped.is_discontinuity


def test_timeline_never_moves_backwards_across_a_discontinuity():
    """Downstream code treats timestamp_ms as monotonic; a loop must not rewind it."""
    n = TimestampNormalizer(MS_TIME_BASE, discontinuity_threshold_ms=2000)
    for pts in range(0, 5000, 1000):
        last = n.normalize(pts)

    wrapped = n.normalize(0)
    assert wrapped.timestamp_ms >= last.timestamp_ms

    following = n.normalize(1000)
    assert following.timestamp_ms >= wrapped.timestamp_ms
    assert not following.is_discontinuity


def test_frame_ids_keep_increasing_across_a_discontinuity():
    n = TimestampNormalizer(MS_TIME_BASE, discontinuity_threshold_ms=2000)
    first = n.normalize(0)
    n.normalize(1000)
    wrapped = n.normalize(0)

    assert wrapped.frame_id > first.frame_id


def test_segment_index_increments_on_discontinuity():
    n = TimestampNormalizer(MS_TIME_BASE, discontinuity_threshold_ms=2000)
    n.normalize(0)
    assert n.segment_index == 0

    n.normalize(60_000)
    assert n.segment_index == 1


def test_small_gaps_are_not_discontinuities():
    n = TimestampNormalizer(MS_TIME_BASE, discontinuity_threshold_ms=2000)
    n.normalize(0)
    result = n.normalize(500)
    assert not result.is_discontinuity


def test_missing_pts_still_produces_a_usable_frame():
    n = TimestampNormalizer(MS_TIME_BASE)
    result = n.normalize(None)
    assert result.pts is None
    assert result.frame_id == 0
    assert not result.is_discontinuity


def test_reset_preserves_frame_id_identity():
    """frame_id is the stable identity for a frame; a reconnect must not reuse it."""
    n = TimestampNormalizer(MS_TIME_BASE)
    n.normalize(0)
    n.normalize(40)
    n.reset()

    after = n.normalize(0)
    assert after.frame_id == 2


def test_rejects_invalid_time_base():
    with pytest.raises(ValueError):
        TimestampNormalizer((0, 1000))


def test_non_millisecond_time_base_converts_correctly():
    # 1/90000 is the standard MPEG-TS clock.
    n = TimestampNormalizer((1, 90_000))
    n.normalize(0)
    result = n.normalize(90_000)  # exactly one second later
    assert result.timestamp_ms == 1000
