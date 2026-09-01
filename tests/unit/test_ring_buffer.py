from core.domain.models import FramePacket
from video.buffering.ring_buffer import DecodedRingBuffer


def _make_frame(frame_id: int, timestamp_ms: int) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        pts=frame_id,
        timestamp_ms=timestamp_ms,
        capture_timestamp_ms=timestamp_ms,
        width=960,
        height=540,
        source_id="test",
    )


def test_buffer_evicts_oldest_when_full():
    buf = DecodedRingBuffer(max_frames=3)
    for i in range(5):
        buf.append(_make_frame(i, i * 33))

    assert len(buf) == 3
    assert buf.get_by_frame(0) is None
    assert buf.get_by_frame(1) is None
    assert buf.get_by_frame(4) is not None


def test_get_window_filters_by_timestamp():
    buf = DecodedRingBuffer(max_frames=10)
    for i in range(10):
        buf.append(_make_frame(i, i * 100))

    window = buf.get_window(end_timestamp_ms=900, duration_ms=300)
    timestamps = [f.timestamp_ms for f in window]

    assert timestamps == [600, 700, 800, 900]


def test_get_range_filters_by_frame_id():
    buf = DecodedRingBuffer(max_frames=10)
    for i in range(10):
        buf.append(_make_frame(i, i * 100))

    frames = buf.get_range(3, 5)
    assert [f.frame_id for f in frames] == [3, 4, 5]
