import numpy as np
import pytest

from core.domain.models import FramePacket
from core.interfaces.video import EncodedPacket
from video.buffering.ring_buffer import DecodedRingBuffer, EncodedRingBuffer


def make_frame(frame_id: int, timestamp_ms: int, with_image: bool = False) -> FramePacket:
    return FramePacket(
        frame_id=frame_id,
        pts=frame_id,
        timestamp_ms=timestamp_ms,
        capture_timestamp_ms=timestamp_ms,
        width=8,
        height=8,
        source_id="test",
        image=np.zeros((8, 8, 3), dtype=np.uint8) if with_image else None,
    )


def make_packet(pts: int, keyframe: bool = False, size: int = 100) -> EncodedPacket:
    return EncodedPacket(
        data=b"\x00" * size,
        pts=pts,
        dts=pts,
        is_keyframe=keyframe,
        stream_index=0,
        size_bytes=size,
    )


class TestDecodedRingBuffer:
    def test_evicts_oldest_when_over_frame_limit(self):
        buf = DecodedRingBuffer(max_frames=3)
        for i in range(5):
            buf.append(make_frame(i, i * 33))

        assert len(buf) == 3
        assert buf.get_by_frame(0) is None
        assert buf.get_by_frame(1) is None
        assert buf.get_by_frame(4) is not None

    def test_evicts_by_duration(self):
        buf = DecodedRingBuffer(max_frames=1000, max_duration_ms=500)
        for i in range(20):
            buf.append(make_frame(i, i * 100))

        assert buf.duration_ms() <= 500
        assert buf.get_by_frame(19) is not None
        assert buf.get_by_frame(0) is None

    def test_eviction_drops_the_buffers_own_reference(self):
        """Bounded memory means the buffer stops holding the pixel data.

        Eviction must NOT mutate the evicted FramePacket in place (e.g. by
        nulling .image on the shared object) — get_window()/get_range() hand
        out these exact objects to callers such as a triggered analysis, which
        can still be reading them on another thread after eviction runs.
        Mutating a frame a caller is holding corrupted in-flight analyses
        (a frame that passed an `image is not None` check moments earlier
        would go empty mid-batch). The buffer only needs to drop its OWN
        reference; total_bytes() and get_by_frame() reflect that.
        """
        buf = DecodedRingBuffer(max_frames=2)
        first = make_frame(0, 0, with_image=True)
        buf.append(first)
        buf.append(make_frame(1, 33, with_image=True))
        buf.append(make_frame(2, 66, with_image=True))

        # The buffer itself no longer serves frame 0 or counts its bytes...
        assert buf.get_by_frame(0) is None
        assert len(buf) == 2

        # ...but a caller already holding the object still has a valid frame.
        assert first.image is not None
        assert first.image.shape == (8, 8, 3)

    def test_evicts_by_total_bytes(self):
        """A frame count alone does not bound RAM — resolution decides the cost."""
        frame_bytes = 8 * 8 * 3
        buf = DecodedRingBuffer(max_frames=1000, max_bytes=frame_bytes * 4)
        for i in range(20):
            buf.append(make_frame(i, i * 40, with_image=True))

        assert buf.total_bytes() <= frame_bytes * 4
        assert len(buf) <= 4
        # The newest frames are the ones kept.
        assert buf.get_by_frame(19) is not None

    def test_byte_accounting_returns_to_zero_after_clear(self):
        buf = DecodedRingBuffer(max_frames=10)
        for i in range(5):
            buf.append(make_frame(i, i * 40, with_image=True))
        assert buf.total_bytes() > 0

        buf.clear()
        assert buf.total_bytes() == 0
        assert len(buf) == 0

    def test_get_window_filters_by_timestamp(self):
        buf = DecodedRingBuffer(max_frames=20)
        for i in range(10):
            buf.append(make_frame(i, i * 100))

        window = buf.get_window(end_timestamp_ms=900, duration_ms=300)
        assert [f.timestamp_ms for f in window] == [600, 700, 800, 900]

    def test_get_range_filters_by_frame_id(self):
        buf = DecodedRingBuffer(max_frames=20)
        for i in range(10):
            buf.append(make_frame(i, i * 100))

        assert [f.frame_id for f in buf.get_range(3, 5)] == [3, 4, 5]

    def test_latest_returns_newest_frame(self):
        buf = DecodedRingBuffer(max_frames=5)
        assert buf.latest() is None
        for i in range(3):
            buf.append(make_frame(i, i * 40))
        assert buf.latest().frame_id == 2

    def test_frames_held_by_a_caller_survive_concurrent_eviction(self):
        """Regression: a triggered analysis reads a window via get_window(),
        then keeps processing those frames for seconds while the ingest
        thread keeps appending and evicting in the background. Before the
        fix, eviction nulled .image on the shared FramePacket object, so a
        frame that was valid when the analysis grabbed it could go empty
        mid-processing and crash (e.g. cv2.resize on an empty array).
        """
        buf = DecodedRingBuffer(max_frames=3)
        for i in range(3):
            buf.append(make_frame(i, i * 40, with_image=True))

        # Simulate "an analysis grabs the current window" — same objects
        # get_window()/get_range() would hand out, held by the caller.
        held_frames = buf.get_range(0, 2)
        assert all(f.image is not None for f in held_frames)

        # Background ingest keeps running concurrently, evicting everything
        # the analysis is holding.
        for i in range(3, 10):
            buf.append(make_frame(i, i * 40, with_image=True))

        # The frames the analysis is still holding must remain usable.
        for frame in held_frames:
            assert frame.image is not None, (
                f"frame {frame.frame_id} was corrupted by concurrent eviction "
                "while an analysis was still holding it"
            )
            assert frame.image.shape == (8, 8, 3)

    def test_rejects_non_positive_capacity(self):
        with pytest.raises(ValueError):
            DecodedRingBuffer(max_frames=0)


class TestEncodedRingBuffer:
    def test_bounded_by_packet_count(self):
        buf = EncodedRingBuffer(max_packets=4)
        for pts in range(10):
            buf.append(make_packet(pts))

        assert len(buf) == 4

    def test_bounded_by_total_bytes(self):
        buf = EncodedRingBuffer(max_packets=1000, max_bytes=500)
        for pts in range(20):
            buf.append(make_packet(pts, size=100))

        assert buf.total_bytes() <= 500

    def test_drops_native_handle_on_insert(self):
        """Holding live decoder packets would pin memory we do not control."""
        buf = EncodedRingBuffer(max_packets=4)
        packet = make_packet(0)
        packet.raw = object()
        buf.append(packet)

        assert buf.get_range(0, 0)[0].raw is None

    def test_range_extends_back_to_preceding_keyframe(self):
        """A returned range must be independently decodable."""
        buf = EncodedRingBuffer(max_packets=100)
        for pts in range(10):
            buf.append(make_packet(pts, keyframe=(pts % 5 == 0)))

        packets = buf.get_range(start_pts=7, end_pts=9)
        assert packets[0].pts == 5
        assert packets[0].is_keyframe
        assert packets[-1].pts == 9

    def test_range_empty_when_out_of_bounds(self):
        buf = EncodedRingBuffer(max_packets=10)
        for pts in range(5):
            buf.append(make_packet(pts))

        assert buf.get_range(start_pts=50, end_pts=60) == []
