"""Timestamp normalization and frame identity. See architecture.md section 11.

One stable timeline for the whole application:

    source PTS
  + stream time base
  + internal monotonically increasing frame_id
  + capture timestamp (wall clock, for latency measurement only)

Wall-clock time is never used to order frames. Every candidate must trace back
to a deterministic frame_id, and stream discontinuities must be detectable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedTimestamp:
    frame_id: int
    pts: int | None
    timestamp_ms: int
    capture_timestamp_ms: int
    is_discontinuity: bool


class TimestampNormalizer:
    """Converts source PTS into a monotonic application timeline.

    A discontinuity (stream restart, seek, camera/source switch) is detected
    when PTS jumps backwards or forwards by more than `discontinuity_threshold_ms`.
    On discontinuity the normalizer starts a new *segment*: frame_id keeps
    increasing monotonically (it is the stable identity), but timestamp_ms
    continues from where the previous segment left off rather than teleporting.

    That means downstream code can keep treating timestamp_ms as monotonic,
    while `is_discontinuity` tells tracking/velocity logic to break continuity
    (architecture.md section 26 — never compute motion across a cut).
    """

    def __init__(
        self,
        time_base: tuple[int, int],
        discontinuity_threshold_ms: int = 2000,
        start_frame_id: int = 0,
    ):
        numerator, denominator = time_base
        if numerator <= 0 or denominator <= 0:
            raise ValueError(f"Invalid time_base: {time_base}")

        self._time_base_ms = (numerator / denominator) * 1000.0
        self._discontinuity_threshold_ms = discontinuity_threshold_ms
        self._next_frame_id = start_frame_id

        self._segment_index = 0
        self._segment_origin_pts: int | None = None
        self._segment_offset_ms: float = 0.0
        self._last_timestamp_ms: float = 0.0
        self._last_pts_ms: float | None = None

    @property
    def segment_index(self) -> int:
        return self._segment_index

    def normalize(self, pts: int | None) -> NormalizedTimestamp:
        frame_id = self._next_frame_id
        self._next_frame_id += 1

        capture_ms = int(time.monotonic() * 1000)

        # No PTS available (rare, some live sources): fall back to continuing
        # the timeline, flagged as no discontinuity so playback stays smooth.
        if pts is None:
            self._last_timestamp_ms += 1.0
            return NormalizedTimestamp(
                frame_id=frame_id,
                pts=None,
                timestamp_ms=int(self._last_timestamp_ms),
                capture_timestamp_ms=capture_ms,
                is_discontinuity=False,
            )

        pts_ms = pts * self._time_base_ms

        if self._segment_origin_pts is None:
            self._segment_origin_pts = pts
            self._segment_offset_ms = 0.0
            self._last_pts_ms = pts_ms
            self._last_timestamp_ms = 0.0
            return NormalizedTimestamp(
                frame_id=frame_id,
                pts=pts,
                timestamp_ms=0,
                capture_timestamp_ms=capture_ms,
                is_discontinuity=False,
            )

        delta_ms = pts_ms - (self._last_pts_ms or pts_ms)
        is_discontinuity = abs(delta_ms) > self._discontinuity_threshold_ms

        if is_discontinuity:
            # Start a new segment anchored at the current output time, so the
            # application timeline never jumps backwards.
            self._segment_index += 1
            self._segment_origin_pts = pts
            self._segment_offset_ms = self._last_timestamp_ms
            timestamp_ms = self._last_timestamp_ms
        else:
            origin_ms = self._segment_origin_pts * self._time_base_ms
            timestamp_ms = self._segment_offset_ms + (pts_ms - origin_ms)

        self._last_pts_ms = pts_ms
        self._last_timestamp_ms = timestamp_ms

        return NormalizedTimestamp(
            frame_id=frame_id,
            pts=pts,
            timestamp_ms=int(timestamp_ms),
            capture_timestamp_ms=capture_ms,
            is_discontinuity=is_discontinuity,
        )

    def reset(self) -> None:
        """Reset segment state (e.g. after a stream reconnect).

        frame_id deliberately does NOT reset — it is the stable, globally
        unique identity for a frame within a session.
        """
        self._segment_index += 1
        self._segment_origin_pts = None
        self._segment_offset_ms = self._last_timestamp_ms
        self._last_pts_ms = None
