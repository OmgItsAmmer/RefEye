"""Review session — frozen frames the operator can step through.

Why this exists: the decoded rolling buffer keeps evicting while the operator
reviews, because live video never stops. Without a snapshot, frame-by-frame
navigation would work for a few seconds and then start returning blanks as
the buffer moved on underneath it.

So at analysis time we copy a bounded strip of frames around each candidate
and hold them for the session. Frames are JPEG-encoded rather than kept as
raw BGR: a ±15-frame strip for 5 candidates is ~150 frames, which as raw
960x540 BGR would be ~230 MB, and as JPEG is a few MB. Memory stays bounded
(architecture.md section 36) and review stays responsive.
"""

from __future__ import annotations

import cv2
import numpy as np

from core.domain.models import AnalysisResult, FramePacket, RefinedCandidate
from observability.logging.setup import get_logger

logger = get_logger(__name__)


class ReviewStrip:
    """Frozen, JPEG-encoded frames around one candidate."""

    def __init__(self, frames: dict[int, bytes], center_frame_id: int):
        self._frames = frames
        self._ordered = sorted(frames.keys())
        self.center_frame_id = center_frame_id

    @property
    def frame_ids(self) -> list[int]:
        return list(self._ordered)

    @property
    def is_empty(self) -> bool:
        return not self._ordered

    def nearest_index(self, frame_id: int) -> int:
        if not self._ordered:
            return 0
        return min(
            range(len(self._ordered)),
            key=lambda i: abs(self._ordered[i] - frame_id),
        )

    def frame_at(self, index: int) -> tuple[int, np.ndarray] | None:
        if not self._ordered:
            return None
        index = max(0, min(index, len(self._ordered) - 1))
        frame_id = self._ordered[index]
        buffer = np.frombuffer(self._frames[frame_id], dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            return None
        return frame_id, image

    def total_bytes(self) -> int:
        return sum(len(data) for data in self._frames.values())

    def frames_before(self, frame_id: int) -> list[tuple[int, np.ndarray]]:
        """Every held frame strictly earlier than `frame_id`, oldest first.

        For the offside pipeline's identity-tracking warm-up
        (`OffsidePipeline.warm_up`): the strip already holds exactly the
        run-up to a confirm, so no second trip to the live buffer is needed.
        """
        frames = []
        for earlier_id in self._ordered:
            if earlier_id >= frame_id:
                break
            buffer = np.frombuffer(self._frames[earlier_id], dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is not None:
                frames.append((earlier_id, image))
        return frames


class ReviewSession:
    """Everything the review UI needs, decoupled from the live buffer."""

    def __init__(
        self,
        request_id: str,
        result: AnalysisResult,
        strips: list[ReviewStrip],
    ):
        self.request_id = request_id
        self.result = result
        self._strips = strips

        self._candidate_index = 0
        self._frame_index = 0
        # Candidates arrive ranked (best first). Stepping through the play
        # ("second pass", "third pass", ...) needs chronological order
        # instead, so the operator's Up/Down presses walk the play in the
        # order it happened rather than in confidence order — this is just
        # `result.candidates` indices sorted by when each one occurred.
        self._chronological_order = sorted(
            range(len(self.candidates)),
            key=lambda i: self.candidates[i].refined_frame_id,
        )
        self._sync_frame_to_candidate()

    # -- candidates ---------------------------------------------------------

    @property
    def candidates(self) -> list[RefinedCandidate]:
        return self.result.candidates

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def candidate_index(self) -> int:
        return self._candidate_index

    @property
    def chronological_order(self) -> list[int]:
        """Candidate indices (into `candidates`) ordered by when each event
        happened in the play, earliest first — pass 1, pass 2, pass 3, ..."""
        return list(self._chronological_order)

    @property
    def current_candidate(self) -> RefinedCandidate | None:
        if not self.candidates:
            return None
        return self.candidates[self._candidate_index]

    def select_candidate(self, index: int) -> None:
        if not self.candidates:
            return
        self._candidate_index = max(0, min(index, len(self.candidates) - 1))
        self._sync_frame_to_candidate()

    def next_candidate(self) -> None:
        """Jump to the next event in the play, in the order it happened —
        e.g. pass 1 -> pass 2 -> pass 3 -> the shot. Not confidence order."""
        self._select_chronological(offset=1)

    def previous_candidate(self) -> None:
        """Jump to the previous event in the play, in play order."""
        self._select_chronological(offset=-1)

    def _select_chronological(self, offset: int) -> None:
        if not self._chronological_order:
            return
        position = self._chronological_order.index(self._candidate_index)
        position = max(0, min(position + offset, len(self._chronological_order) - 1))
        self.select_candidate(self._chronological_order[position])

    def jump_to_best(self) -> None:
        """Candidates arrive ranked, so the best is always index 0."""
        self.select_candidate(0)

    # -- frames -------------------------------------------------------------

    @property
    def current_strip(self) -> ReviewStrip | None:
        if not self._strips:
            return None
        return self._strips[self._candidate_index]

    def current_frame(self) -> tuple[int, np.ndarray] | None:
        strip = self.current_strip
        if strip is None:
            return None
        return strip.frame_at(self._frame_index)

    def next_frame(self) -> None:
        strip = self.current_strip
        if strip is not None:
            self._frame_index = min(self._frame_index + 1, len(strip.frame_ids) - 1)

    def previous_frame(self) -> None:
        self._frame_index = max(0, self._frame_index - 1)

    @property
    def frame_position(self) -> tuple[int, int]:
        """(1-based position, total) within the current candidate's strip."""
        strip = self.current_strip
        if strip is None or strip.is_empty:
            return (0, 0)
        return (self._frame_index + 1, len(strip.frame_ids))

    @property
    def frame_offset_from_candidate(self) -> int:
        """How far the operator has stepped from the AI's chosen frame."""
        strip = self.current_strip
        current = self.current_frame()
        if strip is None or current is None:
            return 0
        return current[0] - strip.center_frame_id

    def _sync_frame_to_candidate(self) -> None:
        strip = self.current_strip
        candidate = self.current_candidate
        if strip is None or candidate is None:
            self._frame_index = 0
            return
        self._frame_index = strip.nearest_index(candidate.refined_frame_id)

    def total_bytes(self) -> int:
        return sum(strip.total_bytes() for strip in self._strips)


def build_review_session(
    result: AnalysisResult,
    frames: list[FramePacket],
    frames_each_side: int = 15,
    jpeg_quality: int = 85,
) -> ReviewSession:
    """Freeze a bounded strip of frames around each candidate.

    `frames` is the clip the analysis ran on, so everything needed is already
    in hand — no second trip to the live buffer, which may already have moved
    on past these frames.
    """
    by_id = {f.frame_id: f for f in frames if f.image is not None}
    ordered_ids = sorted(by_id)

    strips: list[ReviewStrip] = []
    for candidate in result.candidates:
        center = candidate.refined_frame_id
        window = [
            frame_id
            for frame_id in ordered_ids
            if abs(frame_id - center) <= frames_each_side
        ]

        encoded: dict[int, bytes] = {}
        for frame_id in window:
            image = by_id[frame_id].image
            ok, buffer = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            )
            if ok:
                encoded[frame_id] = buffer.tobytes()

        strips.append(ReviewStrip(encoded, center_frame_id=center))

    session = ReviewSession(result.request_id, result, strips)

    logger.debug(
        "review_session_built",
        request_id=result.request_id,
        candidates=len(result.candidates),
        frames_held=sum(len(s.frame_ids) for s in strips),
        memory_kb=session.total_bytes() // 1024,
    )
    return session
