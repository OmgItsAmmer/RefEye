"""Generate a scripted synthetic clip plus ground-truth event labels.

Real match footage is not committed (licensing, and video does not belong in
git), and the development environment has no access to broadcast material.
This generates a stand-in that exercises the whole pipeline deterministically:
H.264 in MP4 with real PTS and keyframes, players, a ball, and a *scripted*
sequence of ball contacts written alongside as ground truth.

    python -m tools.video_sampling.make_sample_clip

Outputs:
    tests/fixtures/sample_match.mp4
    tests/fixtures/sample_match.labels.json   (schema per architecture.md §51)

**This is a pipeline fixture, not a CV benchmark.** The figures are coloured
rectangles, so a COCO detector sees nothing in them — use the `fixture`
detector provider for this clip and `yolo` for real footage. Nothing here
says anything about accuracy on real broadcasts; that requires client
footage (architecture.md section 21).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import av
import numpy as np

# Colours are chosen to be unambiguous for the fixture detector.
PITCH_GREEN = (40, 110, 55)
PITCH_LINE = (70, 150, 85)
BALL_COLOR = (245, 245, 245)
TEAM_A = (40, 40, 200)      # BGR: red-ish shirt
TEAM_B = (200, 70, 40)      # BGR: blue-ish shirt
KEEPER = (30, 200, 220)     # BGR: yellow keeper


@dataclass
class Player:
    x: float
    y: float
    color: tuple[int, int, int]
    role: str = "player"
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class Contact:
    """A scripted ball contact: the ground truth the pipeline should find."""

    frame: int
    action: str
    player_index: int
    notes: str = ""


@dataclass
class Script:
    """Ball waypoints and the contacts that connect them."""

    contacts: list[Contact] = field(default_factory=list)
    waypoints: list[tuple[int, float, float]] = field(default_factory=list)


def build_script(total_frames: int, width: int, height: int) -> tuple[Script, list[Player]]:
    """A cross -> header -> save -> rebound -> shot sequence.

    Deliberately includes the rebound chain from architecture.md section 28,
    because "which of these two shots did the operator mean?" is exactly the
    case the candidate list has to handle.
    """
    players = [
        Player(x=width * 0.18, y=height * 0.60, color=TEAM_A),          # 0 crosser
        Player(x=width * 0.62, y=height * 0.42, color=TEAM_A),          # 1 header
        Player(x=width * 0.86, y=height * 0.50, color=KEEPER, role="goalkeeper"),  # 2
        Player(x=width * 0.60, y=height * 0.66, color=TEAM_A),          # 3 rebound shooter
        Player(x=width * 0.45, y=height * 0.30, color=TEAM_B),          # 4 defender
    ]

    # (frame, action, player) — the ball changes direction sharply at each.
    contacts = [
        Contact(frame=int(total_frames * 0.18), action="cross", player_index=0,
                notes="deep cross from the left"),
        Contact(frame=int(total_frames * 0.40), action="header", player_index=1,
                notes="header on goal"),
        Contact(frame=int(total_frames * 0.58), action="goalkeeper_contact",
                player_index=2, notes="keeper parries"),
        Contact(frame=int(total_frames * 0.78), action="shot", player_index=3,
                notes="rebound shot"),
    ]

    waypoints = [(0, width * 0.10, height * 0.62)]
    for contact in contacts:
        player = players[contact.player_index]
        waypoints.append((contact.frame, player.x, player.y - height * 0.03))
    waypoints.append((total_frames - 1, width * 0.95, height * 0.45))

    return Script(contacts=contacts, waypoints=waypoints), players


def ball_position(script: Script, frame_index: int) -> tuple[float, float]:
    """Piecewise-linear travel between contacts, with a little arc."""
    points = script.waypoints
    for (f0, x0, y0), (f1, x1, y1) in zip(points, points[1:]):
        if f0 <= frame_index <= f1:
            span = max(f1 - f0, 1)
            t = (frame_index - f0) / span
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t
            # Arc so vertical velocity varies; keeps the trajectory from
            # being a perfectly straight line the refiner cannot score.
            y -= math.sin(t * math.pi) * 18.0
            return x, y
    return points[-1][1], points[-1][2]


def render_frame(
    index: int,
    total: int,
    width: int,
    height: int,
    script: Script,
    players: list[Player],
) -> np.ndarray:
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :] = PITCH_GREEN

    stride = max(width // 8, 1)
    for x in range(0, width, stride):
        img[:, x : x + 3] = PITCH_LINE

    # Penalty box, so the scene reads as a final third.
    box_x = int(width * 0.78)
    img[int(height * 0.20) : int(height * 0.80), box_x : box_x + 3] = PITCH_LINE

    pw, ph = max(width // 90, 5), max(height // 10, 12)
    for player in players:
        # Players drift slightly so player-motion evidence is non-zero.
        drift = math.sin((index / max(total, 1)) * math.tau + player.x) * 6.0
        cx, cy = int(player.x + drift), int(player.y)
        img[
            max(0, cy - ph) : min(height, cy + ph),
            max(0, cx - pw) : min(width, cx + pw),
        ] = player.color

    bx, by = ball_position(script, index)
    r = max(width // 150, 4)
    cv_y0, cv_y1 = max(0, int(by) - r), min(height, int(by) + r)
    cv_x0, cv_x1 = max(0, int(bx) - r), min(width, int(bx) + r)
    img[cv_y0:cv_y1, cv_x0:cv_x1] = BALL_COLOR

    return img


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tests/fixtures/sample_match.mp4")
    parser.add_argument("--seconds", type=int, default=16)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = args.seconds * args.fps
    script, players = build_script(total, args.width, args.height)

    container = av.open(str(out), "w")
    stream = container.add_stream("libx264", rate=args.fps)
    stream.width, stream.height, stream.pix_fmt = args.width, args.height, "yuv420p"

    for i in range(total):
        frame = av.VideoFrame.from_ndarray(
            render_frame(i, total, args.width, args.height, script, players),
            format="bgr24",
        )
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    labels_path = out.with_suffix(".labels.json")
    labels_path.write_text(
        json.dumps(
            {
                "video": out.name,
                "fps": args.fps,
                "width": args.width,
                "height": args.height,
                "synthetic": True,
                "events": [
                    {
                        "frame": c.frame,
                        "type": c.action,
                        "notes": c.notes,
                        "difficulty_tags": ["synthetic"],
                    }
                    for c in script.contacts
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    size_mb = out.stat().st_size / 1_048_576
    print(f"Wrote {out} — {total} frames, {args.seconds}s @ {args.fps}fps, {size_mb:.1f} MB")
    print(f"Wrote {labels_path} — {len(script.contacts)} ground-truth events")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
