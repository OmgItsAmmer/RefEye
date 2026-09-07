# Pose checkpoint (offside body points, M2.2)

This directory holds `yolo11m-pose.pt` — the COCO-17 pose model that finds
each player's body points (crucially, their feet) so offside can be measured
from a foot rather than from the middle of a bounding box. **Not committed to
git** (a binary under `models/**/*.pt`; see `.gitignore`).

## Getting it

```bash
curl -L -o models/pose/yolo11m-pose.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m-pose.pt
```

(~42 MB. Ultralytics publishes it under AGPL-3.0 — the same licence and
source as `models/detector/yolo11m.pt`, which RefEye already ships against.
The nano variant, `yolo11n-pose.pt`, still works as a smaller/faster
fallback — see "Why medium, not nano" below for why it isn't the default
any more.)

That is the only setup step. `offside.body_keypoints.checkpoint` in
`config/default.yaml` already points here, and the app loads it at startup
alongside the other models.

## If it is missing

RefEye still starts and every Milestone 1 feature keeps working — live
preview, hotkey analysis, candidate review. What you lose is offside body
points, and the status line says so explicitly rather than quietly
substituting a guess. This is the same "degrade, don't crash" policy the
action spotter uses (architecture.md sections 38, 49).

## Why a *pose* model at all

A detector returns a rectangle. Offside is judged on where a player's foot,
knee, hip or shoulder is — and a player mid-stride can have a foot most of a
stride away from the centre of their box. Feet matter twice over: they are
also the only part of a player that touches the pitch, and pitch calibration
(M2.1) can only project points that lie on the pitch *plane*.

## Why it runs on crops at all

Measured on the client reference clip (`data/videos/client_m2_test_video.mp4`,
1280x720 broadcast, players ~90-100px tall):

| approach | players found |
|---|---|
| whole frame, `imgsz` 1280 | 1-2 of ~17 |
| per-player crops, `imgsz` 256 | 11-13 of ~17, most with a confident ankle |

Wide broadcast framing leaves too few pixels per player for whole-frame pose
estimation; cropping each detected player and upscaling restores them.

## Why medium, not nano

The original assumption here was that the crop already restores enough
resolution that a bigger model on top wouldn't help much — "each crop is
small, the nano model is enough." That assumption turned out to be wrong on
at least one real clip. Re-measured after swapping nano → medium, same
per-crop pipeline, across all six clips under `data/videos/`, tracking the
share of players who fell all the way through the ladder to a box-bottom
guess (no ankle, no knee — see `offside/body_keypoints/`):

| clip | box-bottom guess, nano | box-bottom guess, medium |
|---|---|---|
| `client_m2_test_video.mp4` | 33% | **13%** |
| `demo_video_offside_3.mp4` | 32% | 25% |
| `demo_video_offside_4.mp4` | 27% | 18% |
| `demo_video_offside_2.mp4` | 22% | 19% |
| `demo_video_offside_7.mp4` | 21% | 19% |
| `demo_video_offside_1.mp4` | 8% | 9% |

A real, measured drop on every clip that had room to improve (the one clip
already at 8% — the highest native resolution of the six, 1920x1080 — stayed
flat, which is consistent with the fix being about the model rather than
noise). This matters because a box-bottom guess is penalised hard (×0.3
confidence) and widens M2.5's error bar directly — fewer of them means more
verdicts clear the confidence floor instead of coming back "too close to
call" for a reason that was actually "the pose model, not the play."

Since this path (M2.2) is triggered once per confirmed frame, not run
continuously, the extra inference cost of `yolo11m-pose.pt` over
`yolo11n-pose.pt` is not something an operator would notice. If a stronger
model is still wanted later, RTMPose or ViTPose are documented to hold up
better specifically in low-resolution/distant-subject conditions than any
YOLO-pose size — but neither is a drop-in checkpoint swap like this one was;
both need a different runtime integration behind the same `PoseEstimator`
protocol.
