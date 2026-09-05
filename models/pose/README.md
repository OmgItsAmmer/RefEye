# Pose checkpoint (offside body points, M2.2)

This directory holds `yolo11n-pose.pt` — the COCO-17 pose model that finds
each player's body points (crucially, their feet) so offside can be measured
from a foot rather than from the middle of a bounding box. **Not committed to
git** (a binary under `models/**/*.pt`; see `.gitignore`).

## Getting it

```bash
curl -L -o models/pose/yolo11n-pose.pt \
  https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
```

(~6 MB. Ultralytics publishes it under AGPL-3.0 — the same licence and source
as `models/detector/yolo11n.pt`, which RefEye already ships against.)

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

## Why the nano model, and why it runs on crops

Measured on the client reference clip (`data/videos/client_m2_test_video.mp4`,
1280x720 broadcast, players ~90-100px tall):

| approach | players found |
|---|---|
| whole frame, `imgsz` 1280 | 1-2 of ~17 |
| per-player crops, `imgsz` 256 | 11-13 of ~17, most with a confident ankle |

Wide broadcast framing leaves too few pixels per player for whole-frame pose
estimation; cropping each detected player and upscaling restores them. Because
each crop is small, the nano model is enough — and a bigger model would not
fix the underlying pixel shortage. If accuracy on distant players proves
insufficient during M2.8 tuning, the upgrade path is `yolo11s-pose.pt` /
`yolo11m-pose.pt` (a checkpoint swap in config, no code change) or a
crop-native pose model such as RTMPose behind the same `PoseEstimator`
protocol.
