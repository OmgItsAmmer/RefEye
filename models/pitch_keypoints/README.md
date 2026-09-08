# Pitch keypoint checkpoint (automatic METRIC calibration)

This directory holds `soccana_keypoint.pt` — a YOLO11-pose model fine-tuned
to find 29 named pitch landmarks (penalty area corners, goal area corners,
pitch corners, centre circle) directly from a broadcast frame. **Not
committed to git** (a binary under `models/**/*.pt`; see `.gitignore`).

## Getting it

```bash
python -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='Adit-jain/Soccana_Keypoint', filename='Model/weights/best.pt')
shutil.copy(path, 'models/pitch_keypoints/soccana_keypoint.pt')
"
```

(~86 MB. Trained on SoccerNet's official camera-calibration dataset —
confirmed from the checkpoint's own training metadata. Public on Hugging
Face: [Adit-jain/Soccana_Keypoint](https://huggingface.co/Adit-jain/Soccana_Keypoint).)

## Why this exists

Every earlier route to METRIC calibration (real metres, not just direction)
needed an operator to click four named landmarks by hand — and that clicking
only ever existed in the internal debug Inspector, never in the shipped app.
This model finds the same landmarks by itself, so METRIC calibration can
happen without anyone at the mouse. Operator marks (where that flow exists)
still outrank this — a person confirming a point is stronger evidence than
any model — this only runs when there are none.

## Validated against real footage before being turned on by default

Three different broadcast clips, at two different resolutions and from two
different leagues/broadcasters, each independently:

| Clip | Confident points found | Fit confidence | Mean error |
|---|---|---|---|
| `demo_video_offside_1.mp4` (1920x1080) | 8 | 0.95 | 0.09m |
| `client_m2_test_video.mp4` (1280x720) | 8 | 0.95 | 0.11m |
| `demo_video_offside_3.mp4` (1280x720) | 7 | 0.90 | 0.19m |

Checked visually too, not just numerically: the calibrated homography was
used to project the pitch model's own line drawing back onto each source
frame, and the projected penalty box / six-yard box / goal lines land within
a few pixels of the real painted ones in every case (see the conversation
history / `offside/pitch_calibration/auto_landmarks.py` for the method).

One honest failure in the same batch: a close-up shot with no box lines
visible at all correctly found 0 usable points and fell through to the
existing directional-only fallback — the point of the confidence-and-count
gating (`PitchCalibrationConfig.auto_landmarks.min_points`), not a bug.

## What it doesn't cover

Only 21 of the model's 29 keypoints are used — the straight-corner points
(pitch corners, penalty area corners, goal area corners) that map directly
onto this project's existing `PitchModel.landmarks()`. The 8 centre-circle
and penalty-arc points are left unmapped; using them needs new landmark
geometry added to `PitchModel` and its own validation, not done here. In
practice this rarely matters — a broadcast shot showing a penalty area
already offers up to 8 of the 21 usable points on its own.

## If it is missing

`auto_landmarks: enabled: false` in config, or simply not having the
checkpoint present, degrades cleanly: the pipeline falls back to the
directional-only automatic calibration that existed before this file did,
and logs why (`auto_landmarks_unavailable`) rather than crashing the offside
path — the same "degrade, don't crash" policy every other model in this
project follows.
