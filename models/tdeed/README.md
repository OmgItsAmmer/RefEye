# T-DEED checkpoint

This directory holds `checkpoint_best.pt` — the trained weights for
SoccerNetBall_challenge1, T-DEED's ball-action-spotting model (1st place,
2024 SoccerNet Ball Action Spotting Challenge). **Not committed to git**
(a ~50MB binary; see `.gitignore`).

## Getting it

1. Go to the checkpoint release linked from the [T-DEED repository](https://github.com/arturxe2/T-DEED#trained-models) (Google Drive folder).
2. Download the file for **`SoccerNetBall_challenge1`**. Confirm it's this
   one, not FineDiving/Tennis/FigureSkating/etc. — those are trained on
   different sports with different output classes and will silently produce
   wrong results (RefEye assumes the SoccerNetBall 12-class label set).
3. Place it here as `checkpoint_best.pt`:

   ```text
   models/tdeed/checkpoint_best.pt
   ```

4. Set `ai.action_spotter.provider: "tdeed"` in `config/default.yaml`
   (already the default). Start RefEye — the status bar reads **AI ready**
   once it loads. No further setup or code change needed.

## Verifying you have the right file

On load, RefEye performs a **strict** PyTorch state-dict load — if the
checkpoint's architecture doesn't match exactly (wrong sport, wrong T-DEED
variant), loading fails loudly with a clear error rather than running with
silently wrong weights. If you see `AI degraded` with a checkpoint-related
message, the file is present but is not the SoccerNetBall_challenge1 weights.

## What the model architecture code is

The T-DEED network class itself (not the checkpoint) is vendored into
`ai/action_spotting/tdeed/_vendor/` — see that directory's own README for
what was copied from upstream and why.

## Fully offline, no first-run network dependency

The vendored architecture constructs a RegNetY backbone via `timm`. Upstream
T-DEED builds this with `pretrained=True`, which would fetch ImageNet
starting weights from Hugging Face Hub on first run — but this checkpoint
immediately overwrites every one of those weights via `strict=True`
`load_state_dict` anyway, so the vendored copy in
`ai/action_spotting/tdeed/_vendor/model/model.py` is patched to
`pretrained=False`. No network access is required at any point, on the
first launch or any other.
