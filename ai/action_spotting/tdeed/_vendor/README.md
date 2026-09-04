# Vendored T-DEED model code

Source: https://github.com/arturxe2/T-DEED (`model/` directory only)

This is the model **architecture** code only — the network class needed to
load the trained weights. Training scripts, dataset loaders, and evaluation
harnesses from the original repo are intentionally not included; they are not
needed for inference and would pull in extra dependencies (the `SoccerNet`
evaluation package, `wandb`, etc.) for no runtime benefit.

## Why vendored rather than a git submodule / pip dependency

T-DEED is not published as an installable package, and its repository mixes
training/eval/data-prep code with the model definition. Vendoring just the
`model/` package:

- keeps RefEye's dependency footprint to what inference actually needs
- gives the client a single `pip install -r requirements.txt` with no
  "also clone this other repo" step
- pins the exact architecture version this checkpoint was trained against —
  the upstream repo evolving would otherwise risk a shape mismatch with the
  checkpoint in `models/tdeed/`

## Changes made from upstream

1. Absolute imports (`from model.modules import ...`) changed to relative
   (`from .modules import ...`) so the package works at its new location.
2. Added `__init__.py` files (upstream has none; it runs as a script directory).
3. **Not modified in-place**: `TDEEDModel.Impl.update_pred_head()` hardcodes
   `.cuda()`, which crashes on a CPU-only machine. Rather than patch vendored
   code (which would silently diverge from upstream and confuse future
   updates), `adapter.py` reimplements the two-line equivalent of that method
   without the hardcoded device — see the comment at its call site.
4. `model.py`'s backbone constructor: `timm.create_model(..., pretrained=True)`
   changed to `pretrained=False`. Unlike item 3 this genuinely needed an
   in-place patch — it's a constructor argument, not something `adapter.py`
   can override afterward. `pretrained=True` fetches ImageNet starting
   weights from Hugging Face Hub on first run, but `adapter.py`'s
   `strict=True load_state_dict` immediately overwrites every one of those
   weights with the checkpoint anyway, so the fetch is pure waste and a
   needless network dependency for a "no extra steps" packaged install
   (README.md's "Fully offline" section).

## License

T-DEED is released under the license in `checkpoints/../LICENSE` and
`LICENSE E2E-Spot` in the original repository. Verify redistribution terms
for both the code and the checkpoint before shipping to a client
(architecture.md section 55) — this is flagged, not resolved, by vendoring.

## Updating

If the checkpoint changes to one trained against a different T-DEED version,
copy the new `model/` directory from the upstream repo over this one and
reapply items 1 and 4 above (item 2 is just new empty `__init__.py` files;
item 3 lives in `adapter.py`, not here, so nothing to reapply for it).
