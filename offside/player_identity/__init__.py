"""Player identity through the contact moment (M2.4).

Public surface:

    IdentityTracker   the stage — follows players, survives occlusion, dies at cuts
    IdentityResult    its output, with per-player state and confidence
    PlayerIdentity    one player's identity and the evidence for it
    IdentityState     CONFIRMED / TENTATIVE / RECOVERED / CONTESTED

Read `tracker.py`'s docstring for why this is not simply M1's `ByteTracker`,
and for the argument about what kit-colour appearance can and cannot do —
in short, it cannot separate teammates, and the swaps that break an offside
call are exactly the ones it can separate.

`IdentityTracker.update` sets `track_id` on each `PlayerPose`; that is the
only place any stage should be assigning identity, which is the rule M2.2
established when it refused to let the pose model create or rename players.
"""

from offside.player_identity.identity import (
    TRUSTED_STATES,
    IdentityResult,
    IdentityState,
    PlayerIdentity,
)
from offside.player_identity.tracker import IdentityTracker

__all__ = [
    "TRUSTED_STATES",
    "IdentityResult",
    "IdentityState",
    "IdentityTracker",
    "PlayerIdentity",
]
