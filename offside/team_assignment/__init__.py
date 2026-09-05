"""Team, goalkeeper and attacking-side assignment (M2.3).

Public surface, so M2.5 and the review UI import from here rather than
reaching into the internals:

    TeamAssigner            the stage — kit clustering, keeper, attacking side
    TeamAssignment          its output, with confidence, reasons and warnings
    PlayerTeam / PlayerRole one player's team and role
    TeamOverrides           the operator's corrections (M2_Plan section 7)
    JerseyColorExtractor    the swappable feature behind the clustering

No kit colour is configured anywhere: the two teams are discovered from the
footage on every run. Read `assigner.py`'s docstring for why goalkeeper
identification is positional rather than colour-based, and why "which side is
attacking" comes from the ball rather than from the clustering.
"""

from offside.team_assignment.assigner import TeamAssigner, TeamOverride, TeamOverrides
from offside.team_assignment.clustering import fit_team_colors
from offside.team_assignment.jersey_color import (
    JerseyColorExtractor,
    TeamFeatureExtractor,
)
from offside.team_assignment.teams import (
    SOURCE_KIT_COLOUR,
    SOURCE_OPERATOR,
    SOURCE_OUTLIER,
    SOURCE_UNMEASURED,
    TEAM_A,
    TEAM_B,
    TEAM_IDS,
    JerseyColor,
    PlayerRole,
    PlayerTeam,
    TeamAssignment,
    TeamColorModel,
)

__all__ = [
    "SOURCE_KIT_COLOUR",
    "SOURCE_OPERATOR",
    "SOURCE_OUTLIER",
    "SOURCE_UNMEASURED",
    "TEAM_A",
    "TEAM_B",
    "TEAM_IDS",
    "JerseyColor",
    "JerseyColorExtractor",
    "PlayerRole",
    "PlayerTeam",
    "TeamAssigner",
    "TeamAssignment",
    "TeamColorModel",
    "TeamFeatureExtractor",
    "TeamOverride",
    "TeamOverrides",
    "fit_team_colors",
]
