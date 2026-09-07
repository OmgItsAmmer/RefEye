"""Ranking the defending side to find the player the line is drawn through (M2.5).

    rank_opponents    orders every opponent along the goal-to-goal axis
    DefenderRanking   the order, plus who was left out and why
    RankedPlayer      one opponent, placed

Law 11 counts the second-last *opponent*, not the last defender and not the
outfielders only — so the goalkeeper is ranked like anyone else and never
special-cased. See `ranking.py` for why that makes M2.3's goalkeeper
identification affect the explanation rather than the verdict.
"""

from offside.second_last_defender.ranking import (
    DefenderRanking,
    RankedPlayer,
    image_direction_toward_goal,
    rank_opponents,
)

__all__ = [
    "DefenderRanking",
    "RankedPlayer",
    "image_direction_toward_goal",
    "rank_opponents",
]
