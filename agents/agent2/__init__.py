from __future__ import annotations

import sys
from typing import Any, Dict, List

from .graph import beam_search, graph_to_monthly_plan, monte_carlo_search
from models import StrategyPlan, UserProfile, MCResult

# Adjustable beam width — increase for higher quality, decrease for speed.
BEAM_WIDTH = 20


def run_agent2(profile: dict, cards: list[dict]) -> StrategyPlan:
    # Never mutate input profile/cards
    if isinstance(profile, UserProfile):
        profile_dict: Dict[str, Any] = profile.model_dump()
    else:
        profile_dict = dict(profile)

    cards_list: List[dict] = list(cards)

    # Run beam search over the time-expanded graph
    path, totals = beam_search(cards_list, profile_dict, beam_width=BEAM_WIDTH)

    # Convert path into MonthPlan / RecommendedCard Pydantic objects
    monthly_plan, recommended = graph_to_monthly_plan(path, cards_list, profile_dict)

    total_rewards = totals["total_rewards_inr"]
    total_fees = totals["total_fees"]
    net_value = totals["net_value_inr"]

    if net_value < 0:
        print(
            f"WARNING: net_value_inr is negative (₹{net_value:,.2f}) — fees exceed rewards",
            file=sys.stderr,
        )

    mc_result_dict = monte_carlo_search(cards_list, profile_dict, n_simulations=50, cv=0.20)
    mc_result = MCResult(**mc_result_dict)

    return StrategyPlan(
        session_id=str(profile_dict.get("session_id", "")),
        monthly_plan=monthly_plan,
        recommended_cards=recommended,
        total_rewards_inr=round(float(total_rewards), 2),
        total_fees_inr=round(float(total_fees), 2),
        net_value_inr=round(float(net_value), 2),
        mc_result=mc_result,
    )
