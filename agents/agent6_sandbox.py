from typing import Dict, List, Optional

from models import SandboxRequest, SandboxResponse, StrategyPlan


def _get_rate(card: dict, category: str) -> float:
    """Return points_per_100 for the given category from a card's reward_rates list.
    Falls back to the 'other' rate, then 0.0 if neither exists."""
    rates = card.get("reward_rates") or []
    other_pts = 0.0
    for r in rates:
        cat = (r.get("category") or "").lower()
        pts = float(r.get("points_per_100", 0.0) or 0.0)
        if cat == category.lower():
            return pts
        if cat == "other":
            other_pts = pts
    return other_pts


def run_agent6(
    request: SandboxRequest,
    cards: List[dict],
    ai_strategy: Optional[StrategyPlan],
) -> SandboxResponse:
    """
    Lightweight reward recalculation for sandbox mode.

    For each (card_id, category, amount) in spend_overrides:
      - Look up the card in the cards list by id.
      - Fetch points_per_100 for that category from reward_rates.
      - Fetch point_value_inr from the card row.
      - Compute reward = (amount / 100) * points_per_100 * point_value_inr.

    Sums all rewards to produce computed_rewards_inr, then:
      yield_index       = computed_rewards_inr / total_spend
      diff_vs_ai_inr    = computed_rewards_inr - ai_strategy.total_rewards_inr
    """
    # Index cards by id for O(1) lookup
    card_index: Dict[str, dict] = {str(c["id"]): c for c in (cards or [])}

    total_spend = 0.0
    computed_rewards_inr = 0.0

    for card_id, cat_map in request.spend_overrides.items():
        card = card_index.get(str(card_id))
        if card is None:
            # Unknown card – count spend but earn no rewards
            for amount in cat_map.values():
                total_spend += float(amount)
            continue

        point_value_inr = float(card.get("point_value_inr", 0.0) or 0.0)

        for category, amount in cat_map.items():
            amount = float(amount)
            total_spend += amount
            if amount <= 0:
                continue

            points_per_100 = _get_rate(card, category)
            computed_rewards_inr += (amount / 100.0) * points_per_100 * point_value_inr

    yield_index = computed_rewards_inr / total_spend if total_spend > 0 else 0.0

    ai_total = ai_strategy.total_rewards_inr if ai_strategy is not None else 0.0
    diff_vs_ai_inr = computed_rewards_inr - ai_total

    return SandboxResponse(
        computed_rewards_inr=round(computed_rewards_inr, 2),
        yield_index=round(yield_index, 6),
        diff_vs_ai_inr=round(diff_vs_ai_inr, 2),
        simulation_result=None,
    )
