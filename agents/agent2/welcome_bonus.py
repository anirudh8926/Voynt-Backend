from __future__ import annotations

from typing import Any, Dict, List, Set

from models import RecommendedCard


def credit_floor(profile: Dict[str, Any]) -> float:
    credit_range = str(profile.get("credit_score_range", "") or "")
    credit_floor_map = {
        "600-650":  0.40,
        "650-700":  0.55,
        "700-750":  0.65,
        "750+":     0.75,
        "750-800":  0.75,
        "800+":     0.85,
    }
    return float(credit_floor_map.get(credit_range, 0.60))


def evaluate_welcome_bonuses(
    cards: List[dict],
    profile: Dict[str, Any],
    allocated_card_ids: Set[str],
) -> List[RecommendedCard]:
    """
    Evaluate welcome bonus feasibility and recommend apply/activate/retire actions.
    Returns list sorted by expected_bonus_value_inr descending.
    """
    owned = set(profile.get("cards_owned", []) or [])
    floor = credit_floor(profile)

    recs: List[RecommendedCard] = []

    # FOR EACH card NOT in profile["cards_owned"]
    for card in cards:
        cid = str(card.get("id"))
        if cid in owned:
            continue

        bonus_pts = float(card.get("welcome_bonus_pts", 0.0) or 0.0)
        point_value = float(card.get("point_value_inr", 0.0) or 0.0)
        bonus_value_inr = bonus_pts * point_value

        partners = card.get("transfer_partners", None) or []
        if partners:
            best_cpp = max(float(p.get("cpp_inr", 0.0) or 0.0) for p in partners)
            best_ratio = min(
                float(p.get("transfer_ratio", 1.0) or 1.0)
                for p in partners
                if float(p.get("cpp_inr", 0.0) or 0.0) == best_cpp
            )
            if best_ratio > 0:
                transfer_bonus_value = (bonus_pts / best_ratio) * best_cpp
                bonus_value_inr = max(bonus_value_inr, transfer_bonus_value)

        welcome_spend = float(card.get("welcome_spend_inr", 0.0) or 0.0)
        welcome_months = int(card.get("welcome_months", 0) or 0)
        if welcome_months <= 0:
            monthly_spend_needed = float("inf")
        else:
            monthly_spend_needed = welcome_spend / welcome_months

        spend_feasible = monthly_spend_needed <= float(
            profile.get("monthly_spend_inr", 0.0) or 0.0
        )
        time_feasible = welcome_months <= int(profile.get("timeline_months", 0) or 0)
        worth_the_fee = bonus_value_inr > float(card.get("annual_fee_inr", 0.0) or 0.0)
        approval_ok = float(card.get("approval_prob", 0.0) or 0.0) >= floor

        if spend_feasible and time_feasible and worth_the_fee and approval_ok:
            recs.append(
                RecommendedCard(
                    card_id=cid,
                    card_name=str(card.get("name", "")),
                    action="apply",
                    reason=(
                        f"Welcome bonus worth ₹{bonus_value_inr:,.0f}, "
                        f"fee ₹{float(card.get('annual_fee_inr', 0.0) or 0.0):,.0f}, "
                        f"need ₹{monthly_spend_needed:,.0f}/month for "
                        f"{welcome_months} months"
                    ),
                    approval_prob=float(card.get("approval_prob", 0.0) or 0.0),
                    expected_bonus_value_inr=round(float(bonus_value_inr), 2),
                )
            )

    # FOR EACH card IN profile["cards_owned"]
    for card in cards:
        cid = str(card.get("id"))
        if cid not in owned:
            continue

        if cid in allocated_card_ids:
            continue

        best_rate = 0.0
        for r in card.get("reward_rates", []) or []:
            points_per_100 = float(r.get("points_per_100", 0.0) or 0.0)
            best_rate = max(
                best_rate,
                (points_per_100 * float(card.get("point_value_inr", 0.0) or 0.0)) / 100.0,
            )

        if best_rate > 0.02:
            recs.append(
                RecommendedCard(
                    card_id=cid,
                    card_name=str(card.get("name", "")),
                    action="use",
                    reason="You own this card but it's not in your current plan — route spend here",
                    approval_prob=float(card.get("approval_prob", 0.0) or 0.0),
                    expected_bonus_value_inr=0.0,
                )
            )
        else:
            recs.append(
                RecommendedCard(
                    card_id=cid,
                    card_name=str(card.get("name", "")),
                    action="retire",
                    reason="Low reward rate — not worth carrying",
                    approval_prob=float(card.get("approval_prob", 0.0) or 0.0),
                    expected_bonus_value_inr=0.0,
                )
            )

    recs.sort(key=lambda r: float(r.expected_bonus_value_inr or 0.0), reverse=True)
    return recs

