from __future__ import annotations

from typing import Any, Dict, List, Optional

from models import CardAllocation, MonthPlan, RecommendedCard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_card_by_id(card_id: str, cards: List[dict]) -> Optional[Dict[str, Any]]:
    for c in cards:
        if str(c.get("id")) == str(card_id):
            return c
    return None


def _best_category_rate(card: Dict[str, Any]) -> tuple[str, float, float]:
    """Return (category, points_per_100, point_value_inr) for highest-yield category."""
    pv = float(card.get("point_value_inr", 0) or 0)
    best_cat = "other"
    best_ppc = 0.0
    best_rate = 0.0
    for rr in card.get("reward_rates", []) or []:
        ppc = float(rr.get("points_per_100", 0) or 0)
        rate = ppc * pv / 100.0
        if rate > best_rate:
            best_rate = rate
            best_cat = rr.get("category", "other")
            best_ppc = ppc
    return best_cat, best_ppc, pv


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_monthly_plan(
    allocations,  # List[CardAllocation] — one optimised base month
    recommended_cards: List[RecommendedCard],
    cards: List[dict],
    profile: Dict[str, Any],
) -> List[MonthPlan]:
    """
    Produce a timeline-aware monthly plan with three phases:

    1. WELCOME BONUS PHASE (months 1 … welcome_months for each apply card)
       Route `welcome_spend_inr / welcome_months` of spend each month to that
       card (regardless of category optimisation). On the card's final welcome
       month, add the one-time bonus value spike.

    2. CATEGORY OPTIMISATION PHASE (remaining spend every month)
       Scale the base greedy allocations proportionally to whatever spend
       remains after welcome-bonus routing, respecting each card's monthly cap.

    3. ANNUAL FEE EVENTS (month 12, and month 24 if timeline > 12)
       Insert a negative `annual_fee` allocation for each active card and
       deduct the full annual_fee_inr from that month's total.
    """
    timeline_months = int(profile.get("timeline_months", 0) or 0)
    monthly_spend = float(profile.get("monthly_spend_inr", 0) or 0)

    # Card lookup
    card_by_id: Dict[str, Dict[str, Any]] = {str(c.get("id")): c for c in cards}

    # --- Welcome-bonus metadata for apply-cards ---
    class _WC:
        __slots__ = ("card_id", "card_name", "welcome_months", "spend_per_month",
                     "bonus_value", "cat", "ppc", "pv")

        def __init__(self, rec: RecommendedCard, card: Dict[str, Any]) -> None:
            ws = float(card.get("welcome_spend_inr", 0) or 0)
            wm = int(card.get("welcome_months", 0) or 0)
            self.card_id = rec.card_id
            self.card_name = rec.card_name
            self.welcome_months = wm
            self.spend_per_month = ws / wm if wm > 0 else 0.0
            self.bonus_value = float(rec.expected_bonus_value_inr or 0.0)
            self.cat, self.ppc, self.pv = _best_category_rate(card)

    welcome_cards: List[_WC] = []
    for rec in recommended_cards:
        if rec.action != "apply":
            continue
        card = card_by_id.get(rec.card_id)
        if card is None:
            continue
        ws = float(card.get("welcome_spend_inr", 0) or 0)
        wm = int(card.get("welcome_months", 0) or 0)
        if ws > 0 and wm > 0:
            welcome_cards.append(_WC(rec, card))

    # All card IDs in the plan (owned + newly applied for)
    base_alloc_ids = {a.card_id for a in allocations}
    apply_ids = {wc.card_id for wc in welcome_cards}
    all_plan_ids = base_alloc_ids | apply_ids

    # Pre-compute base allocation totals for proportional scaling
    base_total_amount = sum(float(a.amount_inr) for a in allocations)

    monthly_plan: List[MonthPlan] = []

    for month in range(1, timeline_months + 1):
        month_allocs: List[CardAllocation] = []
        remaining_spend = monthly_spend
        bonus_value_this_month = 0.0

        # ── Phase 1: Welcome bonus routing ──────────────────────────────────
        for wc in welcome_cards:
            if month > wc.welcome_months:
                continue
            spend = min(wc.spend_per_month, remaining_spend)
            if spend <= 0:
                continue
            remaining_spend -= spend

            pts = (spend / 100.0) * wc.ppc
            val = pts * wc.pv
            month_allocs.append(CardAllocation(
                card_id=wc.card_id,
                card_name=wc.card_name,
                category=wc.cat,
                amount_inr=round(spend, 2),
                expected_pts=round(pts, 2),
                expected_value_inr=round(val, 2),
            ))

            # One-time bonus spike on the last welcome month
            if month == wc.welcome_months:
                bonus_value_this_month += wc.bonus_value

        # ── Phase 2: Scaled category-optimised allocations ──────────────────
        if allocations and remaining_spend > 0 and base_total_amount > 0:
            scale = remaining_spend / base_total_amount
            for a in allocations:
                scaled_amount = float(a.amount_inr) * scale
                if scaled_amount < 1.0:
                    continue
                month_allocs.append(CardAllocation(
                    card_id=a.card_id,
                    card_name=a.card_name,
                    category=a.category,
                    amount_inr=round(scaled_amount, 2),
                    expected_pts=round(float(a.expected_pts) * scale, 2),
                    expected_value_inr=round(float(a.expected_value_inr) * scale, 2),
                ))

        # ── Phase 3: Annual fee events at month 12 / 24 ─────────────────────
        annual_fee_deduction = 0.0
        if month in (12, 24):
            for cid in all_plan_ids:
                card = card_by_id.get(cid)
                if card is None:
                    continue
                fee = float(card.get("annual_fee_inr", 0) or 0)
                if fee <= 0:
                    continue
                annual_fee_deduction += fee
                month_allocs.append(CardAllocation(
                    card_id=cid,
                    card_name=str(card.get("name", "")),
                    category="annual_fee",
                    amount_inr=0.0,
                    expected_pts=0.0,
                    expected_value_inr=round(-fee, 2),
                ))

        # Total for the month: reward allocations + welcome bonus - annual fees
        reward_value = sum(
            float(a.expected_value_inr)
            for a in month_allocs
            if a.category != "annual_fee"
        )
        total_value = reward_value + bonus_value_this_month - annual_fee_deduction

        monthly_plan.append(MonthPlan(
            month=month,
            allocations=month_allocs,
            total_expected_value_inr=round(total_value, 2),
        ))

    return monthly_plan
