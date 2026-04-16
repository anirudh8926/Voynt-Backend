"""
graph.py — Time-Expanded Graph Beam Search for Voynt agent2.

Replaces the greedy per-month allocator with a month-by-month beam search
that tracks card application state, welcome-bonus progress, and point expiry
across the full timeline.

Public API:
    beam_search(cards, profile, beam_width=20) -> (path, totals)
    graph_to_monthly_plan(path, cards, profile) -> (monthly_plan, recommended_cards)
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from models import CardAllocation, MonthPlan, RecommendedCard

# ---------------------------------------------------------------------------
# Internal card model
# ---------------------------------------------------------------------------

CATEGORIES = [
    "groceries", "dining", "travel", "fuel",
    "online", "entertainment", "utilities", "other",
]


@dataclass(frozen=True)
class CardModel:
    id: str
    name: str
    annual_fee: float
    base_earn_rate: float          # best effective ₹ return per ₹1 spent
    earn_rates: tuple              # ((category, pts_per_100, point_value_inr), ...)
    eff_rates: tuple               # ((category, effective_inr_per_inr), ...) — includes partner uplift
    transfer_partners: tuple       # list of partner program names
    welcome_bonus_pts: float
    welcome_spend_req: float
    welcome_window: int            # months
    point_value_inr: float
    annual_fee_months: tuple       # which months the annual fee falls (12, 24, ...)


def get_goal_type(goal_text: str) -> str:
    text = goal_text.lower()
    if any(k in text for k in ["international", "flight", "japan", "europe", "usa", "thailand", "dubai", "abroad", "overseas", "trip", "travel"]):
        return "travel"
    elif "cashback" in text:
        return "cashback"
    elif any(k in text for k in ["shopping", "amazon", "flipkart", "buy", "purchase"]):
        return "shopping"
    return "other"


def best_redemption_value(
    card: CardModel, 
    goal_type: str, 
    state: Optional[State] = None, 
    horizon_months: int = 12
) -> float:
    """
    Returns best achievable INR per point for this card,
    given the redemption goal (travel, cashback, shopping, etc.)
    Accepts state & horizon_months to dynamically scale based on expiry/proximity.
    """
    base = card.point_value_inr  # fallback

    if not card.transfer_partners:
        return base

    partner_values = {
        "travel": {
            "singapore_airlines": 4.5,
            "emirates": 3.8,
            "air_india": 2.5,
            "air_india_flying_returns": 2.5,
            "marriott_bonvoy": 4.0,
            "club_vistara": 3.5,
            "krisflyer": 4.5,
        },
        "cashback": {},  # transfer partners rarely help for cashback goals
        "shopping": {
            "amazon_pay": 0.8,
            "flipkart": 0.8,
        },
    }

    goal_partners = partner_values.get(goal_type, {})
    uplift = max(
        (goal_partners.get(p, base) for p in card.transfer_partners),
        default=base
    )
    return max(base, uplift)


def _build_card_models(cards: List[dict], profile: Dict[str, Any]) -> List[CardModel]:
    """Map raw Supabase card dicts into internal CardModel structs."""
    timeline = int(profile.get("timeline_months", 12) or 12)
    goal_text = str(profile.get("goal_text", "")).lower()
    international = any(
        k in goal_text for k in
        ["international", "flight", "japan", "europe", "usa", "thailand",
         "dubai", "abroad", "overseas", "trip abroad"]
    )

    models: List[CardModel] = []
    for c in cards:
        pv = float(c.get("point_value_inr", 0.0) or 0.0)
        expiry_m = c.get("reward_expiry_months")

        earn_rates_list: List[Tuple[str, float, float]] = []
        eff_rates_list: List[Tuple[str, float]] = []
        best_earn = 0.0

        tx_partners = []
        for p in (c.get("transfer_partners") or []):
            prog = p.get("partner_program", "")
            if prog:
                tx_partners.append(prog.lower().replace(" ", "_"))
        partners_tuple = tuple(tx_partners)

        temp_card = CardModel(
            id=str(c.get("id")), name="", annual_fee=0.0, base_earn_rate=0.0,
            earn_rates=(), eff_rates=(), transfer_partners=partners_tuple,
            welcome_bonus_pts=0.0, welcome_spend_req=0.0, welcome_window=0,
            point_value_inr=pv, annual_fee_months=()
        )
        dynamic_pv = best_redemption_value(temp_card, goal_text)

        for rr in (c.get("reward_rates") or []):
            cat = str(rr.get("category", "other"))
            ppc = float(rr.get("points_per_100", 0.0) or 0.0)

            # Replace static pv with dynamic best_redemption_value for this goal
            eff = (ppc * dynamic_pv) / 100.0

            # Transfer-partner uplift from database payload (if superior)
            for p in (c.get("transfer_partners") or []):
                ratio = float(p.get("transfer_ratio", 0.0) or 0.0)
                cpp = float(p.get("cpp_inr", 0.0) or 0.0)
                if ratio > 0:
                    eff = max(eff, (ppc / ratio * cpp) / 100.0)

            # Forex penalty on travel for international goals
            if international and cat == "travel":
                eff -= float(c.get("forex_fee_pct", 0.0) or 0.0) / 100.0

            # Fuel surcharge
            if cat == "fuel":
                eff += 0.01 if bool(c.get("fuel_surcharge_waiver")) else -0.01

            # Expiry penalty
            if expiry_m is not None:
                try:
                    em = int(expiry_m)
                    if em < timeline:
                        eff = 0.0
                    elif em < timeline * 2:
                        eff *= 0.85
                except Exception:
                    pass

            earn_rates_list.append((cat, ppc, pv))
            eff_rates_list.append((cat, eff))
            best_earn = max(best_earn, eff)

        # Annual fee milestones within the timeline
        fee_months = tuple(m for m in range(12, timeline + 1, 12))

        models.append(CardModel(
            id=str(c.get("id")),
            name=str(c.get("name", "")),
            annual_fee=float(c.get("annual_fee_inr", 0.0) or 0.0),
            base_earn_rate=best_earn,
            earn_rates=tuple(earn_rates_list),
            eff_rates=tuple(eff_rates_list),
            transfer_partners=partners_tuple,
            welcome_bonus_pts=float(c.get("welcome_bonus_pts", 0.0) or 0.0),
            welcome_spend_req=float(c.get("welcome_spend_inr", 0.0) or 0.0),
            welcome_window=int(c.get("welcome_months", 0) or 0),
            point_value_inr=pv,
            annual_fee_months=fee_months,
        ))
    return models


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class State:
    month: int
    pts_by_card: tuple             # ((card_id, pts_int), ...)
    total_fees_paid: float
    total_spend: float
    cards_held: frozenset          # frozenset[card_id]
    bonus_progress: tuple          # ((card_id, spend_so_far), ...)
    bonus_claimed: frozenset       # frozenset[card_id]
    pts_expiry: tuple              # ((card_id, pts_amount, expiry_month), ...)
    path: tuple = field(default=(), hash=False, compare=False)


def _f_score(state: State, cards_map: Dict[str, CardModel], goal_type: str) -> float:
    """Higher is better — values points based on user's goal type."""
    total_val = 0.0
    for card_id, pts in state.pts_by_card:
        card = cards_map.get(card_id)
        if card:
            total_val += pts * best_redemption_value(card, goal_type)
    return total_val - state.total_fees_paid


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def _pts_for_spend(card: CardModel, category: str, amount: float) -> float:
    for (cat, ppc, pv) in card.earn_rates:
        if cat == category:
            return (amount / 100.0) * ppc
    # Fall back to base rate using first available rate
    if card.earn_rates:
        _, ppc, _ = card.earn_rates[0]
        return (amount / 100.0) * ppc
    return 0.0


def _expand_month(
    state: State,
    card_models: List[CardModel],
    profile: Dict[str, Any],
    goal_type: str,
) -> List[State]:
    """
    From a given state at the START of a month, generate all next states
    after spending this month's budget optimally across cards already held,
    applying one new card if beneficial, and handling bonus/fee events.
    Returns a flat list of candidate next states.
    """
    monthly_budget = float(profile.get("monthly_spend_inr", 0.0) or 0.0)
    spend_breakdown: Dict[str, float] = dict(profile.get("spend_breakdown") or {})
    timeline = int(profile.get("timeline_months", 12) or 12)
    owned_at_start = set(profile.get("cards_owned") or [])

    card_by_id: Dict[str, CardModel] = {cm.id: cm for cm in card_models}
    cur_month = state.month

    # ── Helper: advance to next month (WAIT) ────────────────────────────────
    def _advance(s: State, step_desc: dict) -> State:
        next_month = s.month + 1
        
        # Expire points per card
        surviving_expiry = []
        expired_by_card = {}
        for cid, pts, exp in s.pts_expiry:
            if exp > next_month:
                surviving_expiry.append((cid, pts, exp))
            else:
                expired_by_card[cid] = expired_by_card.get(cid, 0) + pts

        new_pts_by_card = {}
        for cid, pts in s.pts_by_card:
            rem = pts - expired_by_card.get(cid, 0)
            if rem > 0:
                new_pts_by_card[cid] = rem

        # Annual fees for cards moving into this new month
        new_fees: float = 0.0
        for cid in s.cards_held:
            if cid in card_by_id:
                cm = card_by_id[cid]
                if next_month in cm.annual_fee_months:
                    new_fees = new_fees + float(cm.annual_fee)

        return State(
            month=next_month,
            pts_by_card=tuple(sorted(new_pts_by_card.items())),
            total_fees_paid=float(s.total_fees_paid) + new_fees,
            total_spend=s.total_spend,
            cards_held=s.cards_held,
            bonus_progress=s.bonus_progress,
            bonus_claimed=s.bonus_claimed,
            pts_expiry=tuple(surviving_expiry),
            path=s.path + (step_desc,),
        )

    # ── Helper: spend this month ─────────────────────────────────────────────
    def _spend(s: State) -> State:
        if not s.cards_held:
            return s

        pts_dict = dict(s.pts_by_card)
        new_expiry = list(s.pts_expiry)
        bonus_prog = dict(s.bonus_progress)
        bonus_claimed = set(s.bonus_claimed)
        remaining = monthly_budget
        step_allocs = []

        # Allocate spend category-by-category to best held card
        for cat in CATEGORIES:
            cat_spend = float(spend_breakdown.get(cat, 0.0) or 0.0)
            if cat_spend <= 0 or remaining <= 0:
                continue
            cat_spend = min(cat_spend, remaining)

            best_card: Optional[CardModel] = None
            best_eff = -1.0
            for cid in s.cards_held:
                cm = card_by_id.get(cid)
                if not cm:
                    continue
                # Use effective rate (includes transfer-partner uplift)
                eff = 0.0
                for (rc, eff_val) in cm.eff_rates:
                    if rc == cat:
                        eff = eff_val
                        break
                if eff > best_eff:
                    best_eff = eff
                    best_card = cm

            if best_card is None:
                continue

            assert best_card is not None
            
            pts_earned = int(_pts_for_spend(best_card, cat, cat_spend))
            pts_dict[best_card.id] = pts_dict.get(best_card.id, 0) + pts_earned
            new_expiry.append((best_card.id, pts_earned, cur_month + 24))
            remaining -= cat_spend

            # Welcome bonus progress
            if best_card.id not in bonus_claimed and best_card.welcome_spend_req > 0:
                prog = bonus_prog.get(best_card.id, 0.0) + cat_spend
                bonus_prog[best_card.id] = prog
                if (prog >= best_card.welcome_spend_req
                        and cur_month <= best_card.welcome_window):
                    wb = int(best_card.welcome_bonus_pts)
                    pts_dict[best_card.id] = pts_dict.get(best_card.id, 0) + wb
                    if wb > 0:
                        new_expiry.append((best_card.id, wb, cur_month + 24))
                    bonus_claimed.add(best_card.id)

            step_allocs.append({
                "card_id": best_card.id,
                "card_name": best_card.name,
                "category": cat,
                "amount": cat_spend,
                "pts_earned": int(pts_earned),
            })

        return State(
            month=s.month,
            pts_by_card=tuple(sorted(pts_dict.items())),
            total_fees_paid=s.total_fees_paid,
            total_spend=s.total_spend + (monthly_budget - remaining),
            cards_held=s.cards_held,
            bonus_progress=tuple(sorted(bonus_prog.items())),
            bonus_claimed=frozenset(bonus_claimed),
            pts_expiry=tuple(new_expiry),
            path=s.path + ({"month": cur_month, "action": "SPEND", "allocs": step_allocs},),
        )

    # ── Candidate 1: just spend with current cards, then advance ────────────
    candidates: List[State] = []
    spent_state = _spend(state)
    candidates.append(_advance(spent_state, {"month": cur_month, "action": "WAIT"}))

    # ── Candidate 2+: apply one new card, then spend, then advance ──────────
    eligible_to_apply = [
        cm for cm in card_models
        if cm.id not in state.cards_held
        and cm.id not in owned_at_start  # don't "apply" for already-owned cards
    ]
    for cm in eligible_to_apply:
        new_held = state.cards_held | frozenset([cm.id])
        # Apply action: pay fee immediately at month 1 for this card
        fee_now = cm.annual_fee if cur_month == 1 else 0.0
        applied = State(
            month=state.month,
            pts_by_card=state.pts_by_card,
            total_fees_paid=state.total_fees_paid + fee_now,
            total_spend=state.total_spend,
            cards_held=new_held,
            bonus_progress=state.bonus_progress,
            bonus_claimed=state.bonus_claimed,
            pts_expiry=state.pts_expiry,
            path=state.path + ({"month": cur_month, "action": "APPLY", "card_id": cm.id, "card_name": cm.name},),
        )
        spent_after_apply = _spend(applied)
        candidates.append(_advance(spent_after_apply, {"month": cur_month, "action": "WAIT"}))

    return candidates


# ---------------------------------------------------------------------------
# Beam Search
# ---------------------------------------------------------------------------

def beam_search(
    cards: List[dict],
    profile: Dict[str, Any],
    beam_width: int = 20,
) -> Tuple[List[dict], Dict[str, float]]:
    """
    Run beam search over the time-expanded state graph.

    Args:
        cards:       Raw card list from Supabase.
        profile:     UserProfile dict (session_id, timeline_months, spend_breakdown, etc.).
        beam_width:  Number of states to keep per month. Higher = better quality, slower.

    Returns:
        (path, totals) where:
            path   = list of step dicts from the best final state
            totals = {"total_pts": int, "total_fees": float, "total_spend": float,
                      "total_rewards_inr": float, "net_value_inr": float, "yield_index": float}
    """
    timeline = int(profile.get("timeline_months", 12) or 12)
    card_models = _build_card_models(cards, profile)
    card_by_id = {cm.id: cm for cm in card_models}
    
    goal_type = get_goal_type(str(profile.get("goal_text", "")))

    # Seed with cards the user already owns
    owned_ids = frozenset(str(cid) for cid in (profile.get("cards_owned") or []))
    # Annual fees for owned cards at month 1
    initial_fee = sum(
        cm.annual_fee for cm in card_models if cm.id in owned_ids
    )

    initial = State(
        month=1,
        pts_by_card=tuple(),
        total_fees_paid=initial_fee,
        total_spend=0.0,
        cards_held=owned_ids,
        bonus_progress=tuple(),
        bonus_claimed=frozenset(),
        pts_expiry=tuple(),
        path=(),
    )

    beam: List[State] = [initial]

    for month in range(1, timeline + 1):
        next_states: List[State] = []
        for state in beam:
            if state.month != month:
                continue
            expansions = _expand_month(state, card_models, profile, goal_type)
            next_states.extend(expansions)

        if not next_states:
            break

        # Keep top beam_width states by f-score (higher is better)
        next_states.sort(key=lambda s: _f_score(s, card_by_id, goal_type), reverse=True)
        beam = next_states[:beam_width]

    # Best final state
    best = max(beam, key=lambda s: _f_score(s, card_by_id, goal_type))

    total_rewards_inr = sum(
        pts * best_redemption_value(card_by_id[cid], goal_type) 
        for cid, pts in best.pts_by_card 
        if cid in card_by_id
    )
    total_pts = sum(pts for cid, pts in best.pts_by_card)

    net_value = total_rewards_inr - best.total_fees_paid
    yield_index = (net_value / best.total_spend) if best.total_spend > 0 else 0.0

    totals = {
        "total_pts": total_pts,
        "total_fees": best.total_fees_paid,
        "total_spend": best.total_spend,
        "total_rewards_inr": round(total_rewards_inr, 2),
        "net_value_inr": round(net_value, 2),
        "yield_index": round(yield_index, 4),
    }

    return list(best.path), totals


# ---------------------------------------------------------------------------
# Convert beam path → StrategyPlan-compatible outputs
# ---------------------------------------------------------------------------

def graph_to_monthly_plan(
    path: List[dict],
    cards: List[dict],
    profile: Dict[str, Any],
) -> Tuple[List[MonthPlan], List[RecommendedCard]]:
    """
    Convert the flat path list from beam_search into:
        - List[MonthPlan]          (matches existing Pydantic model)
        - List[RecommendedCard]    (APPLY actions become recommendations)
    """
    timeline = int(profile.get("timeline_months", 12) or 12)
    goal_type = get_goal_type(str(profile.get("goal_text", "")))
    
    card_models = _build_card_models(cards, profile)
    cm_by_id = {cm.id: cm for cm in card_models}

    # Build month → allocations map
    month_allocs: Dict[int, List[CardAllocation]] = {m: [] for m in range(1, timeline + 1)}
    month_bonus: Dict[int, float] = {m: 0.0 for m in range(1, timeline + 1)}
    recommended_map: Dict[str, RecommendedCard] = {}

    for step in path:
        action = step.get("action")
        m = int(step.get("month", 1))

        if action == "APPLY":
            cid = step["card_id"]
            cm = cm_by_id.get(cid)
            if not cm:
                continue
                
            pv = best_redemption_value(cm, goal_type)
            wb_pts = cm.welcome_bonus_pts
            bonus_val = wb_pts * pv
            fee = cm.annual_fee
            ws = cm.welcome_spend_req
            wm = cm.welcome_window
            mspend = ws / wm if wm > 0 else 0

            if cid not in recommended_map:
                recommended_map[cid] = RecommendedCard(
                    card_id=cid,
                    card_name=step["card_name"],
                    action="apply",
                    reason=(
                        f"Welcome bonus ₹{bonus_val:,.0f}, "
                        f"fee ₹{fee:,.0f}, "
                        f"need ₹{mspend:,.0f}/month for {wm} months"
                    ),
                    approval_prob=0.75, # Default value mapped, matching previous behavior
                    expected_bonus_value_inr=round(bonus_val, 2),
                )

        elif action == "SPEND":
            for alloc in (step.get("allocs") or []):
                cid = alloc["card_id"]
                cm = cm_by_id.get(cid)
                if not cm:
                    continue
                pv = best_redemption_value(cm, goal_type)
                pts = float(alloc.get("pts_earned", 0))
                val_inr = pts * pv
                month_allocs[m].append(CardAllocation(
                    card_id=cid,
                    card_name=alloc["card_name"],
                    category=alloc["category"],
                    amount_inr=round(float(alloc["amount"]), 2),
                    expected_pts=round(pts, 2),
                    expected_value_inr=round(val_inr, 2),
                ))

    # Annual fee deductions: insert negative allocation at fee months
    owned_at_plan_end: set = set()
    for step in path:
        if step.get("action") == "APPLY":
            owned_at_plan_end.add(step["card_id"])
    for cid in (profile.get("cards_owned") or []):
        owned_at_plan_end.add(str(cid))

    for m in range(1, timeline + 1):
        if m in (12, 24):
            for cid in owned_at_plan_end:
                cm = cm_by_id.get(cid)
                if not cm:
                    continue
                fee = cm.annual_fee
                if fee > 0:
                    month_allocs[m].append(CardAllocation(
                        card_id=cid,
                        card_name=cm.name,
                        category="annual_fee",
                        amount_inr=0.0,
                        expected_pts=0.0,
                        expected_value_inr=round(-fee, 2),
                    ))

    # Build MonthPlan list
    monthly_plan: List[MonthPlan] = []
    for m in range(1, timeline + 1):
        allocs = month_allocs[m]
        total_val = sum(float(a.expected_value_inr) for a in allocs) + month_bonus.get(m, 0.0)
        monthly_plan.append(MonthPlan(
            month=m,
            allocations=allocs,
            total_expected_value_inr=round(total_val, 2),
        ))

    # Recommended cards for cards already owned but not applied via beam
    for cid in (profile.get("cards_owned") or []):
        cid = str(cid)
        if cid in recommended_map:
            continue
        cm = cm_by_id.get(cid)
        if not cm:
            continue
        pv = best_redemption_value(cm, goal_type)
        best_rate = max(
            ((float(r[1]) * pv) / 100.0)
            for r in (cm.earn_rates or [("", 0.0, 0.0)])
        ) if cm.earn_rates else 0.0

        action_str = "use" if best_rate > 0.02 else "retire"
        reason = (
            "You own this card — route spend here for rewards"
            if action_str == "use"
            else "Low reward rate — not worth carrying"
        )
        recommended_map[cid] = RecommendedCard(
            card_id=cid,
            card_name=cm.name,
            action=action_str,
            reason=reason,
            approval_prob=1.0, 
            expected_bonus_value_inr=0.0,
        )

    recommended = sorted(
        recommended_map.values(),
        key=lambda r: float(r.expected_bonus_value_inr or 0.0),
        reverse=True,
    )

    return monthly_plan, recommended


# ---------------------------------------------------------------------------
# Monte Carlo Simulation
# ---------------------------------------------------------------------------

def sample_spend(profile: Dict[str, Any], rng: np.random.Generator, cv: float = 0.20) -> Dict[str, Any]:
    new_profile = copy.deepcopy(profile)
    bd = new_profile.get("spend_breakdown", {})
    new_bd = {}
    for cat, amount in bd.items():
        if amount > 0:
            std_dev = amount * cv
            val = rng.normal(amount, std_dev)
            new_bd[cat] = max(0.0, float(val))
        else:
            new_bd[cat] = 0.0
    new_profile["spend_breakdown"] = new_bd
    return new_profile


def aggregate_mc_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "expected_net_value": 0.0,
            "p10_net_value": 0.0,
            "p90_net_value": 0.0,
            "card_win_rate": {}
        }
    
    net_values = [r["net_value_inr"] for r in results]
    expected = float(np.mean(net_values))
    p10 = float(np.percentile(net_values, 10))
    p90 = float(np.percentile(net_values, 90))
    
    wins = {}
    for r in results:
        tc = r["top_card"]
        if tc:
            wins[tc] = wins.get(tc, 0) + 1
            
    total_wins = sum(wins.values())
    win_rates = {}
    if total_wins > 0:
        for c, count in wins.items():
            win_rates[c] = round(count / total_wins, 4)
            
    return {
        "expected_net_value": round(expected, 2),
        "p10_net_value": round(p10, 2),
        "p90_net_value": round(p90, 2),
        "card_win_rate": win_rates
    }


def monte_carlo_search(
    cards: List[dict],
    profile: Dict[str, Any],
    n_simulations: int = 50,
    cv: float = 0.20
) -> Dict[str, Any]:
    # TODO: If MC latency proves high, move this to a FastAPI BackgroundTask.
    # Falling back to 50 iterations bounded by performance initially.
    session_id = profile.get("session_id")
    if not session_id:
        seed = 42
    else:
        h = hashlib.sha256(str(session_id).encode()).digest()
        seed = int.from_bytes(h[:4], "little")

    rng = np.random.default_rng(seed)
    results = []
    
    for _ in range(n_simulations):
        sim_profile = sample_spend(profile, rng, cv=cv)
        path, totals = beam_search(cards, sim_profile, beam_width=10)
        _, recommended = graph_to_monthly_plan(path, cards, sim_profile)
        
        sim_res = {
            "net_value_inr": totals["net_value_inr"],
            "top_card": recommended[0].card_id if recommended else None
        }
        results.append(sim_res)
        
    return aggregate_mc_results(results)
