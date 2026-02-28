from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


CATEGORIES = [
    "groceries",
    "dining",
    "travel",
    "fuel",
    "online",
    "entertainment",
    "utilities",
    "other",
]


INTERNATIONAL_KEYWORDS = [
    "international",
    "trip abroad",
    "flight",
    "japan",
    "europe",
    "usa",
    "thailand",
    "dubai",
    "abroad",
    "overseas",
]

INTERNATIONAL_CATEGORIES = ["travel"]


def _get_rate_for_category(card: Dict[str, Any], category: str) -> Dict[str, Any] | None:
    for rate in card.get("reward_rates", []) or []:
        if rate.get("category") == category:
            return rate
    return None


def _is_international_goal(goal_text: str) -> bool:
    t = (goal_text or "").lower()
    return any(k in t for k in INTERNATIONAL_KEYWORDS)


def build_reward_matrix(cards: List[dict], profile: Dict[str, Any]) -> np.ndarray:
    """
    Returns ndarray shape (n_cards, n_categories).
    Each cell = effective ₹ return per ₹100 spent AFTER all adjustments.
    """
    n_cards = len(cards)
    n_categories = len(CATEGORIES)
    matrix = np.zeros((n_cards, n_categories), dtype=float)

    international = _is_international_goal(str(profile.get("goal_text", "")))
    timeline_months = int(profile.get("timeline_months", 0) or 0)

    for i, card in enumerate(cards):
        point_value_inr = float(card.get("point_value_inr", 0.0) or 0.0)
        forex_fee_pct = float(card.get("forex_fee_pct", 0.0) or 0.0)
        expiry_months = card.get("reward_expiry_months", None)

        for j, category in enumerate(CATEGORIES):
            rate = _get_rate_for_category(card, category)
            points_per_100 = float((rate or {}).get("points_per_100", 0.0) or 0.0)

            # Step A — base return
            base_return = (points_per_100 * point_value_inr) / 100.0
            effective_return = base_return

            # Step B — transfer uplift
            partners = card.get("transfer_partners", None) or []
            if partners:
                partner_returns = []
                for p in partners:
                    ratio = float(p.get("transfer_ratio", 0.0) or 0.0)
                    cpp_inr = float(p.get("cpp_inr", 0.0) or 0.0)
                    if ratio <= 0:
                        continue
                    partner_return = (points_per_100 / ratio * cpp_inr) / 100.0
                    partner_returns.append(partner_return)

                if partner_returns:
                    best_partner_return = float(max(partner_returns))
                    if best_partner_return > effective_return:
                        effective_return = best_partner_return

            # Step C — forex penalty (travel only, international goals only)
            if (
                international
                and category in INTERNATIONAL_CATEGORIES
                and forex_fee_pct != 0.0
            ):
                effective_return -= forex_fee_pct / 100.0

            # Step D — fuel surcharge adjustment
            if category == "fuel":
                if bool(card.get("fuel_surcharge_waiver", False)):
                    effective_return += 0.01
                else:
                    effective_return -= 0.01

            # Step E — expiry penalty
            if expiry_months is not None:
                try:
                    expiry_m = int(expiry_months)
                except Exception:
                    expiry_m = None

                if expiry_m is not None:
                    if expiry_m < timeline_months:
                        effective_return *= 0.0
                    elif expiry_m < timeline_months * 2:
                        effective_return *= 0.85

            matrix[i, j] = effective_return

    return matrix


def build_cap_matrix(cards: List[dict]) -> np.ndarray:
    """
    Returns ndarray shape (n_cards, n_categories).
    Each cell = monthly spend cap in ₹ for that card/category.
    np.inf where no cap exists.
    """
    n_cards = len(cards)
    n_categories = len(CATEGORIES)
    cap_matrix = np.full((n_cards, n_categories), np.inf, dtype=float)

    for i, card in enumerate(cards):
        for j, category in enumerate(CATEGORIES):
            rate = _get_rate_for_category(card, category)
            cap = (rate or {}).get("monthly_cap_inr", None)
            if cap is None:
                cap_matrix[i, j] = np.inf
            else:
                cap_matrix[i, j] = float(cap)

    return cap_matrix


def build_eligibility_mask(cards: List[dict], profile: Dict[str, Any]) -> np.ndarray:
    """
    Returns boolean ndarray shape (n_cards,).
    True = card is eligible to be assigned spend.

    When the user has specified cards_owned, ONLY those cards are eligible
    for spend allocation — this keeps the monthly plan honest.
    When no cards are owned, fall back to any active card with decent
    approval odds (first-run / no-card users get a broader recommendation).
    """
    owned = set(profile.get("cards_owned", []) or [])

    mask = np.zeros((len(cards),), dtype=bool)
    for i, card in enumerate(cards):
        cid = card.get("id")
        approval_prob = float(card.get("approval_prob", 0.0) or 0.0)
        is_active = bool(card.get("is_active", True))

        if owned:
            # Strict mode: only allocate from cards the user actually has
            eligible = (str(cid) in owned and is_active)
        else:
            # Open mode: no cards specified — use any active card with decent odds
            eligible = (is_active and approval_prob >= 0.50)

        mask[i] = bool(eligible)

    return mask

