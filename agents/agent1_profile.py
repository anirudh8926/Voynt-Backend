from typing import cast

from models import AnalyzeRequest, UserProfile
from db import supabase


def run_agent1(request: AnalyzeRequest) -> UserProfile:
    """
    Parse and validate the onboarding form submission.
    - Validate goal_amount_inr > 0, timeline_months 1-60
    - Validate all card UUIDs in cards_owned exist in the cards table
    - Validate spend_breakdown categories sum <= monthly_spend_inr
    - Return structured UserProfile
    """
    if request.goal_amount_inr <= 0:
        raise ValueError("goal_amount_inr must be greater than 0")
    if not (1 <= request.timeline_months <= 60):
        raise ValueError("timeline_months must be between 1 and 60")

    if request.cards_owned:
        result = (
            supabase.table("cards")
            .select("id, name")
            .in_("name", request.cards_owned)
            .execute()
        )
        rows = cast(list[dict], result.data or [])
        if not rows:
            raise ValueError(
                "None of the selected cards were recognised. "
                "Please select at least one valid card and try again."
            )
        # Replace incoming name strings with the resolved UUIDs (drop unknowns silently)
        request.cards_owned = [row["id"] for row in rows]

    spend_sum = sum(
        [
            request.spend_breakdown.groceries,
            request.spend_breakdown.dining,
            request.spend_breakdown.travel,
            request.spend_breakdown.fuel,
            request.spend_breakdown.online,
            request.spend_breakdown.entertainment,
            request.spend_breakdown.utilities,
            request.spend_breakdown.other,
        ]
    )
    if spend_sum > request.monthly_spend_inr and spend_sum > 0:
        scale = request.monthly_spend_inr / spend_sum
        request.spend_breakdown.groceries     = round(request.spend_breakdown.groceries     * scale)
        request.spend_breakdown.dining        = round(request.spend_breakdown.dining        * scale)
        request.spend_breakdown.travel        = round(request.spend_breakdown.travel        * scale)
        request.spend_breakdown.fuel          = round(request.spend_breakdown.fuel          * scale)
        request.spend_breakdown.online        = round(request.spend_breakdown.online        * scale)
        request.spend_breakdown.entertainment = round(request.spend_breakdown.entertainment * scale)
        request.spend_breakdown.utilities     = round(request.spend_breakdown.utilities     * scale)
        request.spend_breakdown.other         = round(request.spend_breakdown.other         * scale)

    spend_dict = {
        "groceries":     request.spend_breakdown.groceries,
        "dining":        request.spend_breakdown.dining,
        "travel":        request.spend_breakdown.travel,
        "fuel":          request.spend_breakdown.fuel,
        "online":        request.spend_breakdown.online,
        "entertainment": request.spend_breakdown.entertainment,
        "utilities":     request.spend_breakdown.utilities,
        "other":         request.spend_breakdown.other,
    }

    return UserProfile(
        session_id="",
        goal_text=request.goal_text,
        goal_amount_inr=request.goal_amount_inr,
        timeline_months=request.timeline_months,
        monthly_spend_inr=request.monthly_spend_inr,
        cards_owned=request.cards_owned,
        risk_level=request.risk_level,
        credit_score_range=request.credit_score_range,
        spend_breakdown=spend_dict,
    )

