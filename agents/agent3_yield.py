from models import StrategyPlan, UserProfile, YieldResult


def run_agent3(strategy: StrategyPlan, profile: UserProfile) -> YieldResult:
    """
    Compute yield index: net reward value as a fraction of total card spend.

    Formula: (total_rewards - total_fees) / total_spend
    - No FD opportunity cost: credit card spend is discretionary consumption,
      not investable savings, so an FD hurdle rate doesn't apply here.

    Efficiency score: how much better the plan is vs a flat 1% cashback card.
    Break-even month: first month where cumulative rewards exceed cumulative fees.
    """
    total_spend = profile.monthly_spend_inr * profile.timeline_months
    if total_spend <= 0:
        return YieldResult(
            session_id=strategy.session_id,
            yield_index=0.0,
            break_even_month=profile.timeline_months,
            total_rewards_inr=strategy.total_rewards_inr,
            total_fees_inr=strategy.total_fees_inr,
            net_value_inr=strategy.net_value_inr,
            efficiency_score=0.0,
        )

    yield_index = strategy.net_value_inr / total_spend

    # Efficiency score vs a flat 1% cashback baseline (yield_index of 0.01)
    baseline_yield = 0.01
    efficiency_score = max(0.0, (yield_index / baseline_yield) * 50) if baseline_yield > 0 else 0.0
    efficiency_score = min(efficiency_score, 100.0)

    # Break-even month: first month where monthly rewards cover prorated fees
    monthly_rewards = strategy.total_rewards_inr / max(profile.timeline_months, 1)
    monthly_fees = strategy.total_fees_inr / max(profile.timeline_months, 1)
    if monthly_rewards >= monthly_fees:
        # Rewards cover fees from month 1 — find exact break-even
        break_even_month = max(1, round(strategy.total_fees_inr / monthly_rewards)) if monthly_rewards > 0 else profile.timeline_months
    else:
        # Fees never covered within the timeline
        break_even_month = profile.timeline_months

    return YieldResult(
        session_id=strategy.session_id,
        yield_index=yield_index,
        break_even_month=break_even_month,
        total_rewards_inr=strategy.total_rewards_inr,
        total_fees_inr=strategy.total_fees_inr,
        net_value_inr=strategy.net_value_inr,
        efficiency_score=efficiency_score,
    )
