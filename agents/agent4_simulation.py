from typing import Dict

import numpy as np

from models import SimulationResult, StrategyPlan, UserProfile


RISK_SIGMA: Dict[str, float] = {
    "low": 0.10,
    "medium": 0.20,
    "high": 0.35,
}


async def run_agent4(strategy: StrategyPlan, profile: UserProfile) -> SimulationResult:
    """
    Monte Carlo simulation stub. Returns 0.78 success probability as requested.
    """
    base_value = float(strategy.net_value_inr or strategy.total_rewards_inr or 0.0)

    return SimulationResult(
        session_id=strategy.session_id,
        success_probability=0.78,
        p10_inr=base_value * 0.9,
        p50_inr=base_value,
        p90_inr=base_value * 1.1,
        worst_case_inr=base_value * 0.8,
        best_case_inr=base_value * 1.2,
        risk_score=22.0,
        histogram_data=[],
        run_count=0,
    )

