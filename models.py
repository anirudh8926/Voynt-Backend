from __future__ import annotations

from pydantic import BaseModel


# --- Inputs ---


class SpendBreakdown(BaseModel):
    groceries: float = 0
    dining: float = 0
    travel: float = 0
    fuel: float = 0
    online: float = 0
    entertainment: float = 0
    utilities: float = 0
    other: float = 0


class AnalyzeRequest(BaseModel):
    goal_text: str
    goal_amount_inr: float
    timeline_months: int
    monthly_spend_inr: float
    cards_owned: list[str] = []
    risk_level: str = "medium"
    credit_score_range: str = "700-750"
    spend_breakdown: SpendBreakdown


class SandboxRequest(BaseModel):
    session_id: str
    selected_cards: list[str]
    spend_overrides: dict[str, dict[str, float]]


# --- Pipeline data contracts (passed between agents) ---


class UserProfile(BaseModel):
    session_id: str
    goal_text: str
    goal_amount_inr: float
    timeline_months: int
    monthly_spend_inr: float
    cards_owned: list[str]
    risk_level: str
    credit_score_range: str
    spend_breakdown: dict[str, float]


class CardAllocation(BaseModel):
    card_id: str
    card_name: str
    category: str
    amount_inr: float
    expected_pts: float
    expected_value_inr: float


class MonthPlan(BaseModel):
    month: int
    allocations: list[CardAllocation]
    total_expected_value_inr: float


class RecommendedCard(BaseModel):
    card_id: str
    card_name: str
    action: str
    reason: str
    approval_prob: float
    expected_bonus_value_inr: float = 0


class MCResult(BaseModel):
    expected_net_value: float
    p10_net_value: float
    p90_net_value: float
    card_win_rate: dict[str, float]


class StrategyPlan(BaseModel):
    session_id: str
    monthly_plan: list[MonthPlan]
    recommended_cards: list[RecommendedCard]
    total_rewards_inr: float
    total_fees_inr: float
    net_value_inr: float
    mc_result: MCResult | None = None


class YieldResult(BaseModel):
    session_id: str
    yield_index: float
    break_even_month: int
    total_rewards_inr: float
    total_fees_inr: float
    net_value_inr: float
    efficiency_score: float


class SimulationResult(BaseModel):
    session_id: str
    success_probability: float
    p10_inr: float
    p50_inr: float
    p90_inr: float
    worst_case_inr: float
    best_case_inr: float
    risk_score: float
    histogram_data: list[dict]
    run_count: int = 10000


# --- API Responses ---


class AnalyzeResponse(BaseModel):
    session_id: str
    status: str = "processing"


class ResultsResponse(BaseModel):
    session_id: str
    status: str
    strategy: StrategyPlan | None = None
    simulation: SimulationResult | None = None
    yield_result: YieldResult | None = None
    ai_narrative: str | None = None


class SandboxResponse(BaseModel):
    computed_rewards_inr: float
    yield_index: float
    diff_vs_ai_inr: float
    simulation_result: SimulationResult | None = None

