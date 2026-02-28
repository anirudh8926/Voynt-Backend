import os
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from db import get_all_cards, get_simulation_result, get_strategy_result, supabase
from agents.agent1_profile import run_agent1
from agents.agent6_sandbox import run_agent6
from models import (
    AnalyzeRequest,
    AnalyzeResponse,
    ResultsResponse,
    SandboxRequest,
    SandboxResponse,
    StrategyPlan,
    UserProfile,
    YieldResult,
)
from pipeline import run_pipeline


app = FastAPI(title="Voint Backend")


frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")
origins = {frontend_url, "http://localhost:5173", "http://localhost:3000", "http://localhost:8000", "http://localhost:5174", "http://localhost:5175"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run_pipeline_sync(profile: UserProfile, cards: List[Dict[str, Any]]) -> None:
    import asyncio

    asyncio.run(run_pipeline(profile, cards))


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> AnalyzeResponse:
    """
    Accept AnalyzeRequest, create a session, and launch the pipeline
    as a background task without blocking the request.
    """
    # Agent 1: build UserProfile (without session_id)
    try:
        profile = run_agent1(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Insert session row into Supabase
    session_insert = (
        supabase.table("sessions")
        .insert(
            {
                "goal_text": profile.goal_text,
                "goal_amount_inr": profile.goal_amount_inr,
                "timeline_months": profile.timeline_months,
                "monthly_spend_inr": profile.monthly_spend_inr,
                "cards_owned": profile.cards_owned,
                "risk_level": profile.risk_level,
                "credit_score_range": profile.credit_score_range,
                "spend_breakdown": profile.spend_breakdown,
                "status": "processing",
            }
        )
        .execute()
    )

    data = session_insert.data or []
    if not data:
        raise HTTPException(status_code=500, detail="Failed to create session")

    session_id = str(data[0]["id"])

    # Attach session_id to profile
    profile = profile.model_copy(update={"session_id": session_id})

    # Fetch cards for the pipeline
    cards = get_all_cards()

    # Launch pipeline as background task (non-blocking)
    background_tasks.add_task(_run_pipeline_sync, profile, cards)

    return AnalyzeResponse(session_id=session_id, status="processing")


@app.get("/api/results/{session_id}", response_model=ResultsResponse)
async def get_results(session_id: str) -> ResultsResponse:
    """
    Fetch session status and, if complete, return joined strategy + simulation results.
    """
    session = (
        supabase.table("sessions")
        .select("*")
        .eq("id", session_id)
        .maybe_single()
        .execute()
        .data
    )

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    status = session.get("status", "pending")

    if status != "complete":
        if status == "failed":
            raise HTTPException(
                status_code=500,
                detail="Pipeline failed for this session",
            )
        return ResultsResponse(
            session_id=session_id,
            status=status,
            strategy=None,
            simulation=None,
            yield_result=None,
            ai_narrative=None,
        )

    strategy_data = get_strategy_result(session_id)
    # Best-effort fetch; simulation is currently bypassed in the API response.
    _simulation_data = get_simulation_result(session_id)

    strategy: Optional[StrategyPlan] = None
    yield_result: Optional[YieldResult] = None
    ai_narrative: Optional[str] = None

    if strategy_data is not None:
        ai_narrative = strategy_data.get("ai_narrative")
        strategy = StrategyPlan.model_validate(strategy_data)

        yield_result = YieldResult(
            session_id=session_id,
            yield_index=float(strategy_data.get("yield_index", 0.0)),
            break_even_month=int(strategy_data.get("break_even_month", 0)),
            total_rewards_inr=float(strategy_data.get("total_rewards_inr", 0.0)),
            total_fees_inr=float(strategy_data.get("total_fees_inr", 0.0)),
            net_value_inr=float(strategy_data.get("net_value_inr", 0.0)),
            efficiency_score=0.0,
        )

    return ResultsResponse(
        session_id=session_id,
        status=status,
        strategy=strategy,
        # Temporarily bypass simulation in the public API until the model
        # is fully defined and wired; always return null here.
        simulation=None,
        yield_result=yield_result,
        ai_narrative=ai_narrative,
    )


@app.get("/api/status/{session_id}")
async def get_status(session_id: str) -> Dict[str, str]:
    """
    Lightweight endpoint that returns only { session_id, status }.
    """
    session = (
        supabase.table("sessions")
        .select("status")
        .eq("id", session_id)
        .maybe_single()
        .execute()
        .data
    )

    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"session_id": session_id, "status": session.get("status", "pending")}


@app.get("/api/debug/{session_id}")
async def debug_session(session_id: str) -> JSONResponse:
    """
    Raw dump of session + strategy_results for a given session_id.
    Use this to verify exactly what the pipeline produced without any
    Pydantic transformation: GET http://localhost:8000/api/debug/<session_id>
    """
    session = (
        supabase.table("sessions")
        .select("*")
        .eq("id", session_id)
        .maybe_single()
        .execute()
        .data
    )
    strategy = get_strategy_result(session_id)
    return JSONResponse(content={
        "session": session,
        "strategy_results": strategy,
    })


@app.get("/api/cards")
async def cards() -> JSONResponse:
    """
    Return all active cards with reward_rates joined.
    Adds Cache-Control header for 1 hour.
    """
    data = get_all_cards()
    response = JSONResponse(content=data)
    response.headers["Cache-Control"] = "max-age=3600"
    return response


@app.post("/api/sandbox", response_model=SandboxResponse)
async def sandbox(request: SandboxRequest) -> SandboxResponse:
    """
    Manual what-if recalculation for sandbox mode.
    """
    cards = get_all_cards()
    strategy_data = get_strategy_result(request.session_id)

    ai_strategy: Optional[StrategyPlan] = None
    if strategy_data is not None:
        ai_strategy = StrategyPlan.model_validate(strategy_data)

    return run_agent6(request, cards, ai_strategy)

