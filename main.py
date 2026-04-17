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
origins = {frontend_url, "http://localhost:5173", "http://localhost:3000", "http://localhost:8000", "http://localhost:5174", "http://localhost:5175", "https://voynt-final.vercel.app/"}

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


@app.get("/api/graph/{session_id}")
@app.get("/api/graph/{session_id}")
async def get_strategy_graph(session_id: str) -> JSONResponse:
    """
    Build a rigidly mapped PyVis-compatible graph mapping cards sequentially across time.
    X-axis: Time (Month progression fixed per 250px)
    Y-axis: Dedicated Card 'Swimlanes'
    """
    strategy_data = get_strategy_result(session_id)
    if strategy_data is None:
        raise HTTPException(status_code=404, detail="Strategy result not found")

    monthly_plan = strategy_data.get("monthly_plan") or []

    # Get totals to find best card & assign Y coordinates
    card_totals: dict[str, dict] = {}
    for month_data in monthly_plan:
        for alloc in (month_data.get("allocations") or []):
            cid = alloc.get("card_id", "")
            if cid not in card_totals:
                card_totals[cid] = {"name": alloc.get("card_name", "Card"), "total_value": 0.0}
            if alloc.get("category") != "annual_fee":
                card_totals[cid]["total_value"] += float(alloc.get("expected_value_inr") or 0.0)

    # Assign Y coordinates arbitrarily since we are removing 'best' grouping
    unique_cids = list(card_totals.keys())
    
    y_map = {}
    y_offsets = [0, 150, -150, 300, -300, 450, -450]
    for idx, cid in enumerate(unique_cids):
        y_map[cid] = y_offsets[idx % len(y_offsets)]

    nodes: list[dict] = []
    edges: list[dict] = []

    # Center Goal Node (Start Point)
    center_id = "goal"
    nodes.append({
        "id": center_id, 
        "label": "Start Phase", 
        "group": "goal", 
        "size": 24, 
        "is_optimal": True,
        "x": -200, 
        "y": 0, 
        "fixed": {"x": True, "y": True}
    })
    
    last_card_node = {} # Track the last chronological node for each card

    for idx, month_data in enumerate(monthly_plan):
        m_num = month_data.get("month", idx + 1)
        x_coord = m_num * 320
        
        allocs_by_card = {}
        for alloc in (month_data.get("allocations") or []):
            cid = alloc.get("card_id", "")
            if cid not in allocs_by_card:
                allocs_by_card[cid] = []
            allocs_by_card[cid].append(alloc)
            
        for cid, allocs in allocs_by_card.items():
            cname = card_totals.get(cid, {}).get("name", "Card")
            y_coord = y_map.get(cid, 0)
            
            node_id = f"m{m_num}_c{cid}"
            
            nodes.append({
                "id": node_id,
                "label": f"M{m_num}: {cname[:14]}",
                "group": "card",
                "size": 16,
                "is_optimal": False,
                "x": x_coord, 
                "y": y_coord, 
                "fixed": {"x": True, "y": True},
                "title": f"{cname} used in Month {m_num}"
            })
            
            # Connect edge from previous
            prev_node = last_card_node.get(cid, center_id)
            
            val_sum = sum([float(a.get("expected_value_inr") or 0.0) for a in allocs if a.get("category") != "annual_fee"])
            cats = [a.get("category") for a in allocs if a.get("category") != "annual_fee"]
            fees = [float(a.get("amount_inr") or 0.0) for a in allocs if a.get("category") == "annual_fee"]
            
            if fees:
                edge_label = f"Fee (₹{sum(fees):,.0f})"
                dashes = True
            else:
                cat_str = ", ".join([c.capitalize() for c in set(cats)])
                if len(cat_str) > 18: 
                    cat_str = cat_str[:15] + ".."
                edge_label = f"₹{val_sum:,.0f} ({cat_str})"
                dashes = False
                
            edges.append({
                "from": prev_node,
                "to": node_id,
                "label": edge_label,
                "is_optimal": False,
                "width": 1,
                "dashes": dashes,
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.4}},
            })
            
            last_card_node[cid] = node_id

    return JSONResponse(content={"nodes": nodes, "edges": edges, "best_card_id": None})


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

