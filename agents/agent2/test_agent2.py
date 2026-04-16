from agents.agent2 import run_agent2


def test_greedy_basic():
    # Mock a minimal card list and profile
    # Axis Atlas should win travel (10% after transfer uplift)
    # Amex MRCC should win dining (8% at partner restaurants)
    # ICICI Amazon Pay should win online (5% cashback)

    mock_cards = [
        {
            "id": "axis-atlas-1",
            "name": "Axis Atlas",
            "issuer": "Axis",
            "network": "visa",
            "annual_fee_inr": 5000,
            "forex_fee_pct": 0.0,
            "fuel_surcharge_waiver": False,
            "fuel_surcharge_cap_inr": None,
            "reward_rates": [
                {
                    "category": "travel",
                    "points_per_100": 10.0,
                    "monthly_cap_inr": None,
                    "reward_type": "miles",
                    "rate_variance_pct": 0.20,
                },
                {
                    "category": "dining",
                    "points_per_100": 1.0,
                    "monthly_cap_inr": None,
                    "reward_type": "miles",
                    "rate_variance_pct": 0.20,
                },
                {
                    "category": "online",
                    "points_per_100": 1.0,
                    "monthly_cap_inr": None,
                    "reward_type": "miles",
                    "rate_variance_pct": 0.20,
                },
            ],
            "point_value_inr": 0.20,  # low base redemption
            "monthly_total_cap_inr": None,
            "reward_expiry_months": None,
            "transfer_partners": [
                {
                    "partner_program": "Air India Flying Returns",
                    "partner_type": "airline",
                    "transfer_ratio": 2.0,
                    "min_transfer_pts": 5000,
                    "transfer_fee_inr": 0,
                    "cpp_inr": 2.00,
                }
            ],
            "welcome_bonus_pts": 5000,
            "welcome_spend_inr": 100000,
            "welcome_months": 3,
            "credit_score_min": 750,
            "approval_prob": 0.80,
            "is_active": True,
        },
        {
            "id": "amex-mrcc-1",
            "name": "Amex MRCC",
            "issuer": "Amex",
            "network": "amex",
            "annual_fee_inr": 1500,
            "forex_fee_pct": 3.5,
            "fuel_surcharge_waiver": False,
            "fuel_surcharge_cap_inr": None,
            "reward_rates": [
                {
                    "category": "dining",
                    "points_per_100": 8.0,
                    "monthly_cap_inr": None,
                    "reward_type": "points",
                    "rate_variance_pct": 0.20,
                },
                {
                    "category": "travel",
                    "points_per_100": 0.8,
                    "monthly_cap_inr": None,
                    "reward_type": "points",
                    "rate_variance_pct": 0.20,
                },
                {
                    "category": "online",
                    "points_per_100": 1.0,
                    "monthly_cap_inr": None,
                    "reward_type": "points",
                    "rate_variance_pct": 0.20,
                },
            ],
            "point_value_inr": 1.00,
            "monthly_total_cap_inr": None,
            "reward_expiry_months": None,
            "transfer_partners": [],
            "welcome_bonus_pts": 0,
            "welcome_spend_inr": 0,
            "welcome_months": 0,
            "credit_score_min": 750,
            "approval_prob": 0.78,
            "is_active": True,
        },
        {
            "id": "icici-amazon-1",
            "name": "ICICI Amazon Pay",
            "issuer": "ICICI",
            "network": "visa",
            "annual_fee_inr": 0,
            "forex_fee_pct": 0.0,
            "fuel_surcharge_waiver": False,
            "fuel_surcharge_cap_inr": None,
            "reward_rates": [
                {
                    "category": "online",
                    "points_per_100": 5.0,
                    "monthly_cap_inr": None,
                    "reward_type": "cashback",
                    "rate_variance_pct": 0.20,
                },
                {
                    "category": "dining",
                    "points_per_100": 1.0,
                    "monthly_cap_inr": None,
                    "reward_type": "cashback",
                    "rate_variance_pct": 0.20,
                },
                {
                    "category": "travel",
                    "points_per_100": 1.0,
                    "monthly_cap_inr": None,
                    "reward_type": "cashback",
                    "rate_variance_pct": 0.20,
                },
            ],
            "point_value_inr": 1.00,
            "monthly_total_cap_inr": None,
            "reward_expiry_months": None,
            "transfer_partners": [],
            "welcome_bonus_pts": 0,
            "welcome_spend_inr": 0,
            "welcome_months": 0,
            "credit_score_min": 700,
            "approval_prob": 0.77,
            "is_active": True,
        },
    ]

    mock_profile = {
        "session_id": "test-123",
        "goal_text": "Japan trip",
        "goal_amount_inr": 200000,
        "timeline_months": 4,
        "monthly_spend_inr": 40000,
        "cards_owned": [],
        "risk_level": "medium",
        "credit_score_range": "750+",
        "spend_breakdown": {
            "groceries": 8000,
            "dining": 5000,
            "travel": 10000,
            "fuel": 3000,
            "online": 7000,
            "entertainment": 2000,
            "utilities": 3000,
            "other": 2000,
        },
    }

    result = run_agent2(mock_profile, mock_cards)

    assert result.monthly_plan is not None
    assert len(result.monthly_plan) == 4
    assert result.total_rewards_inr > 0
    assert result.net_value_inr == result.total_rewards_inr - result.total_fees_inr

    # Axis Atlas must win travel due to transfer uplift
    travel_alloc = next(
        a
        for month in result.monthly_plan
        for a in month.allocations
        if a.category == "travel"
    )
    assert travel_alloc.card_name == "Axis Atlas"

    print("✓ Agent 2 greedy allocation passed")
    print(f"  Total rewards: ₹{result.total_rewards_inr:,.0f}")
    print(f"  Total fees:    ₹{result.total_fees_inr:,.0f}")
    print(f"  Net value:     ₹{result.net_value_inr:,.0f}")


def test_mc_basic():
    import time
    from agents.agent2.graph import monte_carlo_search, beam_search, graph_to_monthly_plan

    mock_cards = [
        {
            "id": "high-risk-1",
            "name": "High Risk High Reward",
            "issuer": "Bank",
            "network": "visa",
            "annual_fee_inr": 12000,
            "forex_fee_pct": 0.0,
            "fuel_surcharge_waiver": False,
            "reward_rates": [
                {
                    "category": "travel",
                    "points_per_100": 10.0,
                    "monthly_cap_inr": None,
                    "reward_type": "miles"
                }
            ],
            "point_value_inr": 1.0,
            "transfer_partners": [],
            "welcome_bonus_pts": 0,
            "welcome_spend_inr": 0,
            "welcome_months": 0,
            "credit_score_min": 750,
            "approval_prob": 1.0,
            "is_active": True,
        },
        {
            "id": "safe-card-1",
            "name": "Safe Low Fee Card",
            "issuer": "Bank",
            "network": "visa",
            "annual_fee_inr": 500,
            "forex_fee_pct": 0.0,
            "fuel_surcharge_waiver": False,
            "reward_rates": [
                {
                    "category": "travel",
                    "points_per_100": 3.0,
                    "monthly_cap_inr": None,
                    "reward_type": "cashback"
                }
            ],
            "point_value_inr": 1.0,
            "transfer_partners": [],
            "welcome_bonus_pts": 0,
            "welcome_spend_inr": 0,
            "welcome_months": 0,
            "credit_score_min": 750,
            "approval_prob": 1.0,
            "is_active": True,
        }
    ]

    mock_profile = {
        "session_id": "test-mc-123",
        "goal_text": "Japan trip",
        "goal_amount_inr": 200000,
        "timeline_months": 12,
        "monthly_spend_inr": 15000,
        "cards_owned": [],
        "risk_level": "medium",
        "credit_score_range": "750+",
        "spend_breakdown": {
            "travel": 15000,
        },
    }

    start_time = time.time()
    
    # 1. Low CV run (CV=0.01)
    res_low = monte_carlo_search(mock_cards, mock_profile, n_simulations=50, cv=0.01)
    print(f"MC Low CV time: {time.time() - start_time:.2f}s")
    
    assert res_low["p10_net_value"] <= res_low["expected_net_value"] <= res_low["p90_net_value"]
    assert round(sum(res_low["card_win_rate"].values()), 2) == 1.0
    
    path, totals = beam_search(mock_cards, mock_profile, beam_width=10)
    _, recommended = graph_to_monthly_plan(path, mock_cards, mock_profile)
    beam_winner = recommended[0].card_id if recommended else None
    
    mc_winner_low = max(res_low["card_win_rate"].items(), key=lambda x: x[1])[0]
    assert mc_winner_low == beam_winner, f"Expected {beam_winner}, got {mc_winner_low}"
    
    # 2. High CV run (CV=0.80)
    res_high = monte_carlo_search(mock_cards, mock_profile, n_simulations=50, cv=0.80)
    mc_winner_high = max(res_high["card_win_rate"].items(), key=lambda x: x[1])[0]
    
    print(f"Beam winner: {beam_winner}")
    print(f"MC Low CV winner: {mc_winner_low}")
    print(f"MC High CV winner: {mc_winner_high}")
    print(f"High CV Win Rates: {res_high['card_win_rate']}")
    
    # At very high CV, the high risk card might swing into negatives due to fee vs very low noise outcomes, 
    # bringing the safe card up in win rate.
    if mc_winner_high != beam_winner:
        print("MC high CV winner differs from beam winner (Expected behavior via uncertainty).")

if __name__ == "__main__":
    test_greedy_basic()
    test_mc_basic()

