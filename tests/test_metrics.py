"""Metric definitions (docs/METRICS.md) verified against hand-computed values, plus calibration."""
import pytest

import core.orchestrator as orch
from core import events
from core.calibration import MIN_RELIABLE_N, calibration_report
from core.state import build_state


def resolved(agent, tier, correct, bounty, cost, conf=None, p=None, exhibition=False, **kw):
    return {"type": "task_resolved", "agent": agent, "tier": tier, "correct": correct, "bounty": bounty, "cost": cost,
            "exhibition": exhibition, "bid_confidence": conf, "p_est": p, "new_balance": 100, "valid_response": True, **kw}


def stream():
    """Fully specified run. Expected values are derived by hand in the tests below."""
    return [
        {"type": "simulation_started", "agents": ["A", "B"], "starting_balance": 100, "mode": "mock",
         "strategies": {"A": "lender", "B": "aggressive"}},
        {"type": "task_arrived", "round": 1}, {"type": "task_arrived", "round": 2},
        {"type": "task_arrived", "round": 3}, {"type": "task_arrived", "round": 4},
        {"type": "task_arrived", "round": 5},
        {"type": "task_unclaimed", "round": 5},
        resolved("A", "nano", True, 8, 1, conf=0.9, p=0.8),          # +8 -1
        resolved("B", "ultra", True, 20, 12, conf=0.4, p=0.5),       # +20 -12
        resolved("A", "nano", False, 0, 1, conf=0.7, p=0.6),         # -1
        resolved("A", "nano", None, 0, 1, exhibition=True),          # ungraded: costs, no accuracy
        {"type": "loan_made", "loan_id": "t3", "lender": "A", "borrower": "B", "principal": 20, "amount": 20,
         "total_owed": 23, "interest_rate": 0.15, "due_round": 6, "lender_balance": 80, "borrower_balance": 120, "round": 2},
        {"type": "loan_made", "loan_id": "t4", "lender": "A", "borrower": "B", "principal": 10, "amount": 10,
         "total_owed": 12, "interest_rate": 0.15, "due_round": 6, "lender_balance": 70, "borrower_balance": 130, "round": 3},
        {"type": "loan_repayment", "loan_id": "t3", "amount": 23, "outstanding": 0, "status": "repaid",
         "lender_balance": 93, "borrower_balance": 107, "borrower": "B", "lender": "A"},
        {"type": "loan_repayment", "loan_id": "t4", "amount": 5, "outstanding": 7, "status": "active",
         "lender_balance": 98, "borrower_balance": 102, "borrower": "B", "lender": "A"},
        {"type": "loan_default", "loan_id": "t4", "written_off": 7},
        {"type": "round_snapshot", "round": 5, "balances": {"A": 98, "B": 102}, "bankrupt": []},
    ]


def test_headline_metrics_match_hand_computation():
    s = build_state(stream())["stats"]
    assert s["tasks_total"] == 5 and s["tasks_unclaimed"] == 1
    assert s["tasks_attempted"] == 4                                   # 4 solves
    assert s["tasks_resolved"] == 3 and s["tasks_correct"] == 2        # graded only
    assert s["success_rate"] == round(2 / 3, 3)
    assert s["task_completion_rate"] == round(4 / 5, 3)                # solves / tasks arrived
    assert s["compute_burned"] == 1 + 12 + 1 + 1 == 15
    assert s["rewards_total"] == 28 and s["economic_profit"] == 28 - 15
    assert s["compute_per_correct"] == round(15 / 2, 2)
    assert s["loans_made"] == 2 and s["loan_volume"] == 30


def test_loan_metrics_match_hand_computation():
    s = build_state(stream())["stats"]
    assert (s["loans_repaid"], s["loans_defaulted"]) == (1, 1)
    assert s["loan_default_rate"] == 0.5                               # 1 default / (1 repaid + 1 defaulted)
    assert s["repayment_rate"] == round((23 + 5) / (23 + 12), 3)       # repaid / owed on settled loans
    assert s["outstanding_debt"] == 0


def test_escalation_effectiveness_table_matches_hand_computation():
    t = build_state(stream())["tiers"]
    assert t["nano"] == {**t["nano"], "solves": 3, "graded": 2, "correct": 1, "compute": 3, "rewards": 8,
                         "accuracy": 0.5, "net_profit": 5, "profit_per_solve": round(5 / 3, 2)}
    assert t["ultra"] == {**t["ultra"], "solves": 1, "graded": 1, "correct": 1, "compute": 12, "rewards": 20,
                          "accuracy": 1.0, "net_profit": 8, "avg_cost": 12.0}


def test_agent_profile_metrics_match_hand_computation():
    a = build_state(stream())["agents"]
    assert a["A"]["strategy"] == "lender" and a["B"]["strategy"] == "aggressive"
    assert a["A"]["economic_profit"] == 8 - 3 == 5                     # rewards 8, compute 1+1+1
    assert a["B"]["economic_profit"] == 20 - 12 == 8
    assert a["A"]["loans_issued"] == 2 and a["B"]["loans_issued"] == 0
    assert a["A"]["accuracy"] == 0.5 and a["B"]["accuracy"] == 1.0
    assert a["A"]["net_pl"] == 98 - 100 and a["B"]["net_pl"] == 102 - 100      # balance change, includes transfers


def test_net_pl_and_economic_profit_are_different_things():
    """Interest and bounties both move balances, but only rewards - compute is *economic* profit."""
    a = build_state(stream())["agents"]["A"]
    assert a["net_pl"] != a["economic_profit"]


def test_series_is_cumulative_and_monotone():
    pts = build_state(stream())["series"]
    assert pts == [{"round": 5, "tasks": 5, "graded": 3, "correct": 2, "accuracy": round(2 / 3, 3),
                    "compute": 15, "rewards": 28, "loan_volume": 30, "bankrupt": 0}]


def test_every_ratio_is_none_not_zero_when_its_denominator_is_empty():
    s = build_state([{"type": "simulation_started", "agents": ["A"], "starting_balance": 100}])["stats"]
    for k in ("success_rate", "task_completion_rate", "compute_per_correct", "loan_default_rate",
              "repayment_rate", "escalation_rate", "invalid_answer_rate", "policy_override_rate"):
        assert s[k] is None, k
    assert s["economic_profit"] == 0 and s["bankruptcy_rate"] == 0.0


def test_bankruptcy_rate():
    evs = [{"type": "simulation_started", "agents": ["A", "B", "C", "D"], "starting_balance": 100},
           {"type": "round_snapshot", "round": 1, "balances": {"A": 0, "B": 0, "C": 5, "D": 9}, "bankrupt": ["A", "B"]}]
    assert build_state(evs)["stats"]["bankruptcy_rate"] == 0.5


def test_policy_overrides_and_denials_are_counted():
    evs = [{"type": "simulation_started", "agents": ["A"], "starting_balance": 100},
           {"type": "decision", "round": 1, "agent": "A", "policy_reasons": ["ev_below_min"]},
           {"type": "decision", "round": 1, "agent": "A", "policy_reasons": []},
           {"type": "escalation_denied", "reasons": ["escalation_budget_exhausted"]}]
    s = build_state(evs)["stats"]
    assert s["policy_override_rate"] == 0.5 and s["escalations_denied"] == 1


def test_provider_usage_aggregation_and_cost_honesty():
    u = lambda c, **k: {"type": "usage", "usage": {"calls": 5, "calls_with_usage": 4, "input_tokens": 1000, "output_tokens": 200},
                        "estimated_cost_usd": c}
    st = build_state([u(0.01), u(0.02)])
    assert st["provider_usage"]["input_tokens"] == 2000 and st["provider_usage"]["calls_with_usage"] == 8
    assert st["provider_usage"]["estimated_cost_usd"] == pytest.approx(0.03)
    st = build_state([u(0.01), u(None)])                           # one round had no prices -> total unknown
    assert st["provider_usage"]["estimated_cost_usd"] is None
    assert build_state([])["provider_usage"] is None               # mock mode: nothing invented


def test_real_run_aggregates_are_internally_consistent(isolated_data_dir):
    events.reset_log()
    orch.run_simulation(max_rounds=60, round_delay=0, verbose=False, seed=5)
    st = build_state(events.read_all())
    s, tiers = st["stats"], st["tiers"]
    assert sum(t["compute"] for t in tiers.values()) == s["compute_burned"]
    assert sum(t["rewards"] for t in tiers.values()) == s["rewards_total"]
    assert sum(t["graded"] for t in tiers.values()) == s["tasks_resolved"]
    pts = st["series"]
    assert len(pts) == 60 and [p["round"] for p in pts] == list(range(1, 61))
    for key in ("tasks", "graded", "correct", "compute", "rewards", "loan_volume"):
        vals = [p[key] for p in pts]
        assert vals == sorted(vals), key                           # cumulative => non-decreasing
    assert pts[-1]["compute"] == s["compute_burned"] and pts[-1]["loan_volume"] == s["loan_volume"]
    assert sum(a["economic_profit"] for a in st["agents"].values()) == s["economic_profit"]


# ------------------------------------------------------------- calibration
def cal(pairs, **kw):
    return calibration_report([resolved("A", "nano", bool(y), 8, 1, conf=p, p=p, **kw) for p, y in pairs])


def test_perfectly_calibrated_predictions_have_zero_ece_and_the_textbook_brier_score():
    r = cal([(0.8, 1)] * 8 + [(0.8, 0)] * 2)["self_reported"]
    assert r["n"] == 10 and r["ece"] == 0.0
    assert r["brier"] == pytest.approx((8 * 0.2 ** 2 + 2 * 0.8 ** 2) / 10)         # = 0.16


def test_overconfidence_is_exposed():
    r = cal([(0.9, 0)] * 40)["self_reported"]
    assert r["brier"] == pytest.approx(0.81) and r["ece"] == pytest.approx(0.9) and r["reliable"] is True
    b = [x for x in r["bins"] if x["n"]][0]
    assert b["mean_predicted"] == 0.9 and b["observed_accuracy"] == 0.0


def test_small_samples_are_flagged_unreliable():
    r = cal([(0.5, 1)] * (MIN_RELIABLE_N - 1))["self_reported"]
    assert r["reliable"] is False and "indicative" in r["note"]
    assert cal([(0.5, 1)] * MIN_RELIABLE_N)["self_reported"]["reliable"] is True


def test_probability_edges_land_in_valid_bins():
    r = cal([(0.0, 0), (1.0, 1), (0.2, 1), (0.999, 1)])["self_reported"]
    assert r["n"] == 4 and sum(b["n"] for b in r["bins"]) == 4 and r["bins"][-1]["n"] == 2


def test_ungraded_and_malformed_predictions_are_excluded_not_imputed():
    evs = [resolved("A", "nano", None, 0, 1, conf=0.9, p=0.9, exhibition=True),
           resolved("A", "nano", True, 8, 1, conf=None, p=None),
           resolved("A", "nano", True, 8, 1, conf=1.5, p=-0.2),
           resolved("A", "nano", True, 8, 1, conf="0.9", p=True),
           resolved("A", "nano", True, 8, 1, conf=0.6, p=0.6),
           {"type": "decision"}, None, "x"]
    r = calibration_report(evs)
    assert r["self_reported"]["n"] == 1 and r["policy_estimate"]["n"] == 1


def test_empty_calibration_is_none_not_zero():
    r = calibration_report([])["self_reported"]
    assert r["n"] == 0 and r["brier"] is None and r["ece"] is None and r["reliable"] is False


def test_policy_estimate_and_self_reported_are_evaluated_separately():
    evs = [resolved("A", "nano", True, 8, 1, conf=0.9, p=0.5), resolved("A", "nano", False, 0, 1, conf=0.9, p=0.5)]
    r = calibration_report(evs)
    assert r["self_reported"]["brier"] == pytest.approx((0.01 + 0.81) / 2)
    assert r["policy_estimate"]["brier"] == pytest.approx(0.25)
