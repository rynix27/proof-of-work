"""Dashboard-state derivation: tolerance, honesty (None not fake zeros), and agreement with the ledger."""
import pytest

import core.orchestrator as orch
from core import events
from core.state import build_state


def started(agents=("A", "B"), **kw):
    return {"type": "simulation_started", "agents": list(agents), "starting_balance": 100, "mode": "mock", **kw}


def test_empty_log_is_idle_with_no_invented_numbers():
    s = build_state([])
    assert s["status"] == "idle" and s["stats"]["success_rate"] is None
    assert s["stats"]["compute_per_correct"] is None and s["stats"]["repayment_rate"] is None


@pytest.mark.parametrize("junk", [None, 5, "x", [], {}, {"type": None}, {"type": 7}, {"no": "type"}])
def test_malformed_events_are_skipped_never_raise(junk):
    s = build_state([started(), junk, {"type": "task_resolved"}, {"type": "loan_repayment", "loan_id": "zzz"},
                     {"type": "loan_default", "loan_id": "zzz"}])
    assert s["data_quality"]["events"] >= 1


def test_accuracy_excludes_exhibition_and_unclaimed_tasks():
    evs = [started(),
           {"type": "task_arrived", "round": 1}, {"type": "task_arrived", "round": 2},
           {"type": "task_arrived", "round": 3}, {"type": "task_unclaimed", "round": 3},
           {"type": "task_resolved", "agent": "A", "tier": "nano", "correct": True, "bounty": 8, "cost": 1, "new_balance": 107},
           {"type": "task_resolved", "agent": "B", "tier": "ultra", "correct": False, "bounty": 0, "cost": 12, "new_balance": 88},
           {"type": "task_resolved", "agent": "A", "tier": "nano", "correct": None, "exhibition": True, "bounty": 0, "cost": 1, "new_balance": 106}]
    st = build_state(evs)["stats"]
    assert st["tasks_total"] == 3 and st["tasks_unclaimed"] == 1
    assert st["success_rate"] == 0.5                       # 1 of 2 GRADED; exhibition excluded
    assert st["compute_burned"] == 14 and st["rewards_total"] == 8
    assert st["compute_per_correct"] == 14.0
    assert st["escalation_rate"] == 0.0


def test_legacy_v1_events_without_cost_are_costed_from_tier():
    evs = [started(), {"type": "task_resolved", "agent": "A", "tier": "ultra", "correct": True, "bounty": 20, "new_balance": 108}]
    assert build_state(evs)["stats"]["compute_burned"] == 12


def test_loan_lifecycle_tracking_and_repayment_rate():
    evs = [started(),
           {"type": "loan_made", "loan_id": 1, "lender": "A", "borrower": "B", "principal": 20, "amount": 20,
            "total_owed": 23, "interest_rate": 0.15, "due_round": 6, "lender_balance": 80, "borrower_balance": 120},
           {"type": "loan_made", "loan_id": 2, "lender": "A", "borrower": "B", "principal": 10, "amount": 10,
            "total_owed": 12, "interest_rate": 0.15, "due_round": 6, "lender_balance": 70, "borrower_balance": 130},
           {"type": "loan_repayment", "loan_id": 1, "amount": 23, "outstanding": 0, "status": "repaid",
            "lender_balance": 93, "borrower_balance": 107},
           {"type": "loan_default", "loan_id": 2, "written_off": 12}]
    st = build_state(evs)["stats"]
    assert (st["loans_made"], st["loans_repaid"], st["loans_defaulted"]) == (2, 1, 1)
    assert st["loan_volume"] == 30 and st["outstanding_debt"] == 0
    assert st["repayment_rate"] == round(23 / 35, 3)


def test_rankings_require_a_minimum_sample_before_ranking_accuracy():
    evs = [started(),
           {"type": "task_resolved", "agent": "A", "tier": "nano", "correct": True, "bounty": 8, "cost": 1, "new_balance": 107}]
    assert build_state(evs)["rankings"]["accuracy"] == []      # 1 attempt is not a ranking


def test_state_does_not_mutate_its_input():
    import copy
    evs = [started(), {"type": "task_resolved", "agent": "A", "tier": "nano", "correct": True, "bounty": 8, "cost": 1, "new_balance": 107}]
    snap = copy.deepcopy(evs)
    build_state(evs)
    assert evs == snap


def test_derived_stats_agree_with_ledger_truth_after_a_real_run(isolated_data_dir):
    """The dashboard's numbers (from events) must equal the simulation's own accounting."""
    events.reset_log()
    orch.run_simulation(max_rounds=50, round_delay=0, verbose=False, seed=21)
    evs = events.read_all()
    ended = [e for e in evs if e["type"] == "simulation_ended"][0]
    st = build_state(evs)
    truth = ended["agent_stats"]
    assert st["stats"]["compute_burned"] == sum(a["compute_spent"] for a in truth.values())
    assert st["stats"]["rewards_total"] == sum(a["rewards_earned"] for a in truth.values())
    assert st["stats"]["tasks_correct"] == sum(a["correct"] for a in truth.values())
    assert st["stats"]["loans_made"] == ended["loans"]["loans_made"]
    assert st["stats"]["outstanding_debt"] == ended["loans"]["outstanding_debt"]
    assert st["balances"] == ended["final_balances"] and st["status"] == "ended"
    for name, a in truth.items():
        assert st["agents"][name]["net_pl"] == a["net_pl"] and st["agents"][name]["solved"] == a["correct"]
