"""Round rules: auction, escalation, failure accounting, loans, reproducibility."""
import json

import pytest

import core.orchestrator as orch
from core import config, events, tasks
from core.agents import fresh_roster
from core.model_client import ModelCallError
from core.simconfig import SimulationConfig


@pytest.fixture
def setup(isolated_data_dir, monkeypatch):
    events.reset_log()
    tasks.reset_ids()
    agents = fresh_roster()
    return agents, monkeypatch


def script(monkeypatch, decisions, solve=None):
    """decisions: {agent_name: raw dict}; default skip."""
    monkeypatch.setattr(orch, "decide_action",
                        lambda agent, task, rn: decisions.get(agent.name, {"action": "skip"}))
    monkeypatch.setattr(orch, "solve_task",
                        solve or (lambda agent, task, escalate, round_num: {"answer": "WRONG", "confidence": 0.5}))


def types():
    return [e["type"] for e in events.read_all()]


def ev(t):
    return [e for e in events.read_all() if e["type"] == t]


def force_task(monkeypatch, answer="42"):
    t = tasks.Task(id=99, prompt="q?", answer=answer, difficulty="easy", bounty=8)
    monkeypatch.setattr(orch.tasks, "generate_task", lambda rng=None, weights=None: t)


def test_highest_confidence_wins(setup):
    agents, mp = setup
    force_task(mp)
    script(mp, {"Specialist": {"action": "bid", "confidence": 0.7},
                "Gambler": {"action": "bid", "confidence": 0.95},
                "Hoarder": {"action": "bid", "confidence": 0.8}})
    orch.run_round(agents, 1)
    assert ev("auction_resolved")[0]["winner"] == "Gambler"


def test_ties_break_by_rotating_order_not_always_first(setup):
    agents, mp = setup
    force_task(mp)
    winners = []
    for rnd in range(1, 6):
        events.reset_log()
        script(mp, {a.name: {"action": "bid", "confidence": 0.9} for a in agents})
        orch.run_round(fresh_roster(), rnd)
        winners.append(ev("auction_resolved")[0]["winner"])
    assert len(set(winners)) > 1                       # fair rotation
    # deterministic: same inputs -> same winner
    events.reset_log()
    script(mp, {a.name: {"action": "bid", "confidence": 0.9} for a in agents})
    orch.run_round(fresh_roster(), 3)
    assert ev("auction_resolved")[0]["winner"] == winners[2]


def test_bid_from_agent_who_cannot_afford_nano_is_rejected_not_crashed(setup):
    agents, mp = setup
    force_task(mp)
    agents[0].balance = 0
    script(mp, {agents[0].name: {"action": "bid", "confidence": 0.99},
                "Gambler": {"action": "bid", "confidence": 0.6}})
    orch.run_round(agents, 1)
    assert ev("bid_rejected")[0]["agent"] == agents[0].name
    assert ev("auction_resolved")[0]["winner"] == "Gambler"


def test_no_bids_leaves_task_unclaimed(setup):
    agents, mp = setup
    force_task(mp)
    script(mp, {})
    orch.run_round(agents, 1)
    assert ev("task_unclaimed")[0]["reason"] == "no_bids"
    assert all(a.balance == 100 for a in agents)


def test_escalation_only_below_cutoff_and_when_affordable(setup):
    agents, mp = setup
    force_task(mp, "WRONG")
    cutoff = config.CONFIDENCE_ESCALATION_CUTOFF
    script(mp, {"Gambler": {"action": "bid", "confidence": cutoff - 0.01}})
    orch.run_round(agents, 1)
    r = ev("task_resolved")[0]
    assert r["tier"] == "ultra" and r["cost"] == config.ULTRA_COST and r["escalated"] is True
    assert len(ev("escalation")) == 1

    events.reset_log()
    agents = fresh_roster()
    script(mp, {"Gambler": {"action": "bid", "confidence": cutoff}})   # at cutoff: no escalation
    orch.run_round(agents, 1)
    assert ev("task_resolved")[0]["tier"] == "nano"

    events.reset_log()
    agents = fresh_roster()
    agents[4].balance = config.ULTRA_COST - 1                            # cannot afford Ultra
    script(mp, {agents[4].name: {"action": "bid", "confidence": 0.1}})
    orch.run_round(agents, 1)
    assert ev("task_resolved")[0]["tier"] == "nano"


def test_correct_answer_pays_bounty_wrong_loses_only_compute(setup):
    agents, mp = setup
    force_task(mp, "42")
    script(mp, {"Gambler": {"action": "bid", "confidence": 0.9}},
           solve=lambda a, t, escalate, round_num: {"answer": 42, "confidence": 0.9})
    orch.run_round(agents, 1)
    g = next(a for a in agents if a.name == "Gambler")
    assert g.balance == 100 - 1 + 8 and g.tasks_correct == 1

    events.reset_log()
    agents = fresh_roster()
    script(mp, {"Gambler": {"action": "bid", "confidence": 0.9}})
    orch.run_round(agents, 1)
    assert next(a for a in agents if a.name == "Gambler").balance == 99


@pytest.mark.parametrize("exc,charged", [
    (ModelCallError("rejected", kind="auth", charged=False), 0),
    (ModelCallError("rate", kind="rate_limit", charged=False), 0),
    (ModelCallError("conn", kind="connection", charged=False), 0),
    (ModelCallError("slow", kind="timeout", charged=True), 1),
    (RuntimeError("something unexpected"), 0),
])
def test_failed_solve_charging_policy(setup, exc, charged):
    agents, mp = setup
    force_task(mp)

    def boom(*a, **k):
        raise exc
    script(mp, {"Gambler": {"action": "bid", "confidence": 0.9}}, solve=boom)
    orch.run_round(agents, 1)
    err = ev("call_error")[0]
    assert err["charged"] == charged and err["purpose"] == "solve"
    assert next(a for a in agents if a.name == "Gambler").balance == 100 - charged
    assert ev("task_unclaimed")[0]["reason"] == "solve_failed"
    assert not ev("task_resolved")


def test_unparseable_or_invalid_answer_is_charged_and_graded_wrong(setup):
    agents, mp = setup
    force_task(mp)
    for bad in ({"_parse_error": True, "_error": "malformed_json"}, {"answer": ["x"]}, {"answer": None}, {}):
        events.reset_log()
        agents = fresh_roster()
        script(mp, {"Gambler": {"action": "bid", "confidence": 0.9}},
               solve=lambda a, t, escalate, round_num, bad=bad: bad)
        orch.run_round(agents, 1)
        r = ev("task_resolved")[0]
        assert r["correct"] is False and r["valid_response"] is False
        assert next(a for a in agents if a.name == "Gambler").balance == 99   # compute spent


@pytest.mark.parametrize("raw", [
    {"action": "bid", "confidence": float("nan")},
    {"action": "bid", "confidence": "high"},
    {"action": "lend", "amount": float("nan")},
    {"action": "lend", "amount": "lots"},
    {"action": "borrow", "amount": -10},
    {"action": "nuke"},
    {"_parse_error": True, "_error": "malformed_json"},
    {"action": "bid"},
])
def test_invalid_model_decisions_become_safe_skips_and_never_corrupt_balances(setup, raw):
    agents, mp = setup
    force_task(mp)
    script(mp, {a.name: raw for a in agents})
    orch.run_round(agents, 1)
    assert len(ev("decision_invalid")) == 5
    assert all(a.balance == 100 and isinstance(a.balance, int) for a in agents)
    assert all(a.invalid_decisions == 1 for a in agents)


def test_decision_call_failure_skips_only_that_agent(setup):
    agents, mp = setup
    force_task(mp)

    def decide(agent, task, rn):
        if agent.name == "Hoarder":
            raise ModelCallError("down", kind="server")
        return {"action": "bid", "confidence": 0.5}
    mp.setattr(orch, "decide_action", decide)
    mp.setattr(orch, "solve_task", lambda *a, **k: {"answer": "x"})
    orch.run_round(agents, 1)
    assert ev("call_error")[0]["agent"] == "Hoarder" and ev("auction_resolved")


def test_lend_validated_clamped_and_reserve_respected(setup):
    agents, mp = setup
    force_task(mp)
    agents[1].balance = 40                                   # a needy borrower target (<50%)
    script(mp, {"Specialist": {"action": "lend", "amount": 1000}})   # absurd offer -> clamped to MAX
    orch.run_round(agents, 1)
    loan = ev("loan_made")[0]
    assert loan["amount"] == config.MAX_LOAN_AMOUNT and loan["total_owed"] == 58
    assert next(a for a in agents if a.name == "Specialist").balance == 50


def test_borrow_picks_richest_lender_and_creates_tracked_loan(setup):
    agents, mp = setup
    force_task(mp)
    agents[3].balance = 300
    script(mp, {"Gambler": {"action": "borrow", "amount": 20}})
    orch.run_round(agents, 1)
    loan = ev("loan_made")[0]
    assert loan["lender"] == agents[3].name and loan["borrower"] == "Gambler" and loan["initiated_by"] == "borrower"
    assert loan["due_round"] == 1 + config.LOAN_TERM_ROUNDS


def test_borrow_with_no_eligible_lender_is_rejected_visibly(setup):
    agents, mp = setup
    force_task(mp)
    for a in agents:
        a.balance = 5
    script(mp, {"Gambler": {"action": "borrow", "amount": 20}})
    orch.run_round(agents, 1)
    assert ev("loan_rejected")[0]["code"] == "no_lender_available"


# ------------------------------------------------------- full-run properties
def full_run(seed, rounds=40, **cfg):
    events.reset_log()
    orch.run_simulation(max_rounds=rounds, round_delay=0, verbose=False, seed=seed, config_overrides=cfg)
    return events.read_all()


def test_event_log_reconciles_with_ledger_independently(isolated_data_dir):
    """Recompute the money supply from EVENTS ONLY and compare with real balances."""
    ev_ = full_run(seed=11, rounds=60)
    ended = [e for e in ev_ if e["type"] == "simulation_ended"][0]
    start = [e for e in ev_ if e["type"] == "simulation_started"][0]
    minted = sum(e["bounty"] for e in ev_ if e["type"] == "task_resolved")
    burned = sum(e["cost"] for e in ev_ if e["type"] == "task_resolved") + \
        sum(e.get("charged", 0) for e in ev_ if e["type"] == "call_error")
    expected = start["starting_balance"] * len(start["agents"]) + minted - burned
    assert sum(ended["final_balances"].values()) == expected
    assert ended["reconciliation"]["ok"]
    assert all(v >= 0 and isinstance(v, int) for v in ended["final_balances"].values())


def test_same_seed_reproduces_exactly_and_different_seed_differs(isolated_data_dir):
    def fingerprint(ev_):
        return [(e["type"], e.get("agent"), e.get("task_id"), e.get("correct"), e.get("new_balance"))
                for e in ev_ if e["type"] not in ("simulation_started",)]
    a, b, c = full_run(5), full_run(5), full_run(6)
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint(c)
    assert [e for e in a if e["type"] == "simulation_started"][0]["seed"] == 5


def test_event_sequence_is_strictly_increasing_and_ids_unique(isolated_data_dir):
    ev_ = full_run(3, rounds=15)
    seqs = [e["seq"] for e in ev_]
    assert seqs == list(range(1, len(seqs) + 1))
    assert len({e["event_id"] for e in ev_}) == len(ev_)
    assert len({e["run_id"] for e in ev_}) == 1


def test_simulation_started_records_config_mode_and_models(isolated_data_dir):
    s = [e for e in full_run(2, rounds=2, interest_rate=0.3) if e["type"] == "simulation_started"][0]
    assert s["mode"] == "mock" and s["config"]["interest_rate"] == 0.3 and "nano" in s["models"]


def test_loans_default_when_borrower_goes_bankrupt_in_a_real_run(isolated_data_dir, monkeypatch):
    events.reset_log()
    agents = fresh_roster(2)
    cfg = SimulationConfig.defaults(num_agents=2)
    from core.economy import Ledger
    ledger = Ledger(agents, cfg)
    loan = ledger.create_loan(agents[0], agents[1], 10, 1)
    agents[1].balance = 0
    orch._check_bankruptcy(agents[1], 2, ledger)
    assert agents[1].bankrupt and loan.status == "defaulted"
    assert any(e["type"] == "loan_default" for e in events.read_all())


def test_leaderboard_row_carries_stats_for_separate_rankings(isolated_data_dir):
    full_run(4, rounds=30)
    row = events.read_leaderboard()[-1]
    assert row["seed"] == 4 and row["mode"] == "mock"
    s = next(iter(row["agent_stats"].values()))
    for k in ("accuracy", "net_pl", "compute_spent", "rewards_earned", "graded_attempts"):
        assert k in s
