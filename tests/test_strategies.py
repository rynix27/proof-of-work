"""Phase 5: strategy policies, EV maths, risk limits, escalation rules, explainability, no-omniscience."""
import itertools
from dataclasses import replace

import pytest

import core.orchestrator as orch
from core import config, events, tasks
from core.agents import fresh_roster, record_result
from core.simconfig import ConfigError, SimulationConfig
from core.strategies import (BASELINE, ROSTER_STRATEGY, STRATEGIES, decide_escalation, estimate_success,
                             estimate_ultra_success, evaluate_decision, resolve_strategies)
from core.validation import Decision, validate_decision

CFG = SimulationConfig.defaults()


def task(difficulty="easy", bounty=8, answer="SECRET-ANSWER"):
    return tasks.Task(id=1, prompt="q", answer=answer, difficulty=difficulty, bounty=bounty)


def bid(conf=0.9, **kw):
    return Decision("bid", 0, conf, "because", **kw)


def agent(balance=100, **kw):
    a = fresh_roster()[0]
    a.balance = balance
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def ev_(policy, decision, t=None, balance=100, debt=0, others=None, a=None):
    pol = STRATEGIES[policy] if isinstance(policy, str) else policy
    return evaluate_decision(a or agent(balance), decision, t or task(), pol,
                             balance=balance, start=100, debt=debt, cfg=CFG, others=others)


# ------------------------------------------------------------ probability / EV
def test_probability_estimate_formula_is_beta_smoothed_history_blended_with_confidence():
    a = agent()
    a.by_difficulty["easy"] = [8, 6]                       # 6/8 correct on easy
    pol = STRATEGIES["conservative"]                       # prior .5, w 4, trust 0
    assert estimate_success(a, pol, "easy", 0.99) == round((6 + 0.5 * 4) / (8 + 4), 4)       # confidence ignored
    pol = STRATEGIES["aggressive"]                         # prior .75, w 2, trust .5
    p_hist = (6 + 0.75 * 2) / (8 + 2)
    assert estimate_success(a, pol, "easy", 0.4) == round(0.5 * p_hist + 0.5 * 0.4, 4)
    assert estimate_success(a, pol, "hard", None) == 0.75   # no history -> the prior, nothing invented


def test_history_pulls_the_estimate_toward_observed_results():
    a, pol = agent(), STRATEGIES["cost_efficient"]
    before = estimate_success(a, pol, "easy", None)
    for _ in range(20):
        record_result(a, "easy", "nano", False, 1)
    assert estimate_success(a, pol, "easy", None) < before - 0.3
    for _ in range(60):
        record_result(a, "easy", "nano", True, 1)
    assert estimate_success(a, pol, "easy", None) > 0.7


def test_ev_equals_p_times_reward_minus_cost_and_is_reported():
    out = ev_("cost_efficient", bid(0.9), task(bounty=20), balance=100)
    p = estimate_success(agent(), STRATEGIES["cost_efficient"], "easy", 0.9)
    assert out.p_est == p and out.reward == 20 and out.cost == CFG.nano_cost
    assert out.ev == round(p * 20 - CFG.nano_cost, 3)


def test_ev_boundary_is_inclusive():
    pol = replace(STRATEGIES["cost_efficient"], min_p=0, max_bid_fraction=1, min_reserve_fraction=0)
    p = estimate_success(agent(), pol, "easy", 0.9)
    exact = round(p * 8 - 1, 3)
    assert ev_(replace(pol, min_ev=exact), bid(0.9)).decision.action == "bid"
    assert ev_(replace(pol, min_ev=exact + 0.01), bid(0.9)).decision.action == "skip"


# ------------------------------------------------------------------- bidding
def test_baseline_never_overrides_the_model():
    for d in (bid(0.01), bid(0.99), Decision("skip", 0, None, ""), Decision("lend", 500, None, ""),
              Decision("borrow", 999, None, "")):
        out = ev_("baseline", d)
        assert out.decision == d and not out.reasons and not out.overridden


def test_unaffordable_bid_is_left_for_the_auction_to_reject():
    out = ev_("conservative", bid(0.99), balance=0, a=agent(0))
    assert out.decision.action == "bid" and "cost_over_budget_fraction" not in out.reasons


def test_low_ev_bid_is_turned_into_a_skip_with_the_reason_recorded():
    out = ev_("conservative", bid(0.9), task(bounty=2))          # tiny bounty: EV < min_ev
    assert out.decision.action == "skip" and out.model_action == "bid"
    assert "ev_below_min" in out.reasons and out.overridden


def test_p_below_min_blocks_accuracy_focused_on_unproven_hard_tasks():
    a = agent()
    for _ in range(10):
        record_result(a, "hard", "nano", False, 1)
    out = ev_("accuracy_focused", bid(0.9), task("hard", 50), a=a)
    assert out.decision.action == "skip" and "p_below_min" in out.reasons


def test_budget_fraction_and_reserve_limits():
    out = ev_("conservative", bid(), task(bounty=50), balance=10)
    assert "cost_over_budget_fraction" not in out.reasons       # cost 1 is exactly 10% -> allowed ...
    assert "reserve_breach" in out.reasons                      # ... but 9 left breaches the 20% reserve
    assert "cost_over_budget_fraction" in ev_("conservative", bid(), task(bounty=50), balance=9).reasons


def test_aggressive_accepts_marginal_negative_ev_conservative_does_not():
    t = task(bounty=2)
    assert ev_("aggressive", bid(0.9), t).decision.action == "bid"
    assert ev_("conservative", bid(0.9), t).decision.action == "skip"


# ------------------------------------------------------------- lend / borrow
def test_lend_is_clamped_to_cap_and_reserve_and_can_be_disabled():
    out = ev_("conservative", Decision("lend", 40, None, ""), balance=100)
    assert out.decision.action == "lend" and out.decision.amount == 15 and "lend_clamped" in out.reasons
    out = ev_("conservative", Decision("lend", 10, None, ""), balance=100)
    assert out.decision.amount == 10 and not out.reasons
    assert ev_("conservative", Decision("lend", 10, None, ""), balance=62).decision.amount == 2   # keeps 60%
    out = ev_("conservative", Decision("lend", 10, None, ""), balance=60)
    assert out.decision.action == "skip" and "lend_reserve" in out.reasons
    out = ev_("aggressive", Decision("lend", 10, None, ""))
    assert out.decision.action == "skip" and "lend_disabled" in out.reasons


def test_borrow_requires_need_and_respects_the_debt_limit():
    d = Decision("borrow", 10, None, "")
    assert "not_needy" in ev_("conservative", d, balance=80).reasons
    assert ev_("conservative", d, balance=40).decision.action == "borrow"
    out = ev_("conservative", Decision("borrow", 25, None, ""), balance=40, debt=10)    # 35 > 30% of 100
    assert out.decision.action == "skip" and "debt_limit" in out.reasons
    assert ev_("conservative", d, balance=40, debt=20).decision.action == "borrow"      # exactly 30 -> allowed


def test_proactive_lender_offers_surplus_to_a_struggling_agent_only():
    rich, poor, ok = fresh_roster()[:3]
    rich.balance, poor.balance, ok.balance = 160, 40, 100
    skip = Decision("skip", 0, None, "")
    out = ev_("lender", skip, balance=160, others=[poor, ok], a=rich)
    assert out.decision.action == "lend" and out.reasons == ("proactive_lend",)
    assert out.decision.amount == min(int(0.35 * 160), 160 - 30, CFG.max_loan_amount)
    assert ev_("lender", skip, balance=160, others=[ok], a=rich).decision.action == "skip"     # nobody needy
    assert ev_("lender", skip, balance=100, others=[poor], a=rich).decision.action == "skip"   # no surplus
    assert ev_("conservative", skip, balance=160, others=[poor], a=rich).decision.action == "skip"


# ---------------------------------------------------------------- escalation
def esc(policy, conf, balance=100, **kw):
    a = agent(balance, **kw)
    return decide_escalation(a, STRATEGIES[policy], conf, task(bounty=50), balance=balance, start=100, cfg=CFG)


def test_escalation_cutoff_is_strategy_specific():
    assert esc("accuracy_focused", 0.8)[0] is True        # cutoff .85
    assert esc("cost_efficient", 0.5)[0] is False         # cutoff .3 -> not even considered
    assert esc("cost_efficient", 0.2)[0] is True
    assert esc("baseline", 0.59)[0] is True and esc("baseline", 0.6)[0] is False   # config cutoff .6, strict <


def test_escalation_needs_affordability_budget_and_positive_ev():
    assert esc("aggressive", 0.1, balance=CFG.ultra_cost - 1) == (False, ("cannot_afford_ultra",))
    assert esc("conservative", 0.1, ultra_spent=10)[1] == ("escalation_budget_exhausted",)     # budget = 10% of 100
    assert esc("conservative", 0.1) == (True, ())
    a = agent(100)
    a.ultra_record = [10, 0]                               # Ultra has been failing for this agent
    ok, reasons = decide_escalation(a, STRATEGIES["conservative"], 0.1, task(bounty=8), balance=100,
                                    start=100, cfg=CFG)
    assert ok is False and "ultra_ev_negative" in reasons


def test_ultra_ev_estimate_uses_the_agents_own_observed_record():
    a = agent()
    base = estimate_ultra_success(a, STRATEGIES["cost_efficient"])
    a.ultra_record = [10, 10]
    assert estimate_ultra_success(a, STRATEGIES["cost_efficient"]) > base


# ----------------------------------------------------------------- resolving
def test_resolve_strategies_and_roster_mapping():
    names = [a.name for a in fresh_roster()]
    r = resolve_strategies("roster", names)
    assert {n: p.name for n, p in r.items()} == ROSTER_STRATEGY
    assert all(p is BASELINE for p in resolve_strategies(None, names).values())
    assert all(p.name == "lender" for p in resolve_strategies("lender", names).values())
    assert resolve_strategies(["lender", "aggressive"], names[:2])[names[1]].name == "aggressive"
    for bad in ("nope", ["lender"], ["x"] * 5, 5, [1, 2, 3, 4, 5]):
        with pytest.raises(ValueError):
            resolve_strategies(bad, names)


@pytest.mark.parametrize("bad", ["nope", ["lender", "aggressive"], {"a": 1}])
def test_simconfig_validates_strategies(bad):
    with pytest.raises(ConfigError):
        SimulationConfig.from_overrides({"strategies": bad, "num_agents": 5})


def test_simconfig_bounties_validated_and_applied(isolated_data_dir):
    for bad in ([1, 2], [1, 2, "x"], [-1, 2, 3], [1, 2, 5000], "abc"):
        with pytest.raises(ConfigError):
            SimulationConfig.from_overrides({"bounties": bad})
    events.reset_log()
    orch.run_simulation(max_rounds=12, round_delay=0, verbose=False, seed=1, config_overrides={"bounties": [1, 2, 3]})
    assert {e["bounty"] for e in events.read_all() if e["type"] == "task_arrived"} <= {1, 2, 3}


# -------------------------------------------------- validation of model extras
def test_optional_model_estimates_are_validated_and_invalid_ones_dropped_not_fatal():
    d = validate_decision({"action": "bid", "confidence": .7, "estimated_success_probability": .6,
                           "estimated_difficulty": "Hard"})
    assert (d.est_probability, d.est_difficulty, d.warnings) == (.6, "hard", ())
    d = validate_decision({"action": "bid", "confidence": .7, "estimated_success_probability": float("nan"),
                           "estimated_difficulty": "impossible"})
    assert d.action == "bid" and d.est_probability is None and d.est_difficulty is None
    assert set(d.warnings) == {"bad_estimated_success_probability", "bad_estimated_difficulty"}


# ------------------------------------------------------------ explainability
def test_every_decision_event_explains_itself_with_numbers_and_labels_model_text(isolated_data_dir, monkeypatch):
    events.reset_log()
    agents = fresh_roster()
    monkeypatch.setattr(orch.tasks, "generate_task", lambda rng=None, weights=None: task(bounty=2))
    monkeypatch.setattr(orch, "decide_action", lambda a, t, r: {"action": "bid", "confidence": .9,
                                                                 "reason": "trust me, it is easy"})
    monkeypatch.setattr(orch, "solve_task", lambda *a, **k: {"answer": "x"})
    monkeypatch.setattr(config, "STRATEGY_SET", "roster")
    orch.run_round(agents, 1)
    decs = {e["agent"]: e for e in events.read_all() if e["type"] == "decision"}
    cons = decs["Hoarder"]                                           # conservative: tiny bounty -> policy skip
    assert cons["action"] == "skip" and cons["model_action"] == "bid" and "ev_below_min" in cons["policy_reasons"]
    assert "EV=" in cons["explanation"] and cons["strategy"] == "conservative"
    assert cons["model_explanation"] == "trust me, it is easy"       # the model's words, kept separate from ours
    assert isinstance(cons["p_est"], float) and isinstance(cons["ev"], float) and cons["est_cost"] == 1
    assert decs["Gambler"]["action"] == "bid" and not decs["Gambler"]["policy_reasons"]


def test_escalation_denial_is_recorded_with_reasons(isolated_data_dir, monkeypatch):
    events.reset_log()
    agents = fresh_roster()
    monkeypatch.setattr(config, "STRATEGY_SET", "conservative")
    monkeypatch.setattr(orch.tasks, "generate_task", lambda rng=None, weights=None: task(bounty=50))
    monkeypatch.setattr(orch, "decide_action", lambda a, t, r: {"action": "bid", "confidence": 0.1 if a.name == "Gambler" else 0.0})
    monkeypatch.setattr(orch, "solve_task", lambda *a, **k: {"answer": "x"})
    for a in agents:
        a.ultra_spent = 10                                           # budget already used
    orch.run_round(agents, 1)
    den = [e for e in events.read_all() if e["type"] == "escalation_denied"]
    assert den and den[0]["reasons"] == ["escalation_budget_exhausted"]
    assert not [e for e in events.read_all() if e["type"] == "escalation"]


# ---------------------------------------------------------- no omniscience
def test_agent_prompts_never_contain_the_tasks_answer(isolated_data_dir, monkeypatch):
    seen = []
    monkeypatch.setattr(orch, "call_model_json", lambda prompt, tier, system="", meta=None, model=None:
                        seen.append((prompt, system)) or {"action": "skip"})
    t = task(answer="ZXQ-TOP-SECRET-ANSWER")
    a = fresh_roster()[0]
    orch.decide_action(a, t, 1)
    orch.solve_task(a, t, escalate=False, round_num=1)
    assert seen and all("ZXQ-TOP-SECRET-ANSWER" not in p and "ZXQ-TOP-SECRET-ANSWER" not in s for p, s in seen)


def test_decision_prompt_includes_only_the_agents_own_record(isolated_data_dir, monkeypatch):
    seen = []
    monkeypatch.setattr(orch, "call_model_json", lambda prompt, tier, system="", meta=None, model=None:
                        seen.append(prompt) or {"action": "skip"})
    a, b = fresh_roster()[:2]
    record_result(a, "easy", "nano", True, 1)
    record_result(b, "hard", "nano", False, 1)
    orch.decide_action(a, task(), 1)
    assert "easy: 1/1 correct" in seen[0] and "hard" not in seen[0]


# --------------------------------------------------- integration / diversity
_RUN_CACHE = {}


def run(strategies, seed=11, rounds=60):
    """Simulate once per (strategies, seed, rounds) and share the (read-only) result between tests:
    these runs are deterministic, so re-simulating them only costs time."""
    key = (strategies if isinstance(strategies, str) else tuple(strategies), seed, rounds)
    if key not in _RUN_CACHE:
        events.reset_log()
        orch.run_simulation(max_rounds=rounds, round_delay=0, verbose=False, seed=seed,
                            config_overrides={"strategies": strategies})
        ev = events.read_all()
        end = [e for e in ev if e["type"] == "simulation_ended"][0]
        _RUN_CACHE[key] = (ev, end, [e for e in ev if e["type"] == "decision"])
    return _RUN_CACHE[key]


def fingerprint(s):
    ev, end, dec = run(s)
    a = end["agent_stats"].values()
    return dict(bids=sum(1 for d in dec if d["action"] == "bid"), compute=sum(x["compute_spent"] for x in a),
                esc=sum(x["escalations"] for x in a))


def test_the_five_strategies_produce_pairwise_distinct_behaviour(isolated_data_dir):
    fp = {s: tuple(fingerprint(s).values())
          for s in ("conservative", "aggressive", "cost_efficient", "accuracy_focused", "lender")}
    assert all(fp[a] != fp[b] for a, b in itertools.combinations(fp, 2)), fp


def test_strategy_risk_profiles_order_as_designed(isolated_data_dir):
    cons, aggr, cost, acc = (fingerprint(s) for s in ("conservative", "aggressive", "cost_efficient", "accuracy_focused"))
    assert cons["compute"] < aggr["compute"] and cost["compute"] < acc["compute"]
    assert cons["bids"] < aggr["bids"]
    assert acc["esc"] > cons["esc"] and aggr["esc"] > cons["esc"]       # readily-escalating vs rarely-escalating


def test_no_agent_ever_exceeds_its_escalation_budget(isolated_data_dir):
    """Invariant, not a magic number: ultra spend <= budget_fraction * starting balance."""
    for name, pol in STRATEGIES.items():
        if name == "baseline" or pol.escalation_budget_fraction == float("inf"):
            continue
        for seed in (1, 2):
            _, end, _ = run(name, seed=seed, rounds=40)
            for who, st in end["agent_stats"].items():
                assert st["ultra_spent"] <= pol.escalation_budget_fraction * 100, (name, seed, who, st["ultra_spent"])


def test_every_strategy_is_able_to_escalate_at_least_once_within_its_budget():
    """Regression: budgets once sat below the cost of a single Ultra call, silently disabling escalation."""
    for name, pol in STRATEGIES.items():
        assert pol.escalation_budget_fraction * 100 >= CFG.ultra_cost, name


def test_mixed_roster_has_a_working_loan_market_and_the_books_balance(isolated_data_dir):
    ev, end, dec = run("roster", seed=3)
    assert end["reconciliation"]["ok"] and end["loans"]["loans_made"] >= 1
    stats = end["agent_stats"]
    assert {s["strategy"] for s in stats.values()} == set(ROSTER_STRATEGY.values())
    assert stats["Gambler"]["loans_issued"] == 0                    # aggressive cannot lend
    assert stats["LoanShark"]["loans_issued"] >= 1                  # the lender does
    started = [e for e in ev if e["type"] == "simulation_started"][0]
    assert started["strategies"]["Hoarder"] == "conservative"


def test_a_borrower_cannot_pick_a_lender_whose_own_policy_forbids_lending(isolated_data_dir, monkeypatch):
    events.reset_log()
    agents = fresh_roster()
    rich = {"Gambler": 400}                       # aggressive: can_lend=False, and the richest by far
    for a in agents:
        a.balance = rich.get(a.name, 100)
    monkeypatch.setattr(config, "STRATEGY_SET", "roster")
    monkeypatch.setattr(orch.tasks, "generate_task", lambda rng=None, weights=None: task())
    borrower = next(a for a in agents if a.name == "Hoarder")
    borrower.balance = 20                          # conservative borrows when below 50
    monkeypatch.setattr(orch, "decide_action", lambda a, t, r: {"action": "borrow", "amount": 10}
                        if a.name == "Hoarder" else {"action": "skip"})
    orch.run_round(agents, 1)
    made = [e for e in events.read_all() if e["type"] == "loan_made"]
    assert made and made[0]["lender"] != "Gambler"


def test_policy_never_lets_a_balance_go_negative_or_corrupt_the_ledger(isolated_data_dir):
    for s in ("roster", "aggressive", "lender", "accuracy_focused"):
        for seed in (1, 2):
            ev, end, _ = run(s, seed=seed, rounds=40)
            assert end["reconciliation"]["ok"]
            assert all(v >= 0 for v in end["final_balances"].values())


def test_baseline_matches_the_pre_strategy_behaviour(isolated_data_dir):
    """The control group: baseline decisions are exactly the model's, nothing overridden."""
    ev, end, dec = run("baseline")
    assert all(not d["policy_reasons"] and d["action"] == d["model_action"] for d in dec)
    assert not [e for e in ev if e["type"] == "escalation_denied"]
