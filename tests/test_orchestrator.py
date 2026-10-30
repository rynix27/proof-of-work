from core import config, events
from core.agents import Agent, fresh_roster
from core.orchestrator import (
    _check_bankruptcy,
    _maybe_emit_commentary,
    run_round,
    run_simulation,
)


def test_check_bankruptcy_flags_agent_at_or_below_threshold(isolated_data_dir):
    events.reset_log()
    agent = Agent(name="Hoarder", seed_prompt="You are Hoarder.", balance=0)
    _check_bankruptcy(agent, round_num=5)
    assert agent.bankrupt is True


def test_check_bankruptcy_does_not_flag_solvent_agent(isolated_data_dir):
    events.reset_log()
    agent = Agent(name="Hoarder", seed_prompt="You are Hoarder.", balance=10)
    _check_bankruptcy(agent, round_num=5)
    assert agent.bankrupt is False


def test_bankruptcy_emits_event_with_epitaph(isolated_data_dir):
    events.reset_log()
    agent = Agent(name="Hoarder", seed_prompt="You are Hoarder.", balance=0)
    _check_bankruptcy(agent, round_num=5)

    bankruptcy_events = [e for e in events.read_all() if e["type"] == "bankruptcy"]
    assert len(bankruptcy_events) == 1
    assert bankruptcy_events[0]["agent"] == "Hoarder"
    assert bankruptcy_events[0]["epitaph"]  # fake mode still produces a canned epitaph


def test_bankruptcy_check_is_idempotent(isolated_data_dir):
    """Calling _check_bankruptcy again on an already-bankrupt agent
    shouldn't emit a second bankruptcy event (and a second epitaph call)."""
    events.reset_log()
    agent = Agent(name="Hoarder", seed_prompt="You are Hoarder.", balance=0)
    _check_bankruptcy(agent, round_num=5)
    _check_bankruptcy(agent, round_num=6)

    bankruptcy_events = [e for e in events.read_all() if e["type"] == "bankruptcy"]
    assert len(bankruptcy_events) == 1


def test_commentary_fires_on_configured_interval(isolated_data_dir, monkeypatch):
    monkeypatch.setattr(config, "COMMENTARY_INTERVAL_ROUNDS", 3)
    events.reset_log()
    agents = fresh_roster()

    for round_num in range(1, 8):
        _maybe_emit_commentary(agents, round_num)

    commentary_rounds = [e["round"] for e in events.read_all() if e["type"] == "commentary"]
    assert commentary_rounds == [3, 6]


def test_run_round_resolves_a_task_and_updates_balance(isolated_data_dir):
    events.reset_log()
    agents = fresh_roster()
    run_round(agents, round_num=1)

    all_events = events.read_all()
    assert any(e["type"] == "task_arrived" for e in all_events)
    # in fake mode, either someone bid and it resolved, or nobody bid --
    # both are valid outcomes, but the round shouldn't crash either way
    assert any(e["type"] in ("task_resolved", "task_unclaimed") for e in all_events)


def test_run_round_survives_every_agent_decision_failing(isolated_data_dir, monkeypatch):
    """If every single agent's decision call fails, the round should
    complete gracefully (task goes unclaimed) rather than raising."""
    import core.orchestrator as orch

    def always_fail(*args, **kwargs):
        raise RuntimeError("simulated total API outage")

    monkeypatch.setattr(orch, "decide_action", always_fail)

    events.reset_log()
    agents = fresh_roster()
    run_round(agents, round_num=1)  # should not raise

    all_events = events.read_all()
    error_events = [e for e in all_events if e["type"] == "call_error"]
    assert len(error_events) == len(agents)
    assert any(e["type"] == "task_unclaimed" for e in all_events)


def test_run_round_survives_solve_failure_after_successful_bid(isolated_data_dir, monkeypatch):
    """The winning bidder's solve call fails -- the round should still
    resolve gracefully as unclaimed rather than crashing."""
    import core.orchestrator as orch

    def fake_decide(agent, task, round_num):
        return {"action": "bid", "confidence": 0.9}

    def fake_solve(*args, **kwargs):
        raise RuntimeError("simulated solve failure")

    monkeypatch.setattr(orch, "decide_action", fake_decide)
    monkeypatch.setattr(orch, "solve_task", fake_solve)

    events.reset_log()
    agents = fresh_roster()
    run_round(agents, round_num=1)  # should not raise

    all_events = events.read_all()
    assert any(e["type"] == "call_error" and e["purpose"] == "solve" for e in all_events)
    assert any(e["type"] == "task_unclaimed" for e in all_events)


def test_run_simulation_survives_unexpected_error_in_run_round(isolated_data_dir, monkeypatch):
    """Outer safety net: even a totally unexpected exception inside
    run_round shouldn't kill the whole background simulation thread."""
    import core.orchestrator as orch

    call_count = {"n": 0}
    real_run_round = orch.run_round

    def flaky_run_round(agents, round_num):
        call_count["n"] += 1
        if round_num == 2:
            raise RuntimeError("totally unexpected bug")
        return real_run_round(agents, round_num)

    monkeypatch.setattr(orch, "run_round", flaky_run_round)

    run_simulation(max_rounds=4, round_delay=0, verbose=False)

    all_events = events.read_all()
    assert any(e["type"] == "round_error" and e["round"] == 2 for e in all_events)
    assert any(e["type"] == "simulation_ended" for e in all_events)
    # rounds 1, 3, 4 should still have run normally despite round 2 failing
    assert call_count["n"] == 4


def test_run_simulation_ends_and_records_leaderboard_entry(isolated_data_dir):
    run_simulation(max_rounds=5, round_delay=0, verbose=False)
    leaderboard = events.read_leaderboard()
    assert len(leaderboard) == 1
    assert leaderboard[0]["winner"] in [a.name for a in fresh_roster()]


def test_run_simulation_winner_is_highest_balance_not_first_solvent(isolated_data_dir, monkeypatch):
    """Regression test for a real bug: the winner used to be 'first
    solvent agent in list order', which is wrong when several agents
    survive to max_rounds. It must be whoever has the highest balance."""
    import core.orchestrator as orch

    def fake_run_round(agents, round_num):
        # deterministically make the second agent richest
        agents[1].balance += 50

    monkeypatch.setattr(orch, "run_round", fake_run_round)

    run_simulation(max_rounds=3, round_delay=0, verbose=False)

    leaderboard = events.read_leaderboard()
    agents = fresh_roster()
    expected_winner = agents[1].name
    assert leaderboard[0]["winner"] == expected_winner


def test_run_simulation_auto_saves_replay_on_completion(isolated_data_dir):
    run_simulation(max_rounds=5, round_delay=0, verbose=False)
    replays = events.list_replays()
    assert len(replays) == 1
