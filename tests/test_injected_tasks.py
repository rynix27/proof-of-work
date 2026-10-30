from core import tasks
from core.injected_tasks import InjectedTaskQueue
from core.orchestrator import run_round


def test_make_custom_task_with_answer_is_gradable():
    task = tasks.make_custom_task("What is 2+2?", answer="4")
    assert task.answer == "4"
    assert task.bounty == tasks.CUSTOM_TASK_BOUNTY
    assert task.injected is True
    assert tasks.grade(task, "4") is True
    assert tasks.grade(task, "5") is False


def test_make_custom_task_without_answer_is_exhibition():
    task = tasks.make_custom_task("What's your favorite color?")
    assert task.answer is None
    assert task.bounty == 0
    assert tasks.grade(task, "anything at all") is None


def test_custom_task_prompt_is_truncated_and_stripped():
    long_prompt = "  " + ("a" * 1000) + "  "
    task = tasks.make_custom_task(long_prompt)
    assert len(task.prompt) <= 500
    assert not task.prompt.startswith(" ")


def test_queue_pop_returns_none_when_empty():
    q = InjectedTaskQueue()
    assert q.pop_next() is None


def test_queue_is_fifo():
    q = InjectedTaskQueue()
    t1 = tasks.make_custom_task("first")
    t2 = tasks.make_custom_task("second")
    q.push(t1)
    q.push(t2)
    assert q.pop_next().prompt == "first"
    assert q.pop_next().prompt == "second"
    assert q.pop_next() is None


def test_queue_respects_max_pending():
    q = InjectedTaskQueue(max_pending=2)
    for i in range(5):
        q.push(tasks.make_custom_task(f"q{i}"))
    assert q.pending_count() == 2
    # oldest ones should have been dropped, newest kept
    assert q.pop_next().prompt == "q3"
    assert q.pop_next().prompt == "q4"


def test_run_round_picks_up_injected_task_before_random_one(isolated_data_dir, monkeypatch):
    import core.orchestrator as orch
    from core import events
    from core.agents import fresh_roster

    events.reset_log()
    custom = tasks.make_custom_task("What is the injected question?", answer="yes")
    orch.injected_tasks.push(custom)

    agents = fresh_roster()
    run_round(agents, round_num=1)

    arrived = [e for e in events.read_all() if e["type"] == "task_arrived"][0]
    assert arrived["prompt"] == "What is the injected question?"
    assert arrived["injected"] is True


def test_exhibition_task_never_changes_balances(isolated_data_dir, monkeypatch):
    """An ungraded injected task must have zero economic effect beyond the
    ordinary compute cost of answering it -- no bounty, ever, regardless
    of what the agent answers."""
    import core.orchestrator as orch
    from core import events
    from core.agents import fresh_roster

    def fake_decide(agent, task, round_num):
        return {"action": "bid", "confidence": 0.9}

    def fake_solve(agent, task, escalate, round_num):
        # NOTE: solve_task no longer charges compute itself -- the Ledger does it
        # in run_round (single place where balances change). The fake therefore
        # must not deduct anything.
        return {"answer": "whatever, this can't be graded", "confidence": 0.9}

    monkeypatch.setattr(orch, "decide_action", fake_decide)
    monkeypatch.setattr(orch, "solve_task", fake_solve)

    events.reset_log()
    custom = tasks.make_custom_task("What's the meaning of life?")  # no answer -> exhibition
    orch.injected_tasks.push(custom)

    agents = fresh_roster()
    starting_balance = agents[0].balance
    run_round(agents, round_num=1)

    resolved = [e for e in events.read_all() if e["type"] == "task_resolved"][0]
    assert resolved["correct"] is None
    assert resolved["exhibition"] is True
    assert resolved["bounty"] == 0
    # exactly one Nano solve (cost 1) was charged, and no bounty was added back
    from core import config
    changed = [a for a in agents if a.balance != starting_balance]
    assert len(changed) == 1
    assert changed[0].balance == starting_balance - config.NANO_COST


def test_graded_injected_task_pays_bounty_on_correct_answer(isolated_data_dir, monkeypatch):
    import core.orchestrator as orch
    from core import events
    from core.agents import fresh_roster

    def fake_decide(agent, task, round_num):
        return {"action": "bid", "confidence": 0.9}

    def fake_solve(agent, task, escalate, round_num):
        return {"answer": "4", "confidence": 0.9}

    monkeypatch.setattr(orch, "decide_action", fake_decide)
    monkeypatch.setattr(orch, "solve_task", fake_solve)

    events.reset_log()
    custom = tasks.make_custom_task("What is 2+2?", answer="4")
    orch.injected_tasks.push(custom)

    agents = fresh_roster()
    run_round(agents, round_num=1)

    resolved = [e for e in events.read_all() if e["type"] == "task_resolved"][0]
    assert resolved["correct"] is True
    assert resolved["bounty"] == tasks.CUSTOM_TASK_BOUNTY
