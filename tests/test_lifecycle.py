"""Run lifecycle, stale-thread isolation, concurrency, and disk failure handling."""
import json
import threading
import time

import pytest

import core.orchestrator as orch
from core import config, events
from core.control import control
from core.runner import manager


def test_stale_thread_cannot_write_into_a_new_run(isolated_data_dir):
    old = events.reset_log()
    events.bind_run(old)
    assert events.emit("from_old_run_1") is not None
    new = events.reset_log()                 # a reset happens while the old thread is alive
    assert new != old
    assert events.is_stale()
    assert events.emit("from_old_run_2") is None            # dropped
    assert events.save_replay("x") is None                  # dropped
    events.record_leaderboard_entry("A", {"A": 1}, 1, "x")  # dropped
    assert events.read_leaderboard() == []
    events.bind_run(None)                    # the new run's own (unbound/owning) thread
    events.emit("from_new_run")
    assert [e["type"] for e in events.read_all()] == ["from_new_run"]
    assert events.read_all()[0]["seq"] == 1 and events.read_all()[0]["run_id"] == new


def test_reset_log_truncates_and_restarts_sequence(isolated_data_dir):
    events.reset_log()
    for _ in range(3):
        events.emit("x")
    events.reset_log()
    assert events.read_all() == []
    assert events.emit("y")["seq"] == 1


def test_concurrent_emits_produce_only_valid_unique_lines(isolated_data_dir):
    events.reset_log()
    N, T = 200, 8

    def writer(i):
        for j in range(N):
            events.emit("w", thread=i, j=j, pad="x" * 200)
    reader_errors = []
    stop = threading.Event()

    def reader():
        while not stop.is_set():
            try:
                for e in events.read_all():
                    assert "type" in e
            except Exception as ex:  # noqa: BLE001
                reader_errors.append(ex)

    r = threading.Thread(target=reader)
    r.start()
    ts = [threading.Thread(target=writer, args=(i,)) for i in range(T)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    stop.set()
    r.join()
    raw = events.LOG_PATH.read_text().splitlines()
    parsed = [json.loads(line) for line in raw]            # every line is complete JSON
    assert len(parsed) == N * T
    assert sorted(e["seq"] for e in parsed) == list(range(1, N * T + 1))
    assert not reader_errors


def test_reader_tolerates_corrupt_and_partial_lines(isolated_data_dir):
    events.reset_log()
    events.emit("good")
    with open(events.LOG_PATH, "a") as f:
        f.write('{"type": "trunc\n')            # partial/corrupt
        f.write("not json at all\n")
        f.write("[1,2]\n")                       # valid JSON, not an object
        f.write("\n")
    events.emit("good2")
    assert [e["type"] for e in events.read_all()] == ["good", "good2"]


def test_disk_write_failure_does_not_raise_and_is_counted(isolated_data_dir, monkeypatch):
    events.reset_log()
    monkeypatch.setattr(events, "LOG_PATH", isolated_data_dir / "no_such_dir" / "deeper" / "e.jsonl")
    import os
    real_open = open

    def failing_open(path, mode="r", *a, **k):
        if str(path).endswith("e.jsonl") and "a" in mode:
            raise OSError("disk full")
        return real_open(path, mode, *a, **k)
    monkeypatch.setattr("builtins.open", failing_open)
    before = events.write_failures
    e = events.emit("still_ok", x=1)                 # must not raise
    assert e["type"] == "still_ok" and events.write_failures == before + 1


def test_replay_save_is_atomic_and_never_clobbers(isolated_data_dir):
    events.reset_log()
    events.emit("a")
    n1 = events.save_replay("same")
    n2 = events.save_replay("same")
    assert n1 != n2 and (events.REPLAY_DIR / n1).exists() and (events.REPLAY_DIR / n2).exists()
    assert not list(events.REPLAY_DIR.glob("*.tmp"))


@pytest.mark.parametrize("bad", ["../x.jsonl", "..\\x.jsonl", "/etc/passwd", "a/b.jsonl", "x.txt", "", "x.jsonl\n",
                                 ".hidden/../../x.jsonl", None, 5, "a" * 300 + ".jsonl"])
def test_replay_names_are_validated(isolated_data_dir, bad):
    with pytest.raises(ValueError):
        events.replay_path(bad)


def test_save_replay_label_is_sanitized(isolated_data_dir):
    events.reset_log()
    name = events.save_replay("../../evil label<script>")
    assert "/" not in name and "<" not in name and ".." not in name
    assert (events.REPLAY_DIR / name).resolve().parent == events.REPLAY_DIR.resolve()


# ------------------------------------------------------------------- manager
def wait_done(timeout=10):
    t0 = time.time()
    while manager.is_running() and time.time() - t0 < timeout:
        time.sleep(0.02)
    return not manager.is_running()


def test_manager_runs_to_completion_and_records_run_id(isolated_data_dir):
    info = manager.start({"max_rounds": 4, "seed": 1})
    control.round_delay = 0
    assert wait_done()
    evs = events.read_all()
    assert {e["run_id"] for e in evs} == {info["run_id"]}
    assert evs[-1]["type"] in ("simulation_ended",) or any(e["type"] == "simulation_ended" for e in evs)


def test_rapid_resets_leave_exactly_one_coherent_run(isolated_data_dir):
    control.round_delay = 0.05
    infos = [manager.start({"max_rounds": 200, "seed": i}) for i in range(12)]   # 12 back-to-back resets
    time.sleep(0.4)
    live = [t for t in threading.enumerate() if t.name.startswith("simulation-") and t.is_alive()]
    assert len(live) <= 2                                   # at most the current + one finishing its last step
    evs = events.read_all()
    last = infos[-1]["run_id"]
    assert {e["run_id"] for e in evs} == {last}             # nothing from the 11 superseded runs
    assert [e["seq"] for e in evs] == list(range(1, len(evs) + 1))
    assert sum(1 for e in evs if e["type"] == "simulation_started") == 1
    assert manager.shutdown(timeout=5)
    assert wait_done()


def test_reset_during_slow_inflight_call_discards_old_runs_output(isolated_data_dir, monkeypatch):
    """The old thread is stuck inside a slow model call when reset happens."""
    gate = threading.Event()
    real_decide = orch.decide_action

    def slow_decide(agent, task, rn):
        gate.wait(5)                                         # simulates a hung API call
        return {"action": "bid", "confidence": 0.9}
    monkeypatch.setattr(orch, "decide_action", slow_decide)
    control.round_delay = 0
    first = manager.start({"max_rounds": 50, "seed": 1}, join_timeout=0.2)   # bounded join, doesn't hang
    time.sleep(0.2)
    monkeypatch.setattr(orch, "decide_action", real_decide)
    t0 = time.time()
    second = manager.start({"max_rounds": 3, "seed": 2}, join_timeout=0.2)
    assert time.time() - t0 < 2                              # reset is not blocked by the stuck thread
    gate.set()                                               # old call finally returns
    assert wait_done()
    time.sleep(0.2)
    evs = events.read_all()
    assert {e["run_id"] for e in evs} == {second["run_id"]}
    assert first["run_id"] != second["run_id"]


def test_pause_resume_and_stop_are_responsive(isolated_data_dir):
    control.round_delay = 0.1
    manager.start({"max_rounds": 500, "seed": 3})
    time.sleep(0.3)
    control.pause()
    time.sleep(0.3)
    n = len(events.read_all())
    time.sleep(0.4)
    assert len(events.read_all()) == n                       # no progress while paused
    control.resume()
    time.sleep(0.4)
    assert len(events.read_all()) > n
    t0 = time.time()
    assert manager.shutdown(timeout=5)
    assert time.time() - t0 < 3
    assert any(e["type"] == "simulation_ended" and e["reason"] == "stopped" for e in events.read_all())


def test_bad_config_is_rejected_before_any_state_changes(isolated_data_dir):
    from core.simconfig import ConfigError
    events.reset_log()
    events.emit("marker")
    with pytest.raises(ConfigError):
        manager.start({"num_agents": 99})
    assert [e["type"] for e in events.read_all()] == ["marker"]       # log untouched
    assert not manager.is_running()


def test_round_exception_does_not_stop_the_run(isolated_data_dir, monkeypatch):
    calls = {"n": 0}
    real = orch.run_round

    def flaky(agents, rn):
        calls["n"] += 1
        if rn == 2:
            raise RuntimeError("boom")
        return real(agents, rn)
    monkeypatch.setattr(orch, "run_round", flaky)
    orch.run_simulation(max_rounds=4, round_delay=0, verbose=False, seed=1)
    evs = events.read_all()
    assert [e["type"] for e in evs].count("round_error") == 1
    assert evs[-1]["type"] == "simulation_ended" and calls["n"] == 4
