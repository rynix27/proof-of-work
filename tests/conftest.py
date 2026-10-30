"""
Shared fixtures. The big one is `isolated_data_dir`, which redirects every
module-level Path constant (events log, replays, leaderboard, training
log, dataset builder paths) into a fresh pytest tmp_path -- without this,
running the test suite would read and write the real data/ directory.

Also resets core.control's singleton state between tests, since it's a
process-wide object and simulations in different tests shouldn't see each
other's pause/generation state.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import config


@pytest.fixture(autouse=True)
def fake_model_mode(monkeypatch):
    """Every test runs in fake mode by default unless it explicitly
    overrides USE_FAKE_MODEL -- keeps the suite fast and network-free."""
    monkeypatch.setattr(config, "USE_FAKE_MODEL", True)
    monkeypatch.setattr(config, "LOG_TRAINING_DATA", True)


@pytest.fixture(autouse=True)
def _never_touch_real_data(tmp_path, monkeypatch):
    """Safety net: even a test that forgets `isolated_data_dir` can never write
    into the repository's real data/ directory."""
    from core import events, model_client
    monkeypatch.setattr(events, "LOG_PATH", tmp_path / "_auto" / "events.jsonl")
    monkeypatch.setattr(events, "REPLAY_DIR", tmp_path / "_auto" / "replays")
    monkeypatch.setattr(events, "LEADERBOARD_PATH", tmp_path / "_auto" / "leaderboard.jsonl")
    monkeypatch.setattr(model_client, "TRAINING_LOG_PATH", tmp_path / "_auto" / "calls.jsonl")


@pytest.fixture(autouse=True)
def _no_leak_into_real_data():
    """Fail any test that creates or modifies a file in the repository's real data/ directory.
    The redirect fixture above can be defeated (e.g. by monkeypatch.undo()); this makes that loud."""
    from pathlib import Path
    real = Path(__file__).resolve().parent.parent / "data"

    def listing():
        return {p: p.stat().st_mtime_ns for p in real.rglob("*") if p.is_file()} if real.exists() else {}
    before = listing()
    yield
    after = listing()
    leaked = sorted(str(p.relative_to(real)) for p in after if p not in before or after[p] != before[p])
    assert not leaked, f"test leaked files into the real data/ directory: {leaked}"


@pytest.fixture(autouse=True)
def _baseline_strategy_for_rule_tests(monkeypatch):
    """The auction/escalation/accounting rule tests pin the *baseline* policy (no
    strategy gating) so they test the rules in isolation. Tests of the strategy layer
    opt in explicitly via config overrides (strategies=...)."""
    from core import config
    monkeypatch.setattr(config, "STRATEGY_SET", "baseline")


@pytest.fixture(autouse=True)
def _reset_globals():
    """Process-wide state that must not leak between tests."""
    from core import events
    from core.injected_tasks import injected_tasks
    from core.runner import manager
    import server
    server.limiter.reset()
    injected_tasks.clear()
    events.bind_run(None)
    yield
    manager.shutdown(timeout=3)
    events.bind_run(None)


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch):
    from core import events, model_client
    from training import build_dataset

    events_path = tmp_path / "events.jsonl"
    replay_dir = tmp_path / "replays"
    leaderboard_path = tmp_path / "leaderboard.jsonl"
    training_dir = tmp_path / "training"
    dataset_dir = training_dir / "datasets"

    monkeypatch.setattr(events, "LOG_PATH", events_path)
    monkeypatch.setattr(events, "REPLAY_DIR", replay_dir)
    monkeypatch.setattr(events, "LEADERBOARD_PATH", leaderboard_path)
    monkeypatch.setattr(events, "_current_run_id", None)

    monkeypatch.setattr(model_client, "TRAINING_LOG_PATH", training_dir / "calls.jsonl")

    monkeypatch.setattr(build_dataset, "CALLS_PATH", training_dir / "calls.jsonl")
    monkeypatch.setattr(build_dataset, "EVENTS_PATH", events_path)
    monkeypatch.setattr(build_dataset, "REPLAY_DIR", replay_dir)
    monkeypatch.setattr(build_dataset, "OUTPUT_DIR", dataset_dir)

    return tmp_path


@pytest.fixture(autouse=True)
def reset_control():
    """core.control.control is a process-wide singleton -- make sure one
    test's pause/generation state can't leak into the next."""
    from core.control import control
    control.resume()
    control.start_new_generation()
    control.set_speed(0)
    yield
    control.resume()
    control.start_new_generation()
