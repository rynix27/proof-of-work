"""The demo scenario must keep telling its story: tests fail loudly if a change alters it."""
import json

import pytest

from core import events
from training import demo


@pytest.fixture(scope="module")
def demo_events(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    saved = (events.LOG_PATH, events.REPLAY_DIR, events.LEADERBOARD_PATH)
    try:
        yield demo._simulate(demo.DEMO_SEED, out)
    finally:
        events.LOG_PATH, events.REPLAY_DIR, events.LEADERBOARD_PATH = saved


def test_pinned_seed_contains_every_story_beat_and_balances(demo_events):
    assert demo.missing_beats(demo_events) == []


def test_demo_is_mock_mode_and_not_scripted(demo_events):
    start = [e for e in demo_events if e["type"] == "simulation_started"][0]
    assert start["mode"] == "mock" and start["seed"] == demo.DEMO_SEED
    assert start["config"]["strategies"] == "roster" and start["config"]["starting_balance"] == 30


def test_demo_is_deterministic(demo_events, tmp_path):
    again = demo._simulate(demo.DEMO_SEED, tmp_path)
    key = lambda evs: [(e["type"], e.get("agent"), e.get("round"), e.get("new_balance")) for e in evs]
    assert key(again) == key(demo_events)


def test_narrative_is_in_round_order_and_covers_the_story(demo_events):
    story = demo.narrative(demo_events)
    rounds = [r for r, _ in story]
    assert rounds == sorted(rounds)
    text = " ".join(t for _, t in story)
    for needle in ("strategy chose", "escalated", "lent", "fully repaid", "defaulted", "went bankrupt", "Ledger balanced: True"):
        assert needle in text


def test_missing_beats_reports_what_is_absent():
    assert set(demo.missing_beats([])) >= {"loan_made", "bankruptcy", "policy_override", "balanced_books"}
    ok_end = {"type": "simulation_ended", "reconciliation": {"ok": False}}
    assert "balanced_books" in demo.missing_beats([ok_end])


def test_find_seed_returns_the_first_seed_with_every_beat(monkeypatch, tmp_path):
    def fake(seed, out):
        evs = [{"type": "simulation_ended", "reconciliation": {"ok": True}},
               {"type": "decision", "policy_reasons": ["x"]}]
        if seed == 3:
            evs += [{"type": t} for t in demo.REQUIRED]
        return evs
    monkeypatch.setattr(demo, "_simulate", fake)
    assert demo.find_seed(limit=10) == 3
    assert demo.find_seed(limit=2) is None


def test_cli_prints_story_and_installs_replay(isolated_data_dir, capsys):
    saved = events.REPLAY_DIR
    demo.main(["--install-replay"])
    out = capsys.readouterr().out
    assert "Demo scenario (mock mode)" in out and "round" in out and "WARNING" not in out
    installed = saved / f"demo-seed{demo.DEMO_SEED}.jsonl"
    assert installed.exists()
    rows = [json.loads(l) for l in installed.read_text().splitlines()]
    assert rows[0]["type"] == "simulation_started" and rows[-1]["type"] == "simulation_ended"
