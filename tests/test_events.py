from core import events


def test_reset_log_generates_new_run_id(isolated_data_dir):
    run_id_1 = events.reset_log()
    run_id_2 = events.reset_log()
    assert run_id_1 != run_id_2


def test_every_emitted_event_carries_current_run_id(isolated_data_dir):
    run_id = events.reset_log()
    e = events.emit("task_arrived", round=1, task_id=1, prompt="q", difficulty="easy", bounty=8)
    assert e["run_id"] == run_id


def test_read_all_returns_emitted_events_in_order(isolated_data_dir):
    events.reset_log()
    events.emit("a", value=1)
    events.emit("b", value=2)
    all_events = events.read_all()
    assert [e["type"] for e in all_events] == ["a", "b"]


def test_reset_log_clears_previous_events(isolated_data_dir):
    events.reset_log()
    events.emit("a", value=1)
    events.reset_log()
    assert events.read_all() == []


def test_save_and_read_replay_roundtrips(isolated_data_dir):
    events.reset_log()
    events.emit("simulation_started", agents=["A", "B"], starting_balance=100)
    events.emit("task_arrived", round=1, task_id=1, prompt="q", difficulty="easy", bounty=8)

    filename = events.save_replay(label="test-run")
    replayed = events.read_replay(filename)

    assert len(replayed) == 2
    assert replayed[0]["type"] == "simulation_started"


def test_list_replays_reports_event_count_and_summary(isolated_data_dir):
    events.reset_log()
    events.emit("simulation_started", agents=["A"], starting_balance=100)
    events.emit("monopoly", round=5, winner="A")
    events.save_replay(label="A")

    listed = events.list_replays()
    assert len(listed) == 1
    assert listed[0]["event_count"] == 2
    assert listed[0]["summary"] == "A"


def test_replay_path_traversal_is_rejected(isolated_data_dir):
    events.reset_log()
    events.emit("a", value=1)
    events.save_replay(label="safe")

    import pytest
    with pytest.raises(ValueError):
        events.read_replay("../../etc/passwd")


def test_leaderboard_persists_across_resets(isolated_data_dir):
    events.reset_log()
    events.record_leaderboard_entry("Specialist", {"Specialist": 150, "Hoarder": 90}, 20, "max_rounds_reached")

    events.reset_log()  # simulates starting a brand new run
    events.record_leaderboard_entry("Hoarder", {"Specialist": 80, "Hoarder": 160}, 18, "monopoly")

    entries = events.read_leaderboard()
    assert len(entries) == 2
    assert entries[0]["winner"] == "Specialist"
    assert entries[1]["winner"] == "Hoarder"


def test_leaderboard_survives_event_log_reset(isolated_data_dir):
    """The whole point of a separate leaderboard file: events.jsonl gets
    wiped every run, but win history must not be lost."""
    events.reset_log()
    events.record_leaderboard_entry("A", {"A": 100}, 10, "max_rounds_reached")
    events.reset_log()
    assert events.read_all() == []  # events.jsonl is empty after reset
    assert len(events.read_leaderboard()) == 1  # but leaderboard survived
