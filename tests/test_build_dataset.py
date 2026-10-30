import json

from training import build_dataset


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_build_dataset_keeps_only_successful_traces(isolated_data_dir):
    calls = [
        # Specialist bids and solves correctly in run1 round1 -- should be KEPT
        {"run_id": "run1", "round": 1, "agent": "Specialist", "purpose": "decision",
         "tier": "nano", "system": "SYS", "prompt": "p1", "response": '{"action": "bid", "confidence": 0.9}'},
        {"run_id": "run1", "round": 1, "agent": "Specialist", "purpose": "solve",
         "tier": "nano", "system": "SYS", "prompt": "p2", "response": '{"answer": "42", "confidence": 0.9}'},

        # Gambler bids and gets it WRONG in run1 round2 -- should be EXCLUDED
        {"run_id": "run1", "round": 2, "agent": "Gambler", "purpose": "decision",
         "tier": "nano", "system": "SYS", "prompt": "p3", "response": '{"action": "bid", "confidence": 0.4}'},
        {"run_id": "run1", "round": 2, "agent": "Gambler", "purpose": "solve",
         "tier": "ultra", "system": "SYS", "prompt": "p4", "response": '{"answer": "wrong", "confidence": 0.4}'},

        # Hoarder skips round3 -- no matching outcome exists -- should be EXCLUDED
        {"run_id": "run1", "round": 3, "agent": "Hoarder", "purpose": "decision",
         "tier": "nano", "system": "SYS", "prompt": "p5", "response": '{"action": "skip", "confidence": 0.2}'},

        # Specialist bids and solves correctly AGAIN in run2 round1 -- note the
        # SAME round number as run1's example, in a DIFFERENT run_id. This is
        # the case that requires run_id in the join key, not just round number.
        {"run_id": "run2", "round": 1, "agent": "Specialist", "purpose": "decision",
         "tier": "nano", "system": "SYS", "prompt": "p6", "response": '{"action": "bid", "confidence": 0.85}'},
        {"run_id": "run2", "round": 1, "agent": "Specialist", "purpose": "solve",
         "tier": "nano", "system": "SYS", "prompt": "p7", "response": '{"answer": "correct", "confidence": 0.85}'},
    ]

    outcome_events = [
        {"type": "task_resolved", "run_id": "run1", "round": 1, "agent": "Specialist",
         "tier": "nano", "correct": True, "bounty": 8},
        {"type": "task_resolved", "run_id": "run1", "round": 2, "agent": "Gambler",
         "tier": "ultra", "correct": False, "bounty": 0},
        {"type": "task_resolved", "run_id": "run2", "round": 1, "agent": "Specialist",
         "tier": "nano", "correct": True, "bounty": 8},
    ]

    _write_jsonl(build_dataset.CALLS_PATH, calls)
    _write_jsonl(build_dataset.EVENTS_PATH, outcome_events)

    build_dataset.build_datasets(min_examples=1)

    nano_dataset = (build_dataset.OUTPUT_DIR / "nano_all.jsonl").read_text().strip().splitlines()
    ultra_dataset = (build_dataset.OUTPUT_DIR / "ultra_all.jsonl").read_text().strip().splitlines()

    assert len(nano_dataset) == 4  # Specialist's 2 decisions + 2 solves, across both runs
    assert len(ultra_dataset) == 0  # Gambler's ultra call was wrong, correctly excluded

    # confirm the actual content is right, not just the count
    responses = [json.loads(line)["messages"][-1]["content"] for line in nano_dataset]
    assert '{"action": "bid", "confidence": 0.9}' in responses
    assert '{"answer": "42", "confidence": 0.9}' in responses
    assert '{"action": "bid", "confidence": 0.85}' in responses
    assert '{"answer": "correct", "confidence": 0.85}' in responses
    # the wrong/skipped ones must never appear
    assert not any("wrong" in r for r in responses)
    assert not any("skip" in r for r in responses)


def test_build_dataset_per_agent_breakdown(isolated_data_dir):
    calls = [
        {"run_id": "r1", "round": 1, "agent": "Specialist", "purpose": "solve",
         "tier": "nano", "system": "S", "prompt": "p", "response": '{"answer": "4", "confidence": 0.9}'},
    ]
    outcomes = [
        {"type": "task_resolved", "run_id": "r1", "round": 1, "agent": "Specialist",
         "tier": "nano", "correct": True, "bounty": 8},
    ]
    _write_jsonl(build_dataset.CALLS_PATH, calls)
    _write_jsonl(build_dataset.EVENTS_PATH, outcomes)

    build_dataset.build_datasets(min_examples=1)

    per_agent_file = build_dataset.OUTPUT_DIR / "Specialist_solve.jsonl"
    assert per_agent_file.exists()
    assert len(per_agent_file.read_text().strip().splitlines()) == 1


def test_build_dataset_excludes_non_bid_decisions_even_with_correct_outcome(isolated_data_dir):
    """A skip/lend/borrow decision in a round someone else won and solved
    correctly must not be attributed to the skipper as a 'good decision'."""
    calls = [
        {"run_id": "r1", "round": 1, "agent": "Hoarder", "purpose": "decision",
         "tier": "nano", "system": "S", "prompt": "p", "response": '{"action": "skip", "confidence": 0.1}'},
    ]
    outcomes = [
        # note: outcome is for a DIFFERENT agent's key wouldn't even match here,
        # but even if Hoarder's own round somehow resolved correct=True while
        # Hoarder skipped, a skip is not a "bid that worked" and must be excluded
        {"type": "task_resolved", "run_id": "r1", "round": 1, "agent": "Hoarder",
         "tier": "nano", "correct": True, "bounty": 8},
    ]
    _write_jsonl(build_dataset.CALLS_PATH, calls)
    _write_jsonl(build_dataset.EVENTS_PATH, outcomes)

    build_dataset.build_datasets(min_examples=1)

    nano_dataset_path = build_dataset.OUTPUT_DIR / "nano_all.jsonl"
    if nano_dataset_path.exists():
        assert nano_dataset_path.read_text().strip() == ""


def test_build_dataset_handles_empty_calls_file_gracefully(isolated_data_dir, capsys):
    build_dataset.build_datasets(min_examples=1)  # should not raise
    captured = capsys.readouterr()
    assert "No training calls found" in captured.out


def test_build_dataset_reads_outcomes_from_replays_too(isolated_data_dir):
    """Outcomes from past (now-replayed) runs must count too, not just
    the current events.jsonl -- this is how self-play across many
    separate run_self_play.py sessions accumulates into one dataset."""
    calls = [
        {"run_id": "old_run", "round": 1, "agent": "Specialist", "purpose": "solve",
         "tier": "nano", "system": "S", "prompt": "p", "response": '{"answer": "4", "confidence": 0.9}'},
    ]
    _write_jsonl(build_dataset.CALLS_PATH, calls)

    # simulate a past, already-saved replay (events.jsonl itself is empty --
    # this run's outcomes only exist in the replay file now)
    replay_events = [
        {"type": "task_resolved", "run_id": "old_run", "round": 1, "agent": "Specialist",
         "tier": "nano", "correct": True, "bounty": 8},
    ]
    _write_jsonl(build_dataset.REPLAY_DIR / "old-run.jsonl", replay_events)
    _write_jsonl(build_dataset.EVENTS_PATH, [])  # current run has nothing

    build_dataset.build_datasets(min_examples=1)

    nano_dataset = (build_dataset.OUTPUT_DIR / "nano_all.jsonl").read_text().strip().splitlines()
    assert len(nano_dataset) == 1
