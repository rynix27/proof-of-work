"""Dataset builder: determinism, leak-free validation split, metadata alignment, content hashes."""
import hashlib
import json
import random

import pytest

from core import tasks
from training import build_dataset as bd

TRAIN_PROMPTS = [p for p, *_ in tasks._TASK_BANK if not tasks.is_heldout(p)]


def make_logs(prompts, shuffle_seed=None, tier="nano"):
    evs, calls = [], []
    for i, prompt in enumerate(prompts, 1):
        for agent in ("A", "B"):                                    # two agents answer each task
            evs += [{"type": "task_arrived", "run_id": "r", "task_id": i, "prompt": prompt, "injected": False},
                    {"type": "task_resolved", "run_id": "r", "round": i, "agent": agent, "correct": True, "tier": tier,
                     "bounty": 8, "cost": 1, "task_id": i, "injected": False}]
            calls.append({"system": "s", "prompt": f"Question: {prompt} ({agent})", "tier": tier, "purpose": "solve",
                          "response": json.dumps({"answer": "5", "confidence": 0.9}), "run_id": "r", "round": i,
                          "agent": agent, "task_id": i, "model": "m"})
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(calls)
        random.Random(shuffle_seed).shuffle(evs)
    bd.EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bd.EVENTS_PATH.write_text("\n".join(json.dumps(e) for e in evs) + "\n")
    bd.CALLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    bd.CALLS_PATH.write_text("\n".join(json.dumps(c) for c in calls) + "\n")


def read(name):
    return [json.loads(l) for l in (bd.OUTPUT_DIR / name).read_text().splitlines()]


def test_same_inputs_give_byte_identical_outputs_regardless_of_log_order(isolated_data_dir):
    make_logs(TRAIN_PROMPTS)
    bd.build_datasets(min_examples=1, strict=True, val_fraction=0.2)
    first = {f.name: f.read_bytes() for f in bd.OUTPUT_DIR.glob("*.jsonl")}
    make_logs(TRAIN_PROMPTS, shuffle_seed=7)                         # same data, shuffled input order
    bd.build_datasets(min_examples=1, strict=True, val_fraction=0.2)
    second = {f.name: f.read_bytes() for f in bd.OUTPUT_DIR.glob("*.jsonl")}
    assert first == second and first


def test_validation_split_never_shares_a_task_with_training(isolated_data_dir):
    make_logs(TRAIN_PROMPTS)
    bd.build_datasets(min_examples=1, strict=True, val_fraction=0.3)
    train, val = read("nano_train.jsonl"), read("nano_val.jsonl")
    assert train and val                                               # the split is non-trivial
    task_of = lambda ex: next(p for p in TRAIN_PROMPTS if f"Question: {p} (" in ex["messages"][-2]["content"])
    assert {task_of(e) for e in train}.isdisjoint({task_of(e) for e in val})
    assert len(train) + len(val) == len(read("nano_all.jsonl"))
    # both agents' answers to the same task land on the SAME side (a task is atomic)
    for p in TRAIN_PROMPTS:
        sides = {("val" if any(task_of(e) == p for e in val) else "train")}
        assert len(sides) == 1


def test_split_is_stable_and_roughly_proportional(isolated_data_dir):
    sides = [bd._split_bucket(p) < 0.2 for p in TRAIN_PROMPTS]
    assert sides == [bd._split_bucket(p) < 0.2 for p in TRAIN_PROMPTS]
    assert 0 < sum(sides) < len(sides)
    assert all(0.0 <= bd._split_bucket(p) < 1.0 for p in TRAIN_PROMPTS + ["", "x" * 1000])


def test_no_train_or_val_files_unless_a_split_is_requested(isolated_data_dir):
    make_logs(TRAIN_PROMPTS)
    bd.build_datasets(min_examples=1, strict=True)
    assert not (bd.OUTPUT_DIR / "nano_train.jsonl").exists() and not (bd.OUTPUT_DIR / "nano_val.jsonl").exists()
    assert json.loads((bd.OUTPUT_DIR / "manifest.json").read_text())["criteria"]["val_fraction"] == 0.0


def test_metadata_sidecar_is_line_aligned_and_carries_the_economics(isolated_data_dir):
    make_logs(TRAIN_PROMPTS)
    bd.build_datasets(min_examples=1, strict=True)
    ex, meta = read("nano_all.jsonl"), read("nano_all.meta.jsonl")
    assert len(ex) == len(meta) > 0
    for e, m in zip(ex, meta):
        assert m["agent"] in e["messages"][-2]["content"]              # the metadata belongs to THIS example
        assert (m["bounty"], m["cost"], m["net_profit"], m["tier"], m["purpose"]) == (8, 1, 7, "nano", "solve")


def test_manifest_hashes_match_the_files_on_disk(isolated_data_dir):
    make_logs(TRAIN_PROMPTS)
    bd.build_datasets(min_examples=1, strict=True, val_fraction=0.25)
    man = json.loads((bd.OUTPUT_DIR / "manifest.json").read_text())
    assert {"nano_all.jsonl", "nano_train.jsonl", "nano_val.jsonl", "ultra_all.jsonl"} <= set(man["files"])
    for name, info in man["files"].items():
        data = (bd.OUTPUT_DIR / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == info["sha256"]
        assert len(data.splitlines()) == info["examples"]
    assert man["criteria"]["val_fraction"] == 0.25 and "never in both" in man["criteria"]["split"]


def test_ultra_examples_go_to_the_ultra_files(isolated_data_dir):
    make_logs(TRAIN_PROMPTS[:5], tier="ultra")
    bd.build_datasets(min_examples=1, strict=True, min_net_profit=None)
    assert read("ultra_all.jsonl") and read("nano_all.jsonl") == []
    assert all(m["tier"] == "ultra" for m in read("ultra_all.meta.jsonl"))


def test_heldout_tasks_stay_out_of_every_output_file(isolated_data_dir):
    held = [p for p, *_ in tasks._TASK_BANK if tasks.is_heldout(p)]
    make_logs(TRAIN_PROMPTS + held)
    bd.build_datasets(min_examples=1, strict=True, val_fraction=0.2)
    for f in bd.OUTPUT_DIR.glob("*.jsonl"):
        text = f.read_text()
        for p in held:
            assert f"Question: {p} (" not in text, (f.name, p)
