"""
Turns accumulated self-play history into fine-tuning datasets.

Sources (never modified by this script -- raw logs stay raw):
  1. data/training/calls.jsonl  every REAL (non-mock) model call: prompt, system,
     response, tagged with run_id/agent/round/purpose/task_id. Never reset.
  2. task_arrived / task_resolved events from data/events.jsonl and
     data/replays/*.jsonl -- they say whether a call's round worked out.

SELECTION CRITERIA (all documented here, all applied per call)
  - purpose "solve"    : the answer was graded correct
  - purpose "decision" : the agent bid AND went on to answer correctly
  - economics (schema v2 events only): net profit = bounty - compute cost must be
    >= --min-net-profit (default 0). Correct-but-money-losing escalations (e.g.
    an Ultra solve on an 8-bounty task) are therefore NOT imitated.
  - the response must itself be valid: it re-parses and passes core.validation
    (an example that the live app would have rejected is never a training target)
  - exhibition / user-injected tasks are EXCLUDED (user-supplied text: privacy)
  - HELD-OUT tasks (core.tasks.is_heldout) are EXCLUDED so training/evaluate.py
    measures on prompts the model never trained on
  - examples containing API-key-like strings, bearer tokens or e-mail addresses
    are dropped
  - exact duplicates (system+prompt+response) are dropped
  Skipped decisions (skip/lend/borrow) are deliberately excluded: we cannot grade them.
  Task provenance (is it held-out? injected?) comes from task_arrived events. If it
  is unknown, --strict drops the example (CLI default); the library default keeps
  it for backward compatibility.

Outputs (data/training/datasets/): nano_all.jsonl, ultra_all.jsonl,
<Agent>_<purpose>.jsonl, <tier>_all.meta.jsonl (one metadata row per example, aligned by line:
bounty, cost, net profit, strategy, task, run/round), optional <tier>_train/_val.jsonl, and
manifest.json (counts, criteria, skip reasons, sha256 of every file).

REPRODUCIBLE: output order is a pure function of the content (sorted by run, round, agent,
purpose), and the train/validation split is by hash of the TASK prompt -- the same question
can never appear in both, so validation is not contaminated. Same inputs => byte-identical files.

Usage:
    python -m training.build_dataset [--min-examples 20] [--min-net-profit 0] [--val-fraction 0.1] [--allow-unknown-provenance]
"""
import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from core import config
from core.tasks import is_heldout
from core.validation import ValidationError, parse_json_object, validate_answer, validate_decision

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CALLS_PATH = DATA_DIR / "training" / "calls.jsonl"
EVENTS_PATH = DATA_DIR / "events.jsonl"
REPLAY_DIR = DATA_DIR / "replays"
OUTPUT_DIR = DATA_DIR / "training" / "datasets"

_SENSITIVE = re.compile(
    r"(?i)(bearer\s+[A-Za-z0-9_\-\.]{16,}|\b[A-Za-z0-9_\-]{40,}\b|[\w.+-]+@[\w-]+\.[\w.-]+)")


def _read_jsonl(path: Path) -> list:
    from core.events import read_jsonl_tolerant
    return read_jsonl_tolerant(path)[0] if path.exists() else []


def load_all_events() -> list:
    """Every event ever recorded across the current run and all saved replays.
    Duplicates are harmless: they only feed dicts keyed by (run_id, ...)."""
    all_events = list(_read_jsonl(EVENTS_PATH))
    if REPLAY_DIR.exists():
        for path in sorted(REPLAY_DIR.glob("*.jsonl")):
            all_events.extend(_read_jsonl(path))
    return all_events


def build_outcome_map(all_events: list) -> dict:
    """(run_id, round, agent) -> {correct, tier, bounty, cost|None, task_id, injected|None}"""
    outcomes = {}
    for e in all_events:
        if e.get("type") != "task_resolved":
            continue
        key = (e.get("run_id"), e.get("round"), e.get("agent"))
        outcomes[key] = {
            "correct": e.get("correct", False),
            "tier": e.get("tier"),
            "bounty": e.get("bounty", 0),
            "cost": e.get("cost") if isinstance(e.get("cost"), int) else None,
            "task_id": e.get("task_id"),
            "injected": e.get("injected"),
        }
    return outcomes


def build_task_map(all_events: list) -> dict:
    """(run_id, task_id) -> {prompt, injected}"""
    out = {}
    for e in all_events:
        if e.get("type") == "task_arrived" and e.get("task_id") is not None:
            out[(e.get("run_id"), e.get("task_id"))] = {
                "prompt": e.get("prompt", ""), "injected": bool(e.get("injected"))}
    return out


def to_chat_example(call: dict) -> dict:
    messages = []
    if call.get("system"):
        messages.append({"role": "system", "content": call["system"]})
    messages.append({"role": "user", "content": call["prompt"]})
    messages.append({"role": "assistant", "content": call["response"]})
    return {"messages": messages}


def _response_is_valid(call: dict, purpose: str) -> tuple[bool, str | None]:
    """Returns (ok, bid_action_if_decision)."""
    try:
        obj = parse_json_object(call.get("response"))
        if purpose == "solve":
            validate_answer(obj)
            return True, None
        d = validate_decision(obj)
        return True, d.action
    except ValidationError:
        return False, None


def _split_bucket(task_prompt: str | None) -> float:
    """Deterministic value in [0,1) from the TASK prompt, used for the train/validation split."""
    h = hashlib.sha256((task_prompt or "").encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") / 2**32


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_datasets(min_examples: int = 10, min_net_profit: int | None = 0, strict: bool = False,
                   val_fraction: float = 0.0):
    calls = _read_jsonl(CALLS_PATH)
    if not calls:
        print(f"No training calls found at {CALLS_PATH}.")
        print("Run some simulations first (see training/run_self_play.py), "
              "with USE_FAKE_MODEL=0 and LOG_TRAINING_DATA=1 in .env.")
        return None

    all_events = load_all_events()
    outcomes = build_outcome_map(all_events)
    task_map = build_task_map(all_events)

    skipped = defaultdict(int)
    kept = []        # (sort_key, example, meta, agent, purpose, tier, task_prompt)
    seen = set()

    for call in calls:
        key = (call.get("run_id"), call.get("round"), call.get("agent"))
        outcome = outcomes.get(key)
        purpose = call.get("purpose")

        if purpose not in ("solve", "decision"):
            skipped["not_decision_or_solve"] += 1
            continue
        if outcome is None:
            skipped["no_matching_outcome"] += 1
            continue
        if not call.get("prompt") or not call.get("response"):
            skipped["empty_or_malformed_record"] += 1
            continue

        # provenance: held-out / user-injected tasks never enter training data
        task = task_map.get((call.get("run_id"), outcome.get("task_id")))
        injected = outcome.get("injected")
        if injected is None and task is not None:
            injected = task["injected"]
        if call.get("injected") or injected:
            skipped["user_injected_task"] += 1
            continue
        if task is None:
            if strict:
                skipped["unknown_provenance"] += 1
                continue
        elif is_heldout(task["prompt"]):
            skipped["heldout_task"] += 1
            continue

        ok, action = _response_is_valid(call, purpose)
        if not ok:
            skipped["invalid_response"] += 1
            continue

        if purpose == "solve":
            keep = outcome["correct"] is True
        else:
            keep = action == "bid" and outcome["correct"] is True
        if not keep:
            skipped["bad_outcome"] += 1
            continue

        if min_net_profit is not None and outcome["cost"] is not None:
            if outcome["bounty"] - outcome["cost"] < min_net_profit:
                skipped["unprofitable"] += 1
                continue

        if _SENSITIVE.search(call["prompt"]) or _SENSITIVE.search(call["response"]):
            skipped["sensitive_content"] += 1
            continue

        fingerprint = hashlib.sha256(
            json.dumps([call.get("system"), call["prompt"], call["response"]]).encode()).hexdigest()
        if fingerprint in seen:
            skipped["duplicate"] += 1
            continue
        seen.add(fingerprint)

        net = (outcome["bounty"] - outcome["cost"]) if outcome["cost"] is not None else None
        meta = {"run_id": call.get("run_id"), "round": call.get("round"), "agent": call.get("agent"),
                "purpose": purpose, "tier": call.get("tier"), "task_id": outcome.get("task_id"),
                "bounty": outcome["bounty"], "cost": outcome["cost"], "net_profit": net, "model": call.get("model")}
        kept.append(((str(call.get("run_id")), call.get("round") or 0, str(call.get("agent")), purpose),
                     to_chat_example(call), meta, call.get("agent"), purpose, call.get("tier"),
                     task["prompt"] if task else None))
    kept.sort(key=lambda k: k[0])             # deterministic order, independent of log order

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kept_nano = [k for k in kept if k[5] == "nano"]
    kept_ultra = [k for k in kept if k[5] != "nano"]
    kept_by_agent_purpose = defaultdict(list)
    for k in kept:
        kept_by_agent_purpose[(k[3], k[4])].append(k[1])

    def write_jsonl(path: Path, rows: list):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + "\n")

    files = {}
    for tier, group in (("nano", kept_nano), ("ultra", kept_ultra)):
        write_jsonl(OUTPUT_DIR / f"{tier}_all.jsonl", [k[1] for k in group])
        write_jsonl(OUTPUT_DIR / f"{tier}_all.meta.jsonl", [k[2] for k in group])
        files[f"{tier}_all.jsonl"] = len(group)
        if val_fraction > 0:
            val = [k for k in group if k[6] is not None and _split_bucket(k[6]) < val_fraction]
            train = [k for k in group if not (k[6] is not None and _split_bucket(k[6]) < val_fraction)]
            write_jsonl(OUTPUT_DIR / f"{tier}_train.jsonl", [k[1] for k in train])
            write_jsonl(OUTPUT_DIR / f"{tier}_val.jsonl", [k[1] for k in val])
            files[f"{tier}_train.jsonl"], files[f"{tier}_val.jsonl"] = len(train), len(val)
    for (agent, purpose), examples in kept_by_agent_purpose.items():
        write_jsonl(OUTPUT_DIR / f"{agent}_{purpose}.jsonl", examples)

    manifest = {
        "raw_calls": len(calls), "kept_nano": len(kept_nano), "kept_ultra": len(kept_ultra),
        "skipped": dict(skipped),
        "criteria": {"min_net_profit": min_net_profit, "strict_provenance": strict,
                     "heldout_excluded": True, "injected_excluded": True, "dedup": True,
                     "validated_responses_only": True, "val_fraction": val_fraction,
                     "split": "by sha256 of the task prompt (a task is never in both train and val)"},
        "files": {name: {"examples": n, "sha256": _file_sha256(OUTPUT_DIR / name)} for name, n in files.items()},
        "note": "Selection imitates successful, profitable bids/answers; it does not prove improvement. "
                "Measure with training/evaluate.py on held-out tasks.",
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    kept_nano_all, kept_ultra_all = kept_nano, kept_ultra

    print(f"Read {len(calls)} raw calls.")
    for reason, n in sorted(skipped.items()):
        print(f"  skipped {reason}: {n}")
    print()
    print(f"Wrote {OUTPUT_DIR / 'nano_all.jsonl'}  -- {len(kept_nano_all)} examples")
    print(f"Wrote {OUTPUT_DIR / 'ultra_all.jsonl'} -- {len(kept_ultra_all)} examples")
    print()
    print("Per-agent breakdown:")
    for (agent, purpose), examples in sorted(kept_by_agent_purpose.items(), key=lambda kv: str(kv[0])):
        flag = "" if len(examples) >= min_examples else "  <-- below --min-examples"
        print(f"  {str(agent):12s} {purpose:9s} {len(examples):4d} examples{flag}")
    if len(kept_nano_all) < min_examples:
        print()
        print(f"WARNING: only {len(kept_nano_all)} total nano examples, fewer than "
              f"--min-examples ({min_examples}). Run more self-play before fine-tuning.")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--min-examples", type=int, default=10)
    parser.add_argument("--min-net-profit", type=int, default=0,
                        help="keep only examples whose bounty - compute cost is at least this (schema v2 logs)")
    parser.add_argument("--val-fraction", type=float, default=0.0,
                        help="hold out this fraction of TASKS as a validation set (e.g. 0.1); 0 disables")
    parser.add_argument("--allow-unknown-provenance", action="store_true",
                        help="keep calls whose task can't be identified (may include held-out/injected tasks)")
    args = parser.parse_args()
    build_datasets(min_examples=args.min_examples, min_net_profit=args.min_net_profit,
                   strict=not args.allow_unknown_provenance, val_fraction=args.val_fraction)
