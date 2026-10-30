"""
Repeatable base-vs-fine-tuned evaluation on HELD-OUT tasks.

Held-out = core.tasks.is_heldout(prompt): a deterministic ~20% slice of the task
bank that build_dataset.py never trains on. Both models see exactly the same
prompts, system prompts and parameters.

Two probes per model (same inputs for both):
  solve     ask each held-out task directly -> accuracy (with 95% Wilson interval),
            invalid-answer rate, mean latency, simulated cost per correct answer,
            net economic outcome (bounties earned - simulated compute spent)
  decision  show the neutral agent prompt for each held-out task -> invalid-decision
            rate, bid rate, and ESCALATION frequency (share of bids whose
            confidence is below the escalation cutoff)

Usage:
    python -m training.evaluate --label base
    python -m training.evaluate --label tuned --model-id <served fine-tuned model> --out data/eval/tuned.json
    python -m training.evaluate --compare data/eval/base.json data/eval/tuned.json

MOCK MODE (USE_FAKE_MODEL=1) measures the seeded fake model, which is the same
for "base" and "tuned": it exercises this pipeline, it says nothing about
Nemotron. Reports are stamped with the mode. This script never states that a
model "improved": --compare prints measured differences and intervals only.
"""
import argparse
import json
import sys
from pathlib import Path

from core import config, model_client, tasks
from core.model_client import ModelCallError
from core.orchestrator import DECISION_SYSTEM_PROMPT, SOLVE_SYSTEM_PROMPT
from core.simconfig import SimulationConfig
from core.stats import safe_div, wilson_interval
from core.validation import ValidationError, parse_json_object, validate_answer, validate_decision

NEUTRAL_AGENT = "You are a careful reasoning agent with a compute budget."


def heldout_tasks() -> list[tuple[str, str, str]]:
    return [t for t in tasks._TASK_BANK if tasks.is_heldout(t[0])]


def evaluate(model_id: str | None, label: str, tier: str = "nano", repeats: int = 1,
             seed: int = 0) -> dict:
    cfg = SimulationConfig.defaults()
    cost = cfg.ultra_cost if tier == "ultra" else cfg.nano_cost
    items = heldout_tasks()
    model_client.seed_fake(seed)

    solves = correct = invalid_ans = errors = 0
    latency, bounties = 0.0, 0
    decisions = invalid_dec = bids = low_conf_bids = 0

    for _ in range(repeats):
        for prompt, answer, difficulty in items:
            # --- solve probe ---
            solves += 1
            try:
                r = model_client.call_model_json(f"Question: {prompt}", tier=tier,
                                                 system=SOLVE_SYSTEM_PROMPT, model=model_id)
                latency += r.get("_latency", 0.0)
                if r.get("_parse_error"):
                    raise ValidationError(r.get("_error", "malformed_json"))
                ans, _ = validate_answer(r)
                t = tasks.Task(id=0, prompt=prompt, answer=answer, difficulty=difficulty, bounty=0)
                if tasks.grade(t, ans):
                    correct += 1
                    bounties += tasks._BOUNTY_BY_DIFFICULTY[difficulty]
            except ValidationError:
                invalid_ans += 1
            except ModelCallError:
                errors += 1

            # --- decision probe ---
            decisions += 1
            dprompt = (f"{NEUTRAL_AGENT}\n\nRound 1. Your current balance: {cfg.starting_balance} "
                       f"compute-seconds.\nSolving costs {cfg.nano_cost} (cheap tier) or {cfg.ultra_cost} "
                       f"(deep tier).\nIncoming task (difficulty {difficulty}, bounty "
                       f"{tasks._BOUNTY_BY_DIFFICULTY[difficulty]}): {prompt!r}\n"
                       "Decide your action for this round.")
            try:
                r = model_client.call_model_json(dprompt, tier="nano", system=DECISION_SYSTEM_PROMPT,
                                                 model=model_id)
                if r.get("_parse_error"):
                    raise ValidationError(r.get("_error", "malformed_json"))
                d = validate_decision(r)
                if d.action == "bid":
                    bids += 1
                    if d.confidence < cfg.escalation_cutoff:
                        low_conf_bids += 1
            except ValidationError:
                invalid_dec += 1
            except ModelCallError:
                errors += 1

    spent = solves * cost
    return {
        "label": label, "model": model_id or model_client._resolve_model("nano" if tier != "ultra" else "ultra"),
        "tier": tier, "mode": config.mode(), "seed": seed, "repeats": repeats,
        "heldout_tasks": len(items), "solve_attempts": solves,
        "accuracy": safe_div(correct, solves), "accuracy_ci95": wilson_interval(correct, solves),
        "invalid_answer_rate": safe_div(invalid_ans, solves),
        "provider_errors": errors,
        "mean_latency_s": round(latency / solves, 3) if solves else None,
        "sim_compute_per_correct": safe_div(spent, correct, 2),
        "sim_net_outcome": bounties - spent,
        "decisions": decisions, "invalid_decision_rate": safe_div(invalid_dec, decisions),
        "bid_rate": safe_div(bids, decisions - invalid_dec),
        "escalation_frequency": safe_div(low_conf_bids, bids),
        "caveats": ("Mock mode measures the fake model only." if config.USE_FAKE_MODEL else
                    "Small held-out set: read intervals, not point estimates."),
    }


def compare(a: dict, b: dict) -> str:
    keys = ["accuracy", "invalid_answer_rate", "invalid_decision_rate", "sim_compute_per_correct",
            "sim_net_outcome", "escalation_frequency", "mean_latency_s"]
    lines = [f"{'metric':28s} {a['label']:>12s} {b['label']:>12s}"]
    for k in keys:
        lines.append(f"{k:28s} {str(a.get(k)):>12s} {str(b.get(k)):>12s}")
    lines.append(f"accuracy 95% CI: {a['label']}={a.get('accuracy_ci95')}  {b['label']}={b.get('accuracy_ci95')}")
    ia, ib = a.get("accuracy_ci95"), b.get("accuracy_ci95")
    if ia and ib:
        overlap = not (ia[1] < ib[0] or ib[1] < ia[0])
        lines.append("accuracy intervals overlap: " + ("yes -- difference is NOT distinguishable at this sample size"
                                                      if overlap else "no"))
    if a.get("mode") != "live" or b.get("mode") != "live":
        lines.append("NOTE: at least one report is from MOCK mode; it says nothing about real models.")
    lines.append(f"held-out tasks: {a['heldout_tasks']} vs {b['heldout_tasks']} (repeats {a['repeats']}/{b['repeats']})")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="model")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--tier", choices=["nano", "ultra"], default="nano")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"))
    args = ap.parse_args()
    if args.compare:
        a, b = (json.loads(Path(p).read_text()) for p in args.compare)
        print(compare(a, b))
        sys.exit(0)
    if not config.USE_FAKE_MODEL and not config.NEBIUS_API_KEY:
        raise SystemExit("NEBIUS_API_KEY is not set (or use USE_FAKE_MODEL=1 for a pipeline dry run).")
    report = evaluate(args.model_id, args.label, args.tier, args.repeats, args.seed)
    print(json.dumps(report, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
