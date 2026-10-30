"""
Reproducible demo scenario (mock mode, no API key, no network).

    python -m training.demo                     # run it and print a plain-English timeline
    python -m training.demo --install-replay    # also add it to the dashboard's Replays list
    python -m training.demo --find              # search for a seed if this one ever stops working

The scenario is NOT scripted: nothing is staged. It is a normal 50-round run with the
default agent strategies on a poorer economy (starting balance 30), at a seed chosen because
the run naturally contains every beat a demo needs -- a loan made and repaid, a loan
default, a bankruptcy, an escalation to the expensive model, and a strategy overriding the
model's own proposal. Only 1 of 200 tried (config, seed) pairs did, so the seed is pinned
and tests/test_demo.py fails loudly if a code change ever alters what it produces.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from core import events, model_client
from core.orchestrator import run_simulation

DEMO_SEED = 8
DEMO_CONFIG = {"starting_balance": 30, "max_rounds": 50, "strategies": "roster"}
REQUIRED = ("loan_made", "loan_repayment", "loan_default", "bankruptcy", "escalation")


def _simulate(seed: int, out_dir: Path) -> list[dict]:
    events.LOG_PATH = out_dir / "events.jsonl"
    events.REPLAY_DIR = out_dir / "replays"
    events.LEADERBOARD_PATH = out_dir / "leaderboard.jsonl"
    model_client.TRAINING_LOG_PATH = out_dir / "calls.jsonl"
    events.reset_log()
    run_simulation(round_delay=0, verbose=False, seed=seed, config_overrides=dict(DEMO_CONFIG))
    return events.read_all()


def missing_beats(evs: list[dict]) -> list[str]:
    seen = {e["type"] for e in evs}
    out = [b for b in REQUIRED if b not in seen]
    if not any(e["type"] == "decision" and e.get("policy_reasons") for e in evs):
        out.append("policy_override")
    end = [e for e in evs if e["type"] == "simulation_ended"]
    if not end or not end[-1]["reconciliation"]["ok"]:
        out.append("balanced_books")
    return out


def narrative(evs: list[dict]) -> list[tuple]:
    """(round, text) for the first occurrence of each story beat, in round order."""
    first = {}
    for e in evs:
        t = e["type"]
        if t == "decision" and e.get("policy_reasons") and e.get("model_action") != e.get("action"):
            first.setdefault("override", (e["round"],
                f"{e['agent']} ({e['strategy']}) was proposed '{e['model_action']}' by the model, but its strategy "
                f"chose '{e['action']}' [{', '.join(e['policy_reasons'])}]: {e['explanation']}"))
        elif t == "escalation":
            first.setdefault("escalation", (e["round"], f"{e['agent']} escalated to the expensive model (confidence "
                                                        f"{e['confidence']} < cutoff {e['cutoff']}, cost {e['cost']})."))
        elif t == "escalation_denied":
            first.setdefault("denied", (e["round"], f"{e['agent']} wanted to escalate but was refused: "
                                                    f"{', '.join(e['reasons'])}."))
        elif t == "loan_made":
            first.setdefault("loan", (e["round"], f"{e['lender']} lent {e['principal']} to {e['borrower']}; "
                                                  f"{e['total_owed']} due by round {e['due_round']}."))
        elif t == "loan_repayment" and e.get("status") == "repaid":
            first.setdefault("repaid", (e["round"], f"{e['borrower']} fully repaid {e['lender']}."))
        elif t == "loan_default":
            first.setdefault("default", (e["round"], f"Loan #{e['loan_id']} defaulted: {e['written_off']} written off "
                                                     f"({e['reason']})."))
        elif t == "bankruptcy":
            first.setdefault("bankruptcy", (e["round"], f"{e['agent']} went bankrupt."
                                                        + (f" Epitaph: \"{e['epitaph']}\"" if e.get("epitaph") else "")))
        elif t == "simulation_ended":
            bal = e["final_balances"]
            w = max(bal, key=bal.get)
            first["end"] = (e["rounds_played"], f"Run ended ({e['reason']}). Richest: {w} with {bal[w]}. "
                                                f"Ledger balanced: {e['reconciliation']['ok']}.")
    return sorted(first.values(), key=lambda x: x[0])


def find_seed(limit: int = 200) -> int | None:
    """First seed (from 1) whose run contains every required beat."""
    with tempfile.TemporaryDirectory() as tmp:
        for seed in range(1, limit + 1):
            if not missing_beats(_simulate(seed, Path(tmp))):
                return seed
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=DEMO_SEED)
    ap.add_argument("--find", action="store_true", help="search for a seed with every story beat")
    ap.add_argument("--install-replay", action="store_true", help="copy the run into data/replays for the dashboard")
    ap.add_argument("--out", type=Path, default=None, help="scratch directory (default: a temporary one)")
    args = ap.parse_args(argv)
    if args.find:
        seed = find_seed()
        print(f"working seed: {seed}" if seed else "no seed found in 1..200; adjust DEMO_CONFIG")
        return seed
    real_replays = events.REPLAY_DIR
    tmp = tempfile.TemporaryDirectory()
    out = args.out or Path(tmp.name)
    out.mkdir(parents=True, exist_ok=True)
    evs = _simulate(args.seed, out)
    gaps = missing_beats(evs)
    print(f"Demo scenario (mock mode) - seed {args.seed}, config {DEMO_CONFIG}\n")
    for rnd, text in narrative(evs):
        print(f"  round {rnd:>2}: {text}")
    if gaps:
        print(f"\nWARNING: this seed no longer shows: {gaps}. Run `python -m training.demo --find`.")
    if args.install_replay:
        real_replays.mkdir(parents=True, exist_ok=True)
        dest = real_replays / f"demo-seed{args.seed}.jsonl"
        shutil.copy(out / "events.jsonl", dest)
        print(f"\ninstalled replay: {dest} (pick it in the dashboard's Replay menu)")
    tmp.cleanup()
    return evs


if __name__ == "__main__":
    main()
