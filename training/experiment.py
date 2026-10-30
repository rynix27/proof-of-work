"""
Reproducible economy experiments with honest statistics.

SWEEP one parameter across conditions (paired seeds):
    python -m training.experiment --param interest_rate --values 0,0.15,0.4 \
        --trials 10 --rounds 40 --seed 100 --out data/experiments/interest
    python -m training.experiment --param strategies --values baseline,conservative,aggressive ...
    python -m training.experiment --param bounties --values 8:20:50,4:10:25 ...

TOURNAMENT -- strategies play AGAINST each other in one economy, with their seats rotated
every trial so position/tie-break order cannot favour any strategy:
    python -m training.experiment --tournament --trials 10 --rounds 60 --seed 100 --out data/experiments/t1

DESIGN
  * Trial i of EVERY condition uses seed (--seed + i): conditions face the same random
    task sequence ("paired"), so differences are not due to a lucky draw.
  * Metrics are computed by core.state.build_state -- the same code the dashboard uses.
  * Each metric reports n, mean, median, std and a 95% Student-t interval. Differences vs
    the control are PAIRED and flagged `detectable` only when n >= 5 and the interval
    excludes 0. Otherwise the report says "no detectable difference", never a trend claim.
  * Runs write only inside --out; they never touch the live event log, leaderboard or replays.

INTERPRETATION WARNING: in mock mode the "agents" are a random fake model whose accuracy
does not depend on tier, strategy or difficulty-awareness. Results therefore exercise the
economy RULES and this pipeline; they are NOT evidence about LLM behaviour. Live mode costs
real money and needs --confirm-live.
"""
import argparse
import csv
import json
from pathlib import Path

from core import config, events, model_client
from core.orchestrator import run_simulation
from core.simconfig import SimulationConfig
from core.state import build_state
from core.stats import MIN_N_FOR_CLAIMS, describe, paired_difference
from core.strategies import STRATEGIES

METRICS = ("economic_profit", "accuracy", "compute_spent", "escalations", "loans_made", "loans_defaulted",
           "bankruptcies", "wealth_spread", "repayment_rate", "compute_per_correct", "task_completion_rate")
HIGHER_IS_BETTER = {"economic_profit": True, "accuracy": True, "task_completion_rate": True, "repayment_rate": True,
                    "compute_per_correct": False, "compute_spent": False, "loans_defaulted": False,
                    "bankruptcies": False, "wealth_spread": False}


def parse_value(param: str, raw: str):
    raw = raw.strip()
    if param == "strategies":
        return tuple(raw.split("+")) if "+" in raw else raw
    if param in ("bounties", "task_weights"):
        parts = [float(x) for x in raw.split(":")]
        return [int(x) if x.is_integer() and param == "bounties" else x for x in parts]
    f = float(raw)
    return int(f) if f.is_integer() and param not in ("interest_rate", "escalation_cutoff") else f


def _isolate(out_dir: Path):
    events.LOG_PATH = out_dir / "events.jsonl"
    events.REPLAY_DIR = out_dir / "replays"
    events.LEADERBOARD_PATH = out_dir / "leaderboard.jsonl"
    model_client.TRAINING_LOG_PATH = out_dir / "calls.jsonl"


def run_trial(overrides: dict, out_dir: Path) -> dict:
    _isolate(out_dir)
    run_simulation(round_delay=0, verbose=False, config_overrides=overrides)
    evs = events.read_all()
    st = build_state(evs)
    s = st["stats"]
    end = [e for e in evs if e["type"] == "simulation_ended"][-1]
    balances = list(end["final_balances"].values())
    return {
        "seed": end["seed"], "rounds": end["rounds_played"], "reason": end["reason"],
        "bankruptcies": len(st["bankrupt"]), "accuracy": s["success_rate"],
        "compute_spent": s["compute_burned"], "rewards": s["rewards_total"],
        "economic_profit": s["economic_profit"],
        "escalations": sum(a["escalations"] for a in end["agent_stats"].values()),
        "loans_made": s["loans_made"], "loans_defaulted": s["loans_defaulted"],
        "written_off": end["loans"]["written_off"], "repayment_rate": s["repayment_rate"],
        "wealth_spread": max(balances) - min(balances), "compute_per_correct": s["compute_per_correct"],
        "task_completion_rate": s["task_completion_rate"], "ledger_ok": end["reconciliation"]["ok"],
        "_agents": end["agent_stats"],
    }


def summarize(rows: list[dict]) -> dict:
    return {"n": len(rows), "ledger_all_ok": all(r["ledger_ok"] for r in rows),
            "metrics": {m: describe([r.get(m) for r in rows]) for m in METRICS}}


def compare_to_control(cond_rows: dict, control_key: str) -> dict:
    """Paired differences of each condition vs the control for every metric."""
    ctrl = cond_rows[control_key]
    out = {}
    for key, rows in cond_rows.items():
        if key == control_key:
            continue
        out[key] = {m: paired_difference([r.get(m) for r in rows], [r.get(m) for r in ctrl]) for m in METRICS}
    return out


def analysis_markdown(summary: dict) -> str:
    """Plain statements generated ONLY from the numbers, each gated on n and the interval."""
    lines = [f"# Experiment analysis: `{summary['param']}`", "",
             f"Mode: **{summary['mode']}** - {summary['trials']} paired trials x {summary['rounds']} rounds, "
             f"base seed {summary['base_seed']}. Control condition: `{summary['control']}`.", "",
             f"> {summary['warning']}", ""]
    if summary["trials"] < MIN_N_FOR_CLAIMS:
        lines += [f"**Only {summary['trials']} trial(s): fewer than {MIN_N_FOR_CLAIMS}, so no difference is "
                  "reported as detectable.** Re-run with more trials.", ""]
    found = False
    for key, comp in summary["comparisons"].items():
        for m, d in comp.items():
            if d and d["detectable"]:
                found = True
                if m in HIGHER_IS_BETTER:
                    verdict = "favourable" if (d["direction"] == "higher") == HIGHER_IS_BETTER[m] else "unfavourable"
                else:
                    verdict = "descriptive: no good/bad direction"       # e.g. escalations, loans_made
                lines.append(f"- `{key}` vs `{summary['control']}`: **{m}** is {d['direction']} by "
                             f"{d['mean_diff']:+.3g} (95% CI {d['ci95'][0]:+.3g} to {d['ci95'][1]:+.3g}, "
                             f"n={d['n']}, paired) - {verdict}.")
    if not found:
        lines.append("- No metric showed a detectable difference from the control at this sample size "
                     "(a 95% paired interval that excludes 0, with n >= 5).")
    lines += ["", "Absence of a detectable difference is not evidence of no difference; it may reflect too few "
              "trials. Intervals are Student-t on paired differences."]
    return "\n".join(lines)


def _write(out: Path, rows: list[dict], summary: dict):
    flat = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    with open(out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "analysis.md").write_text(analysis_markdown(summary))


def run_sweep(args) -> dict:
    values = [parse_value(args.param, v) for v in args.values.split(",")]
    for v in values:                                    # validate every condition up front
        SimulationConfig.from_overrides({args.param: v, "max_rounds": args.rounds})
    keys = [json.dumps(v) if not isinstance(v, str) else v for v in values]
    cond_rows, all_rows = {}, []
    for key, v in zip(keys, values):
        rows = []
        for i in range(args.trials):
            r = run_trial({args.param: v, "max_rounds": args.rounds, "seed": args.seed + i}, args.out)
            r.update({"param": args.param, "value": key})
            rows.append(r)
        cond_rows[key] = rows
        all_rows += rows
    control = args.control or keys[0]
    if control not in cond_rows:
        raise SystemExit(f"--control {control!r} is not one of the swept values {keys}")
    summary = {"kind": "sweep", "param": args.param, "mode": config.mode(), "trials": args.trials,
               "rounds": args.rounds, "base_seed": args.seed, "control": control,
               "conditions": {k: summarize(r) for k, r in cond_rows.items()},
               "comparisons": compare_to_control(cond_rows, control),
               "warning": _warning()}
    _write(args.out, all_rows, summary)
    return summary


def run_tournament(args) -> dict:
    names = [n for n in ("conservative", "aggressive", "cost_efficient", "accuracy_focused", "lender")]
    rows, per_strategy = [], {n: [] for n in names}
    for i in range(args.trials):
        rot = i % len(names)
        seating = names[rot:] + names[:rot]               # same strategies, different seats each trial
        r = run_trial({"strategies": seating, "max_rounds": args.rounds, "seed": args.seed + i, "num_agents": 5},
                      args.out)
        r.update({"param": "tournament", "value": "+".join(seating)})
        rows.append(r)
        for a in r["_agents"].values():
            per_strategy[a["strategy"]].append({"economic_profit": a["economic_profit"], "accuracy": a["accuracy"],
                                                "compute_spent": a["compute_spent"], "final_balance": a["final_balance"],
                                                "escalations": a["escalations"], "bankrupt": int(a["bankrupt"]),
                                                "loans_issued": a["loans_issued"], "trial": i})
    metrics = ("economic_profit", "accuracy", "compute_spent", "final_balance", "escalations", "bankrupt", "loans_issued")
    table, comparisons = {}, {}
    for n in names:
        table[n] = {m: describe([x[m] for x in per_strategy[n]]) for m in metrics}
        # paired against the field: strategy value minus the mean of ALL strategies in the same trial
        comparisons[n] = {}
        for m in ("economic_profit", "final_balance", "accuracy"):
            field = [sum(per_strategy[o][t][m] for o in names if per_strategy[o][t][m] is not None) /
                     max(1, sum(1 for o in names if per_strategy[o][t][m] is not None)) for t in range(args.trials)]
            comparisons[n][m] = paired_difference([x[m] for x in per_strategy[n]], field)
    summary = {"kind": "tournament", "param": "strategies (tournament, seats rotated)", "mode": config.mode(),
               "trials": args.trials, "rounds": args.rounds, "base_seed": args.seed, "control": "field average",
               "strategies": table, "comparisons": comparisons,
               "ledger_all_ok": all(r["ledger_ok"] for r in rows), "warning": _warning()}
    _write(args.out, rows, summary)
    (args.out / "analysis.md").write_text(_tournament_markdown(summary))
    return summary


def _tournament_markdown(s: dict) -> str:
    lines = [f"# Strategy tournament", "",
             f"Mode: **{s['mode']}** - {s['trials']} trials x {s['rounds']} rounds, seats rotated each trial.", "",
             f"> {s['warning']}", "", "| strategy | mean economic profit | 95% CI | mean accuracy | bankruptcy rate |", "|---|---|---|---|---|"]
    for n, t in s["strategies"].items():
        p, a, b = t["economic_profit"], t["accuracy"], t["bankrupt"]
        ci = f"{p['ci95'][0]:.1f} to {p['ci95'][1]:.1f}" if p and p["ci95"] else "n/a"
        lines.append(f"| {n} | {p['mean']:.1f} | {ci} | {a['mean'] if a else 'n/a'} | {b['mean'] if b else 'n/a'} |")
    lines.append("")
    if s["trials"] < MIN_N_FOR_CLAIMS:
        lines.append(f"**Only {s['trials']} trial(s) (< {MIN_N_FOR_CLAIMS}): no ranking is claimed.**")
    else:
        hit = False
        for n, comp in s["comparisons"].items():
            d = comp["economic_profit"]
            if d and d["detectable"]:
                hit = True
                lines.append(f"- `{n}` earned {d['direction']} economic profit than the field average "
                             f"({d['mean_diff']:+.1f}, 95% CI {d['ci95'][0]:+.1f} to {d['ci95'][1]:+.1f}, n={d['n']}).")
        if not hit:
            lines.append("- No strategy's economic profit differed detectably from the field average.")
    return "\n".join(lines)


def _warning() -> str:
    return ("Mock mode: results describe the fake model and the economy rules, not LLM behaviour."
            if config.USE_FAKE_MODEL else "Live mode: stochastic; compare means with their intervals and n.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--param", help="SimulationConfig field to sweep, e.g. interest_rate")
    ap.add_argument("--values", help="comma-separated values (bounties: 8:20:50; strategies: a+b+c for a list)")
    ap.add_argument("--control", help="control condition (default: first value)")
    ap.add_argument("--tournament", action="store_true", help="run all strategies against each other")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--confirm-live", action="store_true")
    args = ap.parse_args(argv)
    if not args.tournament and not (args.param and args.values):
        ap.error("give --param and --values, or --tournament")
    if args.trials < 1:
        ap.error("--trials must be >= 1")
    if not config.USE_FAKE_MODEL and not args.confirm_live:
        raise SystemExit("Live mode spends real model calls. Re-run with --confirm-live to proceed.")
    args.out.mkdir(parents=True, exist_ok=True)
    summary = run_tournament(args) if args.tournament else run_sweep(args)
    print((args.out / "analysis.md").read_text())
    print(f"\nwrote {args.out/'runs.csv'}, {args.out/'summary.json'}, {args.out/'analysis.md'}")
    return summary


if __name__ == "__main__":
    main()
