"""
Pure function: list of events -> dashboard state. No I/O, no globals, so it
works identically for a live run, a saved replay and an exported file, and can
never modify the source events.

It is tolerant by design: non-dict entries, events without a type, and events
missing optional fields (older schema v1 logs) are skipped or defaulted, never
raised on. Every metric returns None rather than a made-up number when its
inputs don't exist.

All compute numbers are SIMULATED compute-seconds (see core/economy.py), not
provider billing.
"""
from core import config
from core.calibration import calibration_report
from core.stats import gini, safe_div


def _i(v, default=0):
    return v if isinstance(v, int) and not isinstance(v, bool) else default


def build_state(all_events: list) -> dict:
    balances: dict = {}
    bankrupt: list = []
    history: list = []
    started: dict = {}
    per: dict = {}          # agent -> counters
    loans: dict = {}        # loan_id -> loan dict
    last_notable = None
    ended = None
    n_events = n_skipped = 0
    counts = dict(arrived=0, unclaimed=0, solves=0, graded=0, correct=0, escalations=0,
                  decisions=0, invalid_decisions=0, invalid_answers=0, compute=0,
                  rewards=0, nano_calls=0, ultra_calls=0, call_errors=0, charged_errors=0,
                  nano_solves=0)

    tiers: dict = {}                 # tier -> graded results & economics (escalation effectiveness)
    series: list = []                # per-round cumulative points for the charts
    usage = dict(calls=0, calls_with_usage=0, input_tokens=0, output_tokens=0, cost=0.0, cost_known=True, any=False)
    last_auction = None
    decisions_by_round: dict = {}
    run_cum = dict(loan_volume=0)
    extra = dict(policy_overrides=0, escalation_denied=0, solves_invalid=0)

    def tier_rec(t):
        return tiers.setdefault(t, dict(solves=0, graded=0, correct=0, compute=0, rewards=0))

    def agent(name):
        return per.setdefault(name, dict(attempts=0, graded=0, correct=0, compute=0,
                                         rewards=0, escalations=0, invalid=0))

    for e in all_events:
        if not isinstance(e, dict) or not isinstance(e.get("type"), str):
            n_skipped += 1
            continue
        n_events += 1
        t = e["type"]
        if t == "simulation_started":
            started = e
            names = e.get("agents") or []
            sb = _i(e.get("starting_balance"), config.STARTING_BALANCE)
            balances = {n: sb for n in names}
        elif t == "task_arrived":
            counts["arrived"] += 1
        elif t == "task_unclaimed":
            counts["unclaimed"] += 1
        elif t == "decision":
            counts["decisions"] += 1
            counts["nano_calls"] += 1
            if e.get("policy_reasons"):
                extra["policy_overrides"] += 1
            decisions_by_round.setdefault(e.get("round"), []).append(e)
        elif t == "escalation_denied":
            extra["escalation_denied"] += 1
        elif t == "auction_resolved":
            last_auction = e
        elif t == "usage":
            u = e.get("usage") if isinstance(e.get("usage"), dict) else {}
            usage["any"] = True
            usage["calls"] += _i(u.get("calls"))
            usage["calls_with_usage"] += _i(u.get("calls_with_usage"))
            usage["input_tokens"] += _i(u.get("input_tokens"))
            usage["output_tokens"] += _i(u.get("output_tokens"))
            c = e.get("estimated_cost_usd")
            if isinstance(c, (int, float)) and not isinstance(c, bool):
                usage["cost"] += c
            else:
                usage["cost_known"] = False
        elif t == "decision_invalid":
            counts["invalid_decisions"] += 1
            if e.get("agent"):
                agent(e["agent"])["invalid"] += 1
        elif t == "answer_invalid":
            counts["invalid_answers"] += 1
        elif t == "escalation":
            counts["escalations"] += 1
            if e.get("agent"):
                agent(e["agent"])["escalations"] += 1
        elif t == "call_error":
            counts["call_errors"] += 1
            charged = _i(e.get("charged"))
            if charged:
                counts["charged_errors"] += 1
                counts["compute"] += charged
                if e.get("agent"):
                    agent(e["agent"])["compute"] += charged
            if e.get("agent") and isinstance(e.get("new_balance"), int):
                balances[e["agent"]] = e["new_balance"]
        elif t == "task_resolved":
            name = e.get("agent")
            if name is None:
                continue
            tier = e.get("tier")
            cost = e.get("cost")
            if not isinstance(cost, int):  # schema v1: derive from tier
                cost = config.ULTRA_COST if tier == "ultra" else config.NANO_COST
            counts["solves"] += 1
            counts["compute"] += cost
            if tier == "ultra":
                counts["ultra_calls"] += 1
            else:
                counts["nano_calls"] += 1
                counts["nano_solves"] += 1
            tr = tier_rec(tier if tier in ("nano", "ultra") else "nano")
            tr["solves"] += 1
            tr["compute"] += cost
            if e.get("valid_response") is False:
                extra["solves_invalid"] += 1
            a = agent(name)
            a["attempts"] += 1
            a["compute"] += cost
            if isinstance(e.get("new_balance"), int):
                balances[name] = e["new_balance"]
            if not e.get("exhibition"):
                # exhibition ("ask the economy") tasks have no ground truth, so
                # they are excluded from accuracy entirely
                counts["graded"] += 1
                a["graded"] += 1
                tr["graded"] += 1
                if e.get("correct"):
                    counts["correct"] += 1
                    a["correct"] += 1
                    tr["correct"] += 1
                    b = _i(e.get("bounty"))
                    counts["rewards"] += b
                    a["rewards"] += b
                    tr["rewards"] += b
        elif t == "loan_made":
            lid = e.get("loan_id") or f"legacy-{len(loans)}"
            principal = _i(e.get("principal", e.get("amount")))
            rate = e.get("interest_rate") if isinstance(e.get("interest_rate"), (int, float)) else 0
            total = _i(e.get("total_owed"), principal)
            run_cum["loan_volume"] += principal
            loans[lid] = dict(id=lid, lender=e.get("lender"), borrower=e.get("borrower"),
                              principal=principal, interest_rate=rate, total_owed=total,
                              outstanding=_i(e.get("total_owed"), 0) or 0, repaid=0,
                              status="active" if "loan_id" in e else "untracked",
                              created_round=e.get("round"), due_round=e.get("due_round"))
            for who, key in (("lender", "lender_balance"), ("borrower", "borrower_balance")):
                if e.get(who) and isinstance(e.get(key), int):
                    balances[e[who]] = e[key]
        elif t == "loan_repayment":
            l = loans.get(e.get("loan_id"))
            if l:
                l["repaid"] += _i(e.get("amount"))
                l["outstanding"] = _i(e.get("outstanding"))
                l["status"] = e.get("status", l["status"])
            for who, key in (("lender", "lender_balance"), ("borrower", "borrower_balance")):
                if e.get(who) and isinstance(e.get(key), int):
                    balances[e[who]] = e[key]
        elif t == "loan_default":
            l = loans.get(e.get("loan_id"))
            if l:
                l["status"] = "defaulted"
                l["written_off"] = _i(e.get("written_off"))
                l["outstanding"] = 0
        elif t == "round_snapshot":
            if isinstance(e.get("balances"), dict):
                balances = dict(e["balances"])
                history.append({"round": e.get("round"), "balances": e["balances"]})
            bankrupt = list(e.get("bankrupt") or [])
            series.append({"round": e.get("round"), "tasks": counts["arrived"], "graded": counts["graded"],
                           "correct": counts["correct"],
                           "accuracy": safe_div(counts["correct"], counts["graded"]),
                           "compute": counts["compute"], "rewards": counts["rewards"],
                           "loan_volume": run_cum["loan_volume"],
                           "bankrupt": len(bankrupt)})
        elif t == "bankruptcy":
            last_notable = {"kind": "bankruptcy", "agent": e.get("agent"), "round": e.get("round"),
                            "ts": e.get("ts"), "epitaph": e.get("epitaph")}
        elif t == "monopoly":
            last_notable = {"kind": "monopoly", "agent": e.get("winner"), "round": e.get("round"),
                            "ts": e.get("ts")}
        elif t == "simulation_ended":
            ended = e
            if isinstance(e.get("final_balances"), dict):
                balances = e["final_balances"]

    valid = [e for e in all_events if isinstance(e, dict) and isinstance(e.get("type"), str)]
    latest_task = next((e for e in reversed(valid) if e["type"] == "task_arrived"), None)
    latest_commentary = next((e for e in reversed(valid) if e["type"] == "commentary"), None)

    start_bal = _i(started.get("starting_balance"), config.STARTING_BALANCE)
    loan_list = list(loans.values())
    settled = [l for l in loan_list if l["status"] in ("repaid", "defaulted")]
    owed_settled = sum(l["total_owed"] for l in settled)
    loan_stats = {
        "loans_made": len(loan_list),
        "loan_volume": sum(l["principal"] for l in loan_list),
        "outstanding_debt": sum(l["outstanding"] for l in loan_list if l["status"] == "active"),
        "loans_repaid": sum(1 for l in loan_list if l["status"] == "repaid"),
        "loans_defaulted": sum(1 for l in loan_list if l["status"] == "defaulted"),
        "repayment_rate": safe_div(sum(l["repaid"] for l in settled), owed_settled),
    }

    strategies = started.get("strategies") if isinstance(started.get("strategies"), dict) else {}
    debt, lent, issued = {}, {}, {}
    for l in loan_list:
        issued[l["lender"]] = issued.get(l["lender"], 0) + 1
        if l["status"] == "active":
            debt[l["borrower"]] = debt.get(l["borrower"], 0) + l["outstanding"]
            lent[l["lender"]] = lent.get(l["lender"], 0) + l["outstanding"]

    agents_table = {}
    for name in balances:
        a = per.get(name, {})
        bal = balances.get(name, 0)
        agents_table[name] = {
            "balance": bal, "net_pl": bal - start_bal, "bankrupt": name in bankrupt,
            "solved": a.get("correct", 0), "attempts": a.get("attempts", 0),
            "accuracy": safe_div(a.get("correct", 0), a.get("graded", 0)),
            "compute_spent": a.get("compute", 0), "rewards": a.get("rewards", 0),
            "efficiency": safe_div(a.get("rewards", 0), a.get("compute", 0)),
            "escalations": a.get("escalations", 0), "invalid_decisions": a.get("invalid", 0),
            "strategy": strategies.get(name), "economic_profit": a.get("rewards", 0) - a.get("compute", 0),
            "loans_issued": issued.get(name, 0), "outstanding_debt": debt.get(name, 0),
            "lent_out": lent.get(name, 0),
            "escalation_rate": safe_div(a.get("escalations", 0), a.get("attempts", 0)),
        }

    def ranking(key, reverse=True, min_attempts=0):
        rows = [(n, v[key]) for n, v in agents_table.items()
                if v[key] is not None and v["attempts"] >= min_attempts]
        return [{"agent": n, "value": v} for n, v in sorted(rows, key=lambda r: r[1], reverse=reverse)]

    total_decisions = counts["decisions"] + counts["invalid_decisions"]
    stats = {
        # original keys (kept for compatibility)
        "nano_calls": counts["nano_calls"], "ultra_calls": counts["ultra_calls"],
        "compute_burned": counts["compute"], "tasks_resolved": counts["graded"],
        "tasks_correct": counts["correct"],
        "success_rate": safe_div(counts["correct"], counts["graded"]),
        # additions
        "tasks_total": counts["arrived"], "tasks_unclaimed": counts["unclaimed"],
        "tasks_attempted": counts["solves"],
        "rewards_total": counts["rewards"],
        "compute_per_correct": safe_div(counts["compute"], counts["correct"], 2),
        "escalation_rate": safe_div(counts["escalations"], counts["solves"]),
        "invalid_decision_rate": safe_div(counts["invalid_decisions"], total_decisions),
        "call_errors": counts["call_errors"],
        # --- additional rates (definitions: docs/METRICS.md) ---
        "task_completion_rate": safe_div(counts["solves"], counts["arrived"]),
        "economic_profit": counts["rewards"] - counts["compute"],
        "bankruptcy_rate": safe_div(len(bankrupt), len(balances)),
        "loan_default_rate": safe_div(loan_stats["loans_defaulted"],
                                      loan_stats["loans_repaid"] + loan_stats["loans_defaulted"]),
        "policy_override_rate": safe_div(extra["policy_overrides"], counts["decisions"]),
        "escalations_denied": extra["escalation_denied"],
        "invalid_answer_rate": safe_div(counts["invalid_answers"], counts["solves"]),
        **loan_stats,
    }
    tier_table = {}
    for t, r in tiers.items():
        net = r["rewards"] - r["compute"]
        tier_table[t] = {**r, "accuracy": safe_div(r["correct"], r["graded"]),
                         "net_profit": net, "avg_cost": safe_div(r["compute"], r["solves"], 2),
                         "profit_per_solve": safe_div(net, r["solves"], 2)}
    provider_usage = None
    if usage["any"]:
        provider_usage = {"calls": usage["calls"], "calls_with_usage": usage["calls_with_usage"],
                          "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
                          "estimated_cost_usd": round(usage["cost"], 6) if usage["cost_known"] else None,
                          "note": "provider-reported tokens; cost is an estimate from prices YOU configured "
                                  "(None if unset). Separate from simulated compute-seconds."}
    last_round = max((r for r in decisions_by_round if isinstance(r, int)), default=None)

    if ended is not None:
        status = "ended"
    elif not valid:
        status = "idle"
    else:
        status = "running"

    return {
        "run_id": (valid[0].get("run_id") if valid else None),
        "mode": started.get("mode"),               # "mock" | "live" | None (old logs)
        "seed": started.get("seed"),
        "sim_config": started.get("config"),
        "models": started.get("models"),
        "status": status,
        "ended_reason": ended.get("reason") if ended else None,
        "starting_balance": start_bal,
        "balances": balances,
        "bankrupt": bankrupt,
        "tasks_won": {n: v["solved"] for n, v in agents_table.items()},
        "agents": agents_table,
        "rankings": {
            "balance": ranking("balance"),
            "accuracy": ranking("accuracy", min_attempts=3),
            "profit": ranking("net_pl"),
            "efficiency": ranking("efficiency", min_attempts=3),
        },
        "recent_events": valid[-60:],
        "latest_task": latest_task,
        "latest_commentary": latest_commentary,
        "round_count": counts["arrived"],
        "history": history,
        "loans": loan_list[-25:],
        "last_notable": last_notable,
        "gini": round(gini(list(balances.values())), 3) if balances else 0.0,
        "stats": stats,
        "tiers": tier_table,
        "series": series[-300:],
        "calibration": calibration_report(valid),
        "provider_usage": provider_usage,
        "last_auction": last_auction,
        "latest_decisions": decisions_by_round.get(last_round, []) if last_round is not None else [],
        "data_quality": {"events": n_events, "skipped_malformed": n_skipped},
    }
