"""
Run report: a plain-Markdown summary of one run (live or replay), generated ONLY from the
derived state -- the same numbers the dashboard shows -- so the two can never disagree.

It states what was measured and what was not. Mock-mode runs are labelled as such at the top
and in the limitations; every sample-size-sensitive number carries its n.
"""


def _pct(x):
    return "n/a" if x is None else f"{x * 100:.0f}%"


def _num(x, d=2):
    return "n/a" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))


def build_report(state: dict) -> str:
    s = state.get("stats", {})
    mode = state.get("mode") or "unknown"
    cfg = state.get("sim_config") or {}
    L = [f"# Proof of Work - run report", ""]
    if mode == "mock":
        L += ["> **MOCK RUN.** A seeded fake model played this run. It demonstrates the economy mechanics; "
              "it says nothing about how real language models behave.", ""]
    L += [f"- Run id: `{state.get('run_id')}`   Status: **{state.get('status')}**"
          f"{' (' + state['ended_reason'] + ')' if state.get('ended_reason') else ''}",
          f"- Mode: **{mode}**   Seed: `{state.get('seed')}`",
          f"- Models: nano `{(state.get('models') or {}).get('nano')}`, ultra `{(state.get('models') or {}).get('ultra')}`",
          f"- Rounds: {state.get('round_count')}   Agents: {len(state.get('agents', {}))}", ""]

    L += ["## Configuration", ""]
    for k in ("starting_balance", "nano_cost", "ultra_cost", "interest_rate", "loan_term_rounds",
              "escalation_cutoff", "max_loan_amount", "bounties", "strategies"):
        if k in cfg:
            L.append(f"- {k}: `{cfg[k]}`")
    L.append("")

    L += ["## Outcomes (simulated economy)", "",
          f"- Tasks: {s.get('tasks_total')} arrived, {s.get('tasks_attempted')} attempted, "
          f"{s.get('tasks_unclaimed')} unclaimed (completion {_pct(s.get('task_completion_rate'))})",
          f"- Accuracy on graded tasks: **{_pct(s.get('success_rate'))}** "
          f"({s.get('tasks_correct')}/{s.get('tasks_resolved')}); ungraded questions are excluded",
          f"- Simulated compute spent: {s.get('compute_burned')}; rewards paid: {s.get('rewards_total')}; "
          f"economic profit: **{s.get('economic_profit')}**; compute per correct answer: {_num(s.get('compute_per_correct'))}",
          f"- Bankruptcy rate: {_pct(s.get('bankruptcy_rate'))}; inequality (Gini): {_num(state.get('gini'))}",
          f"- Policy overrides of the model's proposal: {_pct(s.get('policy_override_rate'))} of decisions; "
          f"escalations denied: {s.get('escalations_denied')}", ""]

    L += ["## Agents", "", "| agent | strategy | balance | P/L | econ. profit | accuracy | attempts | compute | loans issued | debt |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, a in state.get("agents", {}).items():
        L.append(f"| {n}{' (bust)' if a.get('bankrupt') else ''} | {a.get('strategy') or '-'} | {a.get('balance')} | "
                 f"{a.get('net_pl'):+d} | {a.get('economic_profit')} | {_pct(a.get('accuracy'))} | {a.get('attempts')} | "
                 f"{a.get('compute_spent')} | {a.get('loans_issued')} | {a.get('outstanding_debt')} |")
    L += ["", "P/L is the change in balance (includes loan transfers and interest); economic profit is rewards minus compute.", ""]

    tiers = state.get("tiers") or {}
    if tiers:
        L += ["## Escalation effectiveness", "", "| tier | solves | graded | accuracy | compute | rewards | net profit |", "|---|---|---|---|---|---|---|"]
        for t, r in sorted(tiers.items()):
            L.append(f"| {t} | {r['solves']} | {r['graded']} | {_pct(r['accuracy'])} | {r['compute']} | {r['rewards']} | {r['net_profit']} |")
        small = min((r["graded"] for r in tiers.values()), default=0)
        L += ["", f"Smallest tier sample: n={small}. Differences between tiers on samples this size are "
                  "usually noise; escalation is also chosen *because* confidence was low, which confounds the comparison.", ""]

    L += ["## Loans", "",
          f"- Made: {s.get('loans_made')}, repaid: {s.get('loans_repaid')}, defaulted: {s.get('loans_defaulted')} "
          f"(default rate {_pct(s.get('loan_default_rate'))}); repayment rate {_pct(s.get('repayment_rate'))}",
          f"- Volume: {s.get('loan_volume')}; outstanding debt: {s.get('outstanding_debt')}", ""]

    cal = (state.get("calibration") or {}).get("self_reported") or {}
    L += ["## Confidence calibration", "",
          f"- Self-reported confidence vs outcomes: n={cal.get('n')}, Brier {_num(cal.get('brier'), 3)}, ECE {_num(cal.get('ece'), 3)}"
          f"{'' if cal.get('reliable') else ' - **' + str(cal.get('note') or 'insufficient data') + '**'}",
          "- The model's confidence is measured, not assumed to be calibrated.", ""]

    L += ["## Cost accounting", "",
          f"- **Simulated** compute-seconds (the economy's currency): {s.get('compute_burned')}. This is not money."]
    u = state.get("provider_usage")
    if u:
        L.append(f"- **Provider-reported** usage: {u['input_tokens']} input / {u['output_tokens']} output tokens over "
                 f"{u['calls_with_usage']}/{u['calls']} calls with usage data; estimated cost "
                 f"{'$' + format(u['estimated_cost_usd'], '.4f') if u['estimated_cost_usd'] is not None else 'unknown (prices not configured)'}.")
    else:
        L.append("- No provider usage recorded" + (" (mock mode makes no provider calls)." if mode == "mock" else "."))
    L.append("")

    dq = state.get("data_quality") or {}
    L += ["## Limitations", "",
          f"- Events analysed: {dq.get('events')}; malformed entries skipped: {dq.get('skipped_malformed')}.",
          "- Single run: no confidence interval can be given from one run; use `training.experiment` for comparisons.",
          "- Bids are ranked by the model's self-reported confidence, which can be gamed; bid `amount` is ignored.",
          "- The task bank is small and fixed; accuracy depends on it."]
    if mode == "mock":
        L.append("- Mock mode: the fake model's accuracy does not depend on tier or strategy, so effort/escalation cannot pay off here.")
    return "\n".join(L) + "\n"
