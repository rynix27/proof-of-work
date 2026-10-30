# Metric definitions

Every number on the dashboard, in `/state`, in `/export/summary.json`, in the run report and in
experiment output is computed by **one** function, `core.state.build_state` (experiments call it
too), from the event log. The frontend only displays; it never recomputes. All "compute" figures
are **simulated compute-seconds**, an in-game currency, not money.

A ratio whose denominator is zero is **`None` / "n/a"**, never `0`.

| metric | definition | excludes / notes |
|---|---|---|
| `tasks_total` | `task_arrived` events | includes unclaimed and ungraded tasks |
| `tasks_attempted` | solves that completed (`task_resolved`) | includes ungraded questions |
| `task_completion_rate` | attempted / tasks_total | the rest were unclaimed or failed provider-side |
| `tasks_resolved` | graded solves | **excludes** ungraded "ask the economy" questions |
| `success_rate` (accuracy) | correct / graded solves | an unusable answer counts as graded **wrong** |
| `compute_burned` | sum of `cost` on solves + charged failed calls | decisions are free |
| `rewards_total` | sum of bounties for correct graded answers | |
| `economic_profit` | rewards_total − compute_burned | economy-level; per agent: its rewards − its compute |
| `net_pl` (per agent) | final balance − starting balance | **includes** loan transfers and interest, so it differs from `economic_profit` |
| `compute_per_correct` | compute_burned / correct answers | all compute counts, including on wrong answers |
| `escalation_rate` | escalations / solves | |
| `escalations_denied` | `escalation_denied` events | confidence was low but the strategy refused (reason recorded) |
| `policy_override_rate` | decisions where the strategy changed or limited the model's proposal / all valid decisions | |
| `invalid_decision_rate` | rejected decisions / (valid + rejected) | model output that failed validation |
| `invalid_answer_rate` | unusable answers / solves | |
| `bankruptcy_rate` | bankrupt agents / agents (at the last snapshot) | |
| `loans_made / volume` | count / sum of principal | |
| `loan_default_rate` | defaulted / (repaid + defaulted) | active loans are excluded (not yet resolved) |
| `repayment_rate` | repaid amount / total owed, on **settled** loans | `None` until a loan settles |
| `outstanding_debt` | sum of `outstanding` on active loans | |
| `gini` | Gini coefficient of final/current balances | 0 = equal, 1 = one agent holds everything |
| tier table | per `nano` / `ultra`: solves, graded, correct, compute, rewards, net_profit, profit/solve | see caveat below |
| `series` | cumulative values at each `round_snapshot` | for the trend charts; gaps are `null`, not 0 |
| calibration | per bin: mean stated confidence vs observed accuracy; **Brier** = mean((p − outcome)²); **ECE** = Σ (n_b/N)·\|acc_b − conf_b\| | graded results only; flagged *indicative* below n = 30 |
| `provider_usage` | sum of provider-reported prompt/completion tokens | `None` in mock mode; estimated cost uses **your** configured prices, else `None` |

## Win, balance and skill are different things
The leaderboard's *win* is "highest final balance" (**wealth**). It can reward luck or lending
rather than problem-solving, so rankings are also offered by **accuracy**, **average P/L**,
**reward per compute** and **lowest cost per correct**.

## Caveats that apply to the numbers
- **Escalation effectiveness is confounded.** Agents escalate *because* their confidence was
  low, so Ultra solves are on harder-feeling tasks. And tier samples are small. Treat the tier
  table as descriptive, not causal. In mock mode accuracy does not depend on tier at all.
- **Self-reported confidence is not trusted.** Bids are ranked by it (a design choice that is
  gameable), the policy gives it limited weight, and calibration is measured, not assumed.
- **A single run has no confidence interval.** Compare conditions with `training.experiment`
  (paired seeds, Student-t intervals, ≥ 5 trials before anything is called "detectable").
- **Mock-mode numbers describe the fake model** and the economy rules, not language models.
