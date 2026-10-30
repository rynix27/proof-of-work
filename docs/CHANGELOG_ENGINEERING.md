# Engineering changelog

## Round 2 (this pass)
**Security**
- Cross-site request guard on every POST (`Origin` must be this host or in `ALLOWED_ORIGINS`; `Sec-Fetch-Site: cross-site` refused). Previously any website could pause/reset an open-mode local server.
- Wrong-token attempts throttled per client (`AUTH_FAIL_MAX_PER_MINUTE`, 429 + `Retry-After`).
- Stale-run guard: control requests may carry `X-Run-Id`/`run_id`; a mismatch returns 409. The dashboard sends it.
- Request-size limit enforced by Flask (`MAX_CONTENT_LENGTH`) **and** waitress (`max_request_body_size`); read endpoints rate-limited per client.
- Dashboard: accessible text colour for "bankrupt/failed" (was 2.75:1, now 4.89:1).

**Economy**
- `core/economy.py`: transaction journal with unique ids, idempotency references, all-or-nothing rollback, repayment history, journal-derived reconciliation (independent of the live arithmetic).
- `core/strategies.py`: per-agent policies (conservative, aggressive, cost-efficient, accuracy-focused, lender + a faithful `baseline` control). Probability from the agent's own record, EV = p·reward − cost, risk limits, escalation budget/cutoff/EV rule, lender-side policy for loans, proactive lending; every decision records its numbers and reasons.
- Optional model fields `estimated_success_probability` / `estimated_difficulty` (validated; invalid ones dropped, not fatal). Bounties and strategy sets are configurable and validated.

**Measurement**
- Provider-reported token usage and optional cost estimate (never a guessed price); a real failure is never replaced by fake output.
- Calibration (Brier/ECE/reliability), per-tier escalation effectiveness, cumulative series, additional rates; `docs/METRICS.md`.
- Experiments: Student-t intervals, paired differences, rotation-controlled strategy tournament, analysis that only states what the sample supports.
- Run report (`/export/report.md`), paginated `/events`, experiment browsing, `cost per correct` ranking.
- Deterministic datasets with metadata sidecar, leak-free train/validation split, content hashes; pinned, tested demo scenario (`python -m training.demo`).

## Round 1
See `docs/ENGINEERING_AUDIT.md` (register rows R1-xx): missing pause route, stored XSS, reset race, loans without interest/repayment, unvalidated model output, fragile grading, failed-call accounting, provider config, event-log integrity, replay-name validation, leaderboard integrity, inject abuse, access control/headers, dataset leakage.
