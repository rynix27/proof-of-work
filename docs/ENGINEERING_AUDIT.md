# Engineering audit

**Final state:** 576 tests (92 % line coverage), stable over repeated runs, also green on the oldest
allowed SDK (openai 1.30.5). A real server process was exercised over HTTP. Numbers: `docs/TEST_REPORT.md`.
**Not verified:** live Nebius calls, Docker build, a human looking at the UI in a real browser (see §4).

Severity: **P0** exploitable / data-integrity, **P1** wrong behaviour or serious gap, **P2** robustness /
quality, **P3** polish. Every "fixed" row names the test that now guards it.

## 1. Issue register — found in the code under audit

| ID | Sev | Finding | Evidence | Fix | Guarded by |
|---|---|---|---|---|---|
| R1-01 | P1 | `control_pause()` had no `@app.route` | `POST /control/pause` → 404 | route restored | `test_every_documented_route_is_registered` |
| R1-02 | **P0** | Stored XSS: task prompts, model answers, epitaphs, errors, agent names, replay filenames went through `innerHTML` | code review; DOM harness | text-only DOM builder; JS served as a file; strict CSP | denylist test + jsdom harness (mutation-checked) |
| R1-03 | P1 | Reset race leaked an old run's events into the new run | review | generation under a lock; per-thread run binding; bounded join | `test_lifecycle.py` (12 rapid resets; reset during a hung call) |
| R1-04 | P1 | Loans never repaid; `LOAN_INTEREST_RATE` unused | review | ledger with interest, due dates, defaults | `test_economy.py` (25-seed conservation property) |
| R1-05 | **P0** | Model JSON used unvalidated: `NaN`, strings, negatives reached balances | review | `core/validation.py`; invalid ⇒ safe skip | 60+ parametrized cases |
| R1-06 | P1 | `grade()` crashed on non-string answers | review | coercing, numeric-aware grader | `test_validation.py` |
| R1-07 | P1 | No failed-call accounting; SDK retries stacked on app retries (≤ 9 requests/call); 5xx not retried | SDK signature inspected | typed errors with `charged`; SDK retries off; deadline | `test_model_client_resilience.py` |
| R1-08 | P1 | `max_tokens=300` vs Nemotron *reasoning* models; `content=None` crashed | provider issue + review | 1024 default, empty-content error, `<think>` stripping | unit tests (live value unverified) |
| R1-09 | P1 | Stale provider URL / guessed model ids | checked vs Nebius docs & cookbook | updated | `check_provider` (live check is yours) |
| R1-10..18 | P2 | unlocked event log; replay-name `$` regex accepted `"x.jsonl\n"`; leaderboard could exceed 100 % win rate; misleading prompt/footer; inject abuse; no access control/headers; dataset leakage; … | various | see `CHANGELOG_ENGINEERING.md` | `test_server.py`, `test_lifecycle.py`, `test_training.py` |

## 2. Issue register — found in round 2 (the second review)

| ID | Sev | Finding | Evidence | Fix | Guarded by |
|---|---|---|---|---|---|
| R2-01 | **P0** | **CSRF on open-mode controls**: any web page could pause/reset a local server with a "simple" cross-origin POST | `Origin: https://evil.example` + `text/plain` → **200** | `Origin`/`Sec-Fetch-Site` guard on every POST; `ALLOWED_ORIGINS` | `test_cross_origin_posts_are_refused_in_every_mode` (6 endpoints × 3 modes) |
| R2-02 | P1 | Control token could be guessed without limit | 30 wrong guesses all → 401 | per-client throttle, 429 + `Retry-After` | `test_wrong_token_attempts_are_throttled…` |
| R2-03 | P1 | A stale/replayed control request could act on a *different* run | review | `X-Run-Id` / `run_id` guard → 409; dashboard sends it | `test_stale_run_requests_are_refused_with_409` |
| R2-04 | P1 | "Strategies" were prompt text only: no EV, no risk limits, no policy | 5 agents given identical model output differ only by name | `core/strategies.py` policy layer | `test_strategies.py` (formulas, boundaries, diversity) |
| R2-05 | P1 | No transaction ids, idempotency or rollback in the ledger | review | journal, `ref`/`request_key`, `_atomic`, journal-derived reconciliation | `test_ledger_journal.py` |
| R2-06 | P2 | Body limit was a manual check only | review | + Flask `MAX_CONTENT_LENGTH` + waitress `max_request_body_size` | real-socket waitress test |
| R2-07 | P2 | Read endpoints unthrottled | review | per-client read limiter | `test_read_endpoints_are_rate_limited_per_client` |
| R2-08 | P2 | Bankrupt/failed **text** colour 2.75:1 (WCAG AA needs 4.5) | computed | `--rust-text` 4.89:1 | contrast computed; visual check still yours |
| R2-09 | P2 | No provider usage; simulated and real cost not distinguished | review | token usage capture; optional estimate from *your* prices; never guessed | `test_usage.py` |
| R2-10 | P2 | Model confidence assumed calibrated | review | Brier/ECE/reliability, flagged below n = 30 | `test_metrics.py` |
| R2-11 | P2 | Experiments had no intervals, pairing or seat rotation | review | Student-t, paired differences, rotation tournament, gated analysis | `test_experiments.py` |
| R2-12 | P3 | Dataset order not deterministic; no validation split/hashes | review | sorted output, task-hash split, sidecar, sha256 manifest | `test_dataset_repro.py` |
| R2-13 | P3 | No reproducible demo | — | pinned, tested scenario | `test_demo.py` |

Performance was **measured, not assumed**: `/state` = 18 ms at the 200-round maximum (83 KB payload,
polled every 1.5 s), `/leaderboard` 0.4 ms. Nothing warranted optimisation, so none was done. The
event log is re-parsed per request; revisit (cache by size/mtime) only if rounds/viewers grow a lot.

## 3. Defects I introduced while fixing, and how they were caught
Listed because an audit that hides its own mistakes cannot be trusted.

| What went wrong | Caught by |
|---|---|
| Idempotency key claimed *outside* the rollback snapshot ⇒ a rolled-back request could never be retried | the rollback test, immediately |
| Made baseline's lend cap `INF`; `int(inf × balance)` overflowed in the lender-willingness check | the existing test suite (4 failures) |
| `baseline` clamped oversized lend offers ⇒ not a faithful control group | `test_baseline_never_overrides_the_model` |
| Escalation budgets (5–10 % of start = 5–10) were **below the cost of one Ultra call (12)** ⇒ three strategies could never escalate; my "rarely escalates" claim was really "never", and the zeros I attributed to the confidence cutoff were this | `test_escalation_needs_affordability_budget…` on first run; re-measured; regression test added |
| Policy budget check pre-empted the auction's visible `bid_rejected` event | `test_bid_from_agent_who_cannot_afford_nano…` |
| Strategies had a dormant loan market (0 loans) and `lender` ≡ `cost_efficient` | measuring behaviour *before* asserting on it |
| Stale-run parser crashed on a list body | existing server test |
| Experiment analysis called "fewer escalations" *unfavourable* (invented value judgement) | reading the generated output |
| Dashboard `el()`/`setChildren` mishandled arrays/strings | the DOM harness |
| I wrote **four vacuous `or True` assertions** | self-review; removed; `grep` over tests now finds none |
| My first "vulnerable" mutation was overwritten by the function's own safe render, so the harness *looked* like it passed | repeating the mutation correctly (harness then failed as it should) |
| A ledger test called `monkeypatch.undo()`, which also undid the autouse fixture that keeps tests out of the real `data/` directory ⇒ events leaked into the repo's real `data/events.jsonl` | the final hygiene scan; fixed, and a **leak guard** now fails any test that writes into real `data/` (proven with a deliberate canary) |
| **The sandbox was reset mid-task and uncommitted round-2 work was lost** | rebuilt from the persisted round-1 zip, re-applied and re-verified step by step; checkpointing to persistent storage after each stage |

## 4. Not verified — you must check
1. **Live Nebius calls** (no credentials): run `python -m training.check_provider`. Model ids, `MODEL_MAX_TOKENS=1024`, and usage-field names come from docs/reasoning, not a run.
2. **Fine-tuning**: no job was run. Whether your Nemotron model is fine-tunable, the job fields, and Nebius's required *deploy* step are from documentation.
3. **Docker**: `docker build` / `compose` never ran (Docker unavailable). The same server process was run directly.
4. **Visual UI**: tested in jsdom only; no human has looked at it. jsdom does not fire `onerror`, so the DOM test asserts *no injected elements*.
5. **TLS / reverse proxy / DNS-rebinding**: the app speaks plain HTTP; it does not validate the `Host` header in local mode.

## 5. Known limitations (honest scope)
- **Mock results mean little**: the fake model ignores tier, strategy and balance. An interest-rate sweep changed nothing; escalation cannot pay off; "cost-efficient" wins mock tournaments simply by spending less. The pipeline is real; insight requires live agents.
- Held-out evaluation set = **10 tasks** (accuracy interval ≈ ±25 pts).
- Bids are ranked by self-reported confidence (gameable); bid `amount` is ignored. Policy limits the damage, it does not fix the mechanism.
- Single process, file-backed log; rate limits are in-memory per process; `leaderboard.jsonl` and `calls.jsonl` grow without rotation.
- CSP keeps `style-src 'unsafe-inline'` (inline `style=` attributes).
- A client behind one NAT shares a rate-limit bucket unless `TRUST_PROXY=1` behind a real proxy.
