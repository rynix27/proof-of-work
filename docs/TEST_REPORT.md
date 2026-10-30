# Test report

Generated from a real run of the final code. **576 tests collected; all pass** (3 consecutive full runs identical, ~27 s each). Total line coverage **92%** (2723/2961 statements) over `core/`, `server.py`, `training/`.

Re-run: `pip install -r requirements-dev.txt && (cd tests/browser && npm install) && python -m pytest --cov=core --cov=server --cov=training`.

## Environments verified

| environment | result |
|---|---|
| Python 3.12, openai 3.40 (httpx2), pytest 9.2 | all pass |
| clean venv, **openai 1.30.5** (oldest allowed by `requirements.txt`, plain httpx) | 575 pass, 1 skipped (the jsdom DOM test needs Node) |
| real server process (waitress) over HTTP | exercised manually: auth, CSRF, stale-run guard, endpoints, SIGTERM -> clean exit |
| Docker image build | **NOT verified** (Docker unavailable in the audit environment) |
| live Nebius API | **NOT verified** (no credentials) |

## Tests per file

| file | tests | what it guards |
|---|---|---|
| `tests/test_build_dataset.py` | 5 | (baseline suite) |
| `tests/test_control.py` | 6 | (baseline suite) |
| `tests/test_dashboard_dom.py` | 1 | real-DOM render of hostile data in every panel |
| `tests/test_dataset_repro.py` | 8 | byte-identical datasets, leak-free split, metadata, hashes |
| `tests/test_demo.py` | 7 | the pinned demo seed keeps telling its story |
| `tests/test_economy.py` | 46 | loan lifecycle, interest maths, conservation property |
| `tests/test_endpoints_round2.py` | 28 | report, pagination, experiment browsing, leaderboard cost ranking |
| `tests/test_events.py` | 9 | (baseline suite) |
| `tests/test_experiments.py` | 26 | statistics, paired comparisons, tournament, reproducibility, isolation |
| `tests/test_injected_tasks.py` | 9 | (baseline suite) |
| `tests/test_ledger_journal.py` | 24 | transaction ids, idempotency, atomic rollback, journal reconciliation |
| `tests/test_lifecycle.py` | 24 | resets, stale threads, concurrency, disk failure |
| `tests/test_metrics.py` | 18 | every metric vs hand-computed values; calibration |
| `tests/test_model_client.py` | 8 | (baseline suite) |
| `tests/test_model_client_resilience.py` | 21 | provider failure classification, bounds, mock determinism |
| `tests/test_orchestrator.py` | 12 | (baseline suite) |
| `tests/test_orchestrator_rules.py` | 30 | auction, escalation, charging policy, reproducibility |
| `tests/test_security.py` | 9 | forbidden DOM sinks, secret redaction, config hygiene |
| `tests/test_security_round2.py` | 19 | CSRF, token throttling, stale-run guard, body limits (incl. real waitress socket) |
| `tests/test_server.py` | 49 | routing, access control, input validation, limits, exports |
| `tests/test_state.py` | 15 | state derivation tolerance and honesty |
| `tests/test_stats.py` | 6 | (baseline suite) |
| `tests/test_strategies.py` | 34 | EV/probability formulas, risk limits, escalation budget, explainability, no-omniscience, behavioural diversity |
| `tests/test_tasks.py` | 12 | (baseline suite) |
| `tests/test_training.py` | 48 | dataset filters, fine-tune boundary, evaluation, provider check, config |
| `tests/test_usage.py` | 18 | provider token usage, cost honesty, no silent fake fallback |
| `tests/test_validation.py` | 84 | untrusted model output, grading edge cases |

## Coverage by module

| module | statements | missed | covered |
|---|---|---|---|
| `core/agents.py` | 42 | 3 | 93% |
| `core/calibration.py` | 30 | 0 | 100% |
| `core/config.py` | 97 | 15 | 85% |
| `core/control.py` | 38 | 0 | 100% |
| `core/economy.py` | 207 | 1 | 100% |
| `core/events.py` | 155 | 15 | 90% |
| `core/injected_tasks.py` | 28 | 0 | 100% |
| `core/model_client.py` | 178 | 2 | 99% |
| `core/orchestrator.py` | 313 | 25 | 92% |
| `core/report.py` | 43 | 0 | 100% |
| `core/runner.py` | 38 | 0 | 100% |
| `core/security.py` | 12 | 0 | 100% |
| `core/simconfig.py` | 95 | 0 | 100% |
| `core/state.py` | 197 | 16 | 92% |
| `core/stats.py` | 46 | 0 | 100% |
| `core/strategies.py` | 146 | 1 | 99% |
| `core/tasks.py` | 72 | 2 | 97% |
| `core/validation.py` | 131 | 0 | 100% |
| `server.py` | 421 | 50 | 88% |
| `training/__init__.py` | 0 | 0 | 100% |
| `training/build_dataset.py` | 174 | 8 | 95% |
| `training/check_provider.py` | 39 | 7 | 82% |
| `training/demo.py` | 86 | 5 | 94% |
| `training/evaluate.py` | 90 | 30 | 67% |
| `training/experiment.py` | 158 | 5 | 97% |
| `training/finetune.py` | 106 | 34 | 68% |
| `training/run_self_play.py` | 19 | 19 | 0% |

## What the uncovered lines are

Mostly `if __name__ == '__main__'` CLI entry points (`run_self_play`, `finetune`, `evaluate`), `server.main()` (exercised manually over real HTTP instead), and live-network branches that cannot run without credentials. Coverage is a floor on what is exercised, not proof of correctness: several real defects in this project were found by property/adversarial tests and the DOM harness rather than by line coverage.

## Harness integrity

- The DOM harness was **mutation-checked**: re-introducing an `innerHTML` bug (in both the original and the new panels) makes it fail (exit 1); the real dashboard passes (exit 0). jsdom does not fire `onerror` (it does not load images), so the effective assertion is *no injected elements*.
- **Leak guard**: an autouse fixture fails any test that creates/modifies a file in the repository's real `data/` directory (verified with a deliberate canary test).
- Grep for vacuous assertions (`or True`) over `tests/`: none.
