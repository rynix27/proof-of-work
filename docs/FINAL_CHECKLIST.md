# Final checklist before you submit

Deadline per the Devpost page: **Oct 30, 2026**. Items marked ☐ need *your* credentials or machine; I could not do them.

## Must do (nothing below has been run by the authors)
- ☐ `USE_FAKE_MODEL=0 NEBIUS_API_KEY=... python -m training.check_provider` — all PASS. This is the first real contact with the API. If "non-empty content" fails, raise `MODEL_MAX_TOKENS`.
- ☐ Set a **spend limit on your Nebius key** (the app's budgets are simulated compute, not billing).
- ☐ One short live run (`MAX_ROUNDS` small); confirm the banner says **LIVE**, `/state.provider_usage` shows tokens, and no `call_error` flood.
- ☐ `docker build -t proof-of-work . && docker run ...` once (Docker was unavailable during the audit).
- ☐ Open the dashboard in a real browser (desktop + phone width) — the DOM test used jsdom; no one has *looked* at it.
- ☐ Rehearse `python -m training.demo --install-replay`, then play the replay in the dashboard.

## Keep your claims inside the evidence
- ✅ Can claim: deterministic economy with a balanced, journaled ledger; validated model I/O; strategy policies with explainable EV decisions; measured (not assumed) confidence calibration; reproducible experiments; hardened server; 576 tests.
- ❌ Do **not** claim: that fine-tuning improved anything (none was run), that Ultra escalation "works" (mock accuracy ignores tier), any LLM-behaviour finding from mock runs, or real-money cost figures (usage is provider-reported only in live mode; prices are yours to configure).
- If you fine-tune: build the dataset (`--val-fraction 0.1`), train, **deploy** the adapter (Nebius requires it), then `training.evaluate` base vs tuned on the held-out tasks. The held-out set is only 10 prompts: grow the task bank first or you cannot detect an effect.

## Deploy (see docs/DEPLOYMENT.md)
- ☐ `CONTROL_TOKEN` set (or accept view-only), `PUBLIC_MODE=1`, TLS at the proxy, `TRUST_PROXY=1` only behind a trusted proxy, **one replica**.
- ☐ Decide `INJECT_MAX_PER_HOUR_GLOBAL` — in live mode each viewer question costs model calls.
