# Proof of Work 

## The idea

Most agent demos give an AI unlimited compute and ask it to do a task.
Proof of Work does the opposite: five Nemotron-powered agents share a
tiny, real compute budget. Every decision — bid on a task, skip it, lend
spare balance to another agent, borrow from one — costs compute-seconds
drawn from a two-tier model system:

- **Nemotron Nano** — cheap, fast, used for every bid/skip/lend decision
  and for solving most tasks
- **Nemotron Ultra** — expensive, slower, used only when an agent's own
  confidence in its answer is too low to risk it

Agents don't follow a script. Each has a short personality seed prompt
(Specialist, Generalist, LoanShark, Hoarder, Gambler) that biases their
strategy, but the actual bid/skip/lend/borrow decision comes from a real
model call every round. Over a run, a lending market forms, some agents
go bankrupt, others get bailed out, and — usually — one strategy starts
to dominate. None of that is scripted; it's what falls out of the
personalities interacting with a shared, scarce resource.

A live dashboard shows the whole thing as it happens: agent balances,
a running balance-history chart, an economy stats panel, the current
task, and a scrolling transaction ticker — with live pause/resume/speed
controls and full run replay so you can capture a great run once and
show it on demand instead of hoping live randomness cooperates.

## Features

- **Live economy simulation** — 5 personality-seeded Nemotron agents
  bidding, skipping, lending, and borrowing every round against a real
  compute budget
- **Two-tier model escalation** — Nano handles every decision and most
  task-solving; agents escalate to Ultra only when their own confidence
  is too low, and that escalation costs more compute-seconds
- **Balance history chart** — a live line chart (hand-drawn on canvas,
  no external chart library) showing every agent's fortunes diverge
  over the course of a run
- **Economy stats panel** — Nano vs Ultra call counts, total compute
  burned, tasks resolved, success rate — a direct, visual answer to
  "how did you use Token Factory and Nemotron"
- **Live controls** — pause, resume, reset, and change the round speed
  from the dashboard while a run is live
- **Replay save/load** — snapshot any run (manually, or automatically
  whenever one ends) and play it back later from a dropdown, so your
  demo never depends on a live run happening to be dramatic at the
  right moment
- **Dramatic banners + sound** — a full-width banner and a synthesized
  audio cue (no external sound files) fire on bankruptcies and when one
  agent achieves a monopoly
- **Bankruptcy epitaphs** — when an agent goes bankrupt, Nemotron writes
  a short one-line epitaph in that personality's voice, shown in the
  banner and the ticker
- **Live sportscaster commentary** — every few rounds, a separate
  Nemotron call narrates the state of the economy like a race announcer
- **Wealth inequality (Gini coefficient)**, computed live and shown in
  the stats panel — an objective "how skewed is this economy right now"
  number, not just a good-looking chart
- **Cross-run leaderboard** — every completed run (this session and
  every one before it) is tallied into a persistent win-rate table, so
  you can say "Specialist wins 61% of 40 runs" instead of pointing at
  one dramatic run and hoping it generalizes
- **Resilient to real API failures** — every real Token Factory call
  retries through rate limits and transient errors; if a call still
  fails, that one agent's decision or that one round is skipped with a
  visible error in the ticker instead of silently freezing the whole
  simulation. Stress-tested against both a total simulated API outage
  and a 30% random failure rate — the simulation kept running in both
  cases.
- **"Ask the economy"** — type any question into the dashboard and it's
  queued as the next round's task. Give an answer and it's graded and
  paid out like any other task; leave it blank and it runs as an
  ungraded exhibition round — agents still reason and answer for real,
  just with no economic effect, so a joke question can't be gamed for
  free money.
- **65-test pytest suite** covering the Gini math, grading/normalization,
  event log + leaderboard persistence across runs, pause/resume/reset
  state, the retry logic (including simulated rate limits and total API
  outages), the bankruptcy/epitaph flow, and — critically — the
  fine-tuning dataset builder's join logic, tested against two runs that
  deliberately share the same round numbers to prove run_id
  disambiguation actually works.

## How it uses Nebius + NVIDIA

- Runs entirely on **Nebius Token Factory**, calling **NVIDIA Nemotron**
  models through its OpenAI-compatible API
- Deliberately exercises *two* model tiers per call pattern (Nano vs
  Ultra) rather than one — the escalation decision is itself part of the
  economy, since Ultra calls cost more compute-seconds than Nano calls
- Designed to run the simulation loop as a **Nebius Serverless Job** and
  serve the dashboard/state via a **Nebius Serverless Endpoint** (see
  Architecture below)

## Quickstart

```bash
git clone <this-repo>
cd proof-of-work
pip install -r requirements.txt
cp .env.example .env
```

By default `.env` has `USE_FAKE_MODEL=1`, which runs the entire
simulation against a local fake model — no API key needed. This is how
the dashboard and economy logic were built and tested before Token
Factory access was wired in, and it's still useful for quick local
iteration.

```bash
python server.py
```

Then open **http://localhost:5000** — the dashboard starts polling
immediately and a simulation begins in the background.

### Running against real Nebius Token Factory

1. Get an API key from Nebius Token Factory.
2. In `.env` set `NEBIUS_API_KEY=...` and `USE_FAKE_MODEL=0`.
3. **Verify the provider first** (takes seconds, costs ~nothing):
   ```bash
   python -m training.check_provider
   ```
   It checks that your key works, that `NANO_MODEL` / `ULTRA_MODEL` exist in
   your account's model list, that a reply is **non-empty** (the Nemotron 3
   models are reasoning models; a too-small `MODEL_MAX_TOKENS` returns empty
   content) and that it parses as the JSON the simulation expects.
   This integration has **not** been run against the live API by the authors
   of this repository (no credentials were available) — this script is how
   you verify it.
4. `python server.py`. The dashboard banner always says **MOCK** or **LIVE**.

> The base URL (`https://api.tokenfactory.nebius.com/v1/`) and the model ids
> were checked against Nebius's published docs / cookbook; model availability
> is per-account, so confirm with step 3.

## Economic model

All amounts are **integer compute-seconds** (simulated — not provider billing).

| Rule | Behaviour |
|---|---|
| Round | A task arrives; each solvent agent makes **one** decision: `bid`, `skip`, `lend`, `borrow`. Deciding is free. |
| Auction | Highest self-reported **confidence** wins. Bids from agents who can't afford a Nano solve are rejected. Ties break by a roster order that rotates every round (deterministic, no permanent favourite). The bid `amount` field is currently ignored. Losing bids cost nothing. |
| Escalation | The winner escalates to Ultra only if bid confidence is below **its strategy's cutoff** (default 0.6) **and** it can afford `ULTRA_COST` (12), is within its **escalation budget**, and (for cost-aware strategies) Ultra's expected value is non-negative. Otherwise Nano (cost 1). Refusals are recorded as `escalation_denied` with reasons. |
| Reward | Correct answer mints the task **bounty** (8/20/50 by difficulty) to the winner. Wrong = only the compute is lost. Escalating an 8-bounty task costs 12, so it is unprofitable by design. |
| Ungraded questions | "Ask the economy" questions without an answer are **exhibition**: they cost compute, pay nothing, and are excluded from accuracy. |
| Loans | Simple interest, integer math: `interest = ceil(principal × 15%)`, `owed = principal + interest` (20 → 23). Due `LOAN_TERM_ROUNDS` (5) after creation. Cap `MAX_LOAN_AMOUNT` 50, ≤3 active loans per borrower, lender keeps a reserve of 1. |
| Repayment / default | At the due round the borrower pays `min(balance, owed)`; any shortfall is **defaulted** (the lender absorbs it). Partial payment is supported; a repaid/defaulted loan can never be settled twice. Borrower bankruptcy defaults its active loans immediately. A bankrupt agent can be revived by a new loan; old debts stay written off. |
| Bankruptcy | Balance at/below the line (or unable to afford a Nano solve) ⇒ frozen. |
| Conservation | `Σ balances = start + minted − burned`, always (`Ledger.reconcile()`, asserted in tests and reported in every `simulation_ended` event). |
| Failed model calls | Rejected before processing (auth, 4xx, 429, 5xx, connection) ⇒ **not charged**. Timeout (outcome unknown) or an empty/unusable answer from a provider that processed the request ⇒ **charged once**, never per retry. Unusable answers are graded wrong. |
| Invalid model output | Every decision/answer is validated (type, range, finiteness, size). Invalid ⇒ the agent skips; recorded as `decision_invalid`. NaN, strings, negatives, huge numbers never reach a balance. |

## Agent strategies

The language model **proposes**; each agent's strategy **decides**. A proposal is validated, then
checked by a deterministic, testable policy (`core/strategies.py`) using only what the agent
could actually know: its own balance, debts, track record, the visible task, published costs.
It never sees the answer.

```
p_hist = (correct + prior·w) / (attempts + w)          per difficulty, from the agent's own results
p      = (1 − trust)·p_hist + trust·model_confidence    trust caps how much the model's word counts
EV     = p × bounty − cost
```
Risk limits then apply (minimum EV and p, maximum bid size, reserve, debt limit, loan caps,
escalation budget). The decision event records `p_est`, `ev`, the reasons for any override, a
system-generated explanation, and — kept separate and labelled — the model's own explanation.

| strategy | character | notable behaviour |
|---|---|---|
| `baseline` | control group | follows the model; only hard affordability rules apply |
| `conservative` | protects capital | needs EV ≥ 2, large reserve, tiny loans, rarely escalates |
| `aggressive` | maximises activity | bids on marginal tasks, escalates readily, borrows to keep bidding, never lends |
| `cost_efficient` | reward per compute | EV margin required; escalates only if Ultra is profitable |
| `accuracy_focused` | gets it right | needs p ≥ 0.6, escalates on doubt, accepts lower profit |
| `lender` | earns interest | picky bidder; proactively lends surplus to struggling agents |

By default (`STRATEGY_SET=roster`) each personality plays the matching strategy. The model's
self-reported confidence is **measured** (Brier/ECE, flagged below 30 results), never assumed
calibrated. In mock mode strategies differ in *behaviour* (verified pairwise-distinct) but the
fake model's accuracy ignores them, so mock results show mechanics, not which strategy is smarter.

## Demo, reports and cost accounting

```bash
python -m training.demo --install-replay   # a pinned, un-staged 50-round story; appears in the Replay menu
curl localhost:5000/export/report.md       # plain-English run report (also for ?replay=…)
curl 'localhost:5000/events?after_seq=0&limit=200'   # paginated raw events
```
The demo (seed 8, starting balance 30) naturally contains a strategy overriding the model, an
escalation, a loan made and repaid, a refused escalation, a loan default and a bankruptcy; a test
fails if a code change ever alters that. Two cost notions are **never mixed**: *simulated
compute-seconds* (the in-game currency) and *provider-reported tokens* (live mode only; an
estimated dollar figure appears only if you configure prices, otherwise "unknown").
Metric definitions: `docs/METRICS.md`.

## Security & access modes

| Mode | How | Controls (`/control/*`) |
|---|---|---|
| Local (default) | binds `127.0.0.1` | open |
| Token | `CONTROL_TOKEN=...` | require header `X-Control-Token` |
| Public | `PUBLIC_MODE=1` | **disabled (403)** unless `CONTROL_TOKEN` is set — fails closed |

Every `POST` is refused unless its `Origin` is this server (or in `ALLOWED_ORIGINS`), so a
malicious web page cannot drive a local server; wrong control tokens are throttled; control
requests may carry the `run_id` they act on and get **409** if that run has been replaced;
bodies are size-limited by both Flask and waitress. "Ask the economy" stays open to viewers but is rate limited (per IP and
globally), size-limited, and refuses with 429 when the queue is full. The
dashboard renders every model- or user-supplied string as text (no HTML
parsing), the server sends a strict CSP (`script-src 'self'`), and errors
returned to clients are generic. Behind a reverse proxy set `TRUST_PROXY=1`.
Run **one** server process (state is in-process); see `docs/DEPLOYMENT.md`.

## Reproducibility & experiments

Every run records its **seed, mode, resolved config and model ids** in the
`simulation_started` event. With the mock model, the same seed gives
identical runs:

```bash
curl -X POST localhost:5000/control/reset -H 'Content-Type: application/json' \
     -d '{"seed": 7, "config": {"num_agents": 4, "interest_rate": 0.3, "max_rounds": 60}}'

python -m training.experiment --param interest_rate --values 0,0.15,0.4 \
       --trials 10 --rounds 40 --seed 100 --out data/experiments/interest
```
The experiment runner uses paired seeds, writes only to its `--out` folder, and reports
mean, median, std and a 95 % Student-t interval with *n*; differences vs the control are paired
and called "detectable" only with n ≥ 5 and an interval excluding 0. For strategies, run a
**tournament** (all five in one economy, seats rotated each trial so position can't favour anyone):
```bash
python -m training.experiment --tournament --trials 10 --rounds 60 --seed 100 --out data/experiments/t1
```
Outputs `runs.csv`, `summary.json` and an `analysis.md` that states only what the sample supports
(browse them in the dashboard's Experiments panel). **In mock mode the agents are a random fake
model**, so results exercise the economy rules and pipeline — they are not
evidence about LLM behaviour. Exports: `/export/events.csv|json`, `/export/summary.json`.

## Architecture

```
core/
  config.py        all tuning knobs (starting balance, costs, thresholds,
                    commentary frequency)
  stats.py         pure statistics helpers (Gini coefficient) -- kept
                    separate so the math is independently testable
  model_client.py  Token Factory client (+ fake mode for offline dev)
  agents.py        Agent class + the 5 personality seed prompts
  tasks.py         task bank (short-answer logic/trivia, auto-graded)
  events.py        structured JSONL event log + replay save/list/read +
                    the persistent cross-run leaderboard -- shared
                    source of truth between the simulation and dashboard
  control.py       thread-safe pause/resume/speed/reset state shared
                    between the simulation loop and the Flask endpoints
  orchestrator.py  the round loop: decide -> bid -> solve -> settle ->
                    lend/borrow -> bankruptcy check (+ epitaph) ->
                    snapshot -> periodic sportscaster commentary
web/
  dashboard.html   live dashboard: ledger, history chart, stats panel,
                    leaderboard, task strip, ticker, controls, replay
                    picker, commentary strip
server.py          Flask app: dashboard, /state, /control/*, /replays,
                    /leaderboard, runs the simulation in a background
                    thread
training/
  run_self_play.py    batch-runs short simulations to accumulate real
                       self-play history quickly
  build_dataset.py    joins raw calls to real outcomes, filters to
                       successful traces, writes fine-tuning datasets
  finetune.py          uploads a dataset and launches a fine-tuning job
                       on Nebius Token Factory
```

`core/orchestrator.py` and `server.py` are split deliberately: the
orchestrator's round loop is the piece meant to run as a **Nebius
Serverless Job**, while the Flask app is the piece meant to sit behind a
**Nebius Serverless Endpoint**. They're run together in one process here
for local development simplicity.

## Tuning the economy

Defaults live in `core/config.py` (starting balance, costs, escalation
cutoff, loan interest/term/caps, bankruptcy line). Any of them can be
overridden **per run, validated**, through `POST /control/reset`
(`config` object) or `SimulationConfig` — see `core/simconfig.py` for
the allowed ranges.

Agent personalities live in `core/agents.py` as plain text seed prompts
— rewrite or add one any time.

## Feedback on Nebius Token Factory / NVIDIA tools

_(fill in after building against the real API — note anything that
surprised you, was harder than expected, or worked especially well)_

## Self-play fine-tuning (stretch goal)

Once the core system is running well against the real API, you can close
the loop: collect real self-play history, filter it down to the
decisions and answers that actually worked, and fine-tune Nemotron Nano
on its own successful behavior.

**1. Collect real self-play data.** Set `USE_FAKE_MODEL=0` in `.env`
with a real API key, then run several short simulations back to back:
```bash
python -m training.run_self_play --runs 20 --rounds-per-run 15
```
Every real call gets appended to `data/training/calls.jsonl`
automatically (as long as `LOG_TRAINING_DATA=1`, the default) — this
persists across every run, unlike `data/events.jsonl` which resets each
time.

**2. Build the dataset.** Joins every call back to its round outcome and
keeps only examples that are (a) correct, (b) for bids, followed by a correct
answer, (c) **profitable** (bounty − compute ≥ 0), (d) valid under the same
validators the app uses, (e) not duplicates, not user-injected questions, no
secrets/e-mail-like strings, and (f) **not from the held-out task slice**:
```bash
python -m training.build_dataset          # strict provenance by default
```
Writes `nano_all.jsonl`, `ultra_all.jsonl`, per-agent files and
`manifest.json` (counts + why things were skipped). Raw logs are never modified.

**3. Launch the fine-tuning job.**
```bash
python -m training.finetune --dataset data/training/datasets/nano_all.jsonl --dry-run
python -m training.finetune --dataset data/training/datasets/nano_all.jsonl --base-model <a model Nebius lists as fine-tunable>
```
> **Not verified against a live account.** Nebius documents the OpenAI-style
> `files` + `fine_tuning.jobs` flow, but (1) check that your Nemotron model is
> in Nebius's *fine-tunable* list, and (2) a successful job yields an adapter
> that Nebius says must be **deployed** before it can be served
> ([deploy docs](https://docs.tokenfactory.nebius.com/fine-tuning/deploy-custom-model)).
> Only put the *deployed served name* in `FINE_TUNED_NANO_MODEL`.

**4. Measure — don't assume.** Evaluate base and tuned models on the same
**held-out** tasks (never trained on), then compare:
```bash
python -m training.evaluate --label base  --out data/eval/base.json
python -m training.evaluate --label tuned --model-id <served name> --out data/eval/tuned.json
python -m training.evaluate --compare data/eval/base.json data/eval/tuned.json
```
Reports accuracy **with a 95 % interval**, invalid-output rate, simulated
cost per correct answer, net outcome and escalation frequency. The task bank
currently has only **10 held-out prompts**, so intervals are wide (≈ ±25 pts):
**grow the bank before drawing any conclusion.** Nothing in this repository
claims a fine-tune has helped.

## Running the tests

```bash
pip install -r requirements-dev.txt
(cd tests/browser && npm install)   # optional: real-DOM XSS test (needs Node)
python -m pytest
```
575 passed, 1 skipped in the latest local test run; coverage was not measured in that run, no API key or network needed (mock model, mocked provider);
about 10 seconds. Covers the ledger (incl. a conservation property test),
validation/grading edge cases, race/lifecycle behaviour, the HTTP API and
access control, a hostile-data DOM test of the dashboard, and the training
pipeline. Also verified on `openai==1.30.5` (the oldest allowed) and 3.x.

## Deploying

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for a Dockerfile-based path
to a public URL, and the honest caveats about what's verified versus
what still needs checking against the real Nebius dashboard.

## Documentation

| doc | contents |
|---|---|
| `docs/ENGINEERING_AUDIT.md` | issue register (found / fixed / how verified), what is **not** verified, known limitations |
| `docs/METRICS.md` | exact definition of every metric |
| `docs/TEST_REPORT.md` | real test counts and coverage |
| `docs/CHANGELOG_ENGINEERING.md` | what changed |
| `docs/FINAL_CHECKLIST.md` | what you must do yourself before submitting |
| `docs/DEPLOYMENT.md` | running it for others |

## License

MIT — see [LICENSE](LICENSE).
