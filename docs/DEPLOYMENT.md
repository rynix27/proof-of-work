# Deployment

Two paths, roughly in order of effort: run the container anywhere for a
quick public URL, or split it properly across Nebius Serverless Job +
Endpoint the way the architecture was designed for.

> **Honest caveat up front**: I don't have live access to the Nebius
> Serverless dashboard or CLI docs, so I can't verify the exact deploy
> commands below against Nebius's actual interface. What I can verify
> (and did): the **same server process the image runs** (waitress, public
> mode, token auth, graceful SIGTERM) was booted and exercised over real
> HTTP outside Docker. What is **not** verified: `docker build` / `docker
> compose up` themselves — Docker was unavailable in the audit environment.
> Build it once yourself before you rely on it for a demo.

## Option A: run the container anywhere (fastest path to a public URL)

This is the simplest way to satisfy the hackathon's "demo URL" requirement
if you don't want to dig into Nebius Serverless specifics right away.
Works on Nebius AI Cloud's general compute, a basic VM, Fly.io, Render,
Railway, or anywhere else that runs a Docker container and exposes a port.

```bash
docker build -t proof-of-work .
docker run -p 5000:5000 -v pow-data:/app/data \
  -e CONTROL_TOKEN="$(openssl rand -hex 24)" \
  --env-file .env proof-of-work
# or: CONTROL_TOKEN=... docker compose up --build   (read-only rootfs, caps dropped)
```

The image is **secure by default**: it runs as a non-root user, defaults to
`PUBLIC_MODE=1` (so `/control/*` returns 403 unless you set `CONTROL_TOKEN`,
in which case the dashboard shows an "Unlock controls" button and sends the
token as `X-Control-Token`), serves with `waitress` (not Flask's dev server),
exposes `/healthz` for orchestrator health checks, and shuts down cleanly on
SIGTERM (writes a final `simulation_ended` event). Put `.env` values in your
platform's secret store rather than baking them into anything.

**Hardening checklist before a public demo**
- [ ] `CONTROL_TOKEN` set (or accept that viewers cannot pause/reset)
- [ ] `TRUST_PROXY=1` **only** if behind a trusted reverse proxy (otherwise all viewers share one rate-limit bucket)
- [ ] TLS terminated by your host/proxy (the app speaks plain HTTP)
- [ ] **Exactly one replica** and one volume — state is in-process
- [ ] Decide `INJECT_MAX_PER_HOUR_GLOBAL` — in live mode every question costs real model calls
- [ ] `USE_FAKE_MODEL=0` only after `python -m training.check_provider` passes; the banner will then read LIVE
- [ ] A provider-side spend limit on your Nebius key (the app's budgets are *simulated* compute, not billing)

Then point whatever hosting platform you're using at port 5000, and
you have a public URL. Check the box titled "runs entirely on Nebius"
in your submission is still satisfied as long as the container's compute
calls go through Token Factory (they do — that's the only place
`NEBIUS_API_KEY` is used) even if the container itself runs on general
compute rather than a Serverless product.

### One real limitation of this option

The current app runs the simulation loop and the Flask server in one
process, storing state in local files (`data/events.jsonl`,
`data/replays/`, `data/leaderboard.jsonl`). That's fine for a demo
(single container, single instance), but it does NOT survive a container
restart, and does NOT work if you ever scale to more than one container
instance (each would have its own independent, disconnected economy).
For a hackathon demo, one always-on container is almost certainly fine.

## Option B: split across Nebius Serverless Job + Serverless Endpoint

This matches the architecture as designed (see the note at the top of
`server.py` and `core/orchestrator.py`): the simulation loop is meant to
run as a **Serverless Job** (a long-running background process), and the
dashboard/API is meant to sit behind a **Serverless Endpoint** (a
request-driven web service).

The blocker to doing this cleanly: a Job and an Endpoint are almost
certainly separate processes/containers on Nebius, which means they
can't share the local `data/` filesystem the way the current single-process
app does. Before splitting them, you'd need to swap the local JSONL files
for something both sides can reach — the two lowest-effort options:

1. **A small shared database** (SQLite on a mounted volume if Nebius
   supports one, or Postgres/Redis if you want something more robust).
   `core/events.py` is the only file that touches the event log directly
   -- swapping its `emit`/`read_all`/`read_replay`/etc. functions to hit
   a database instead of a local file is a contained, one-file change.
2. **Keep it simple and skip the split.** Given the hackathon timeline,
   Option A (one container running both pieces) is almost certainly the
   better use of your remaining time unless a judge specifically asks
   about the Job/Endpoint split.

If you do want to attempt the split:
- Check the Nebius dashboard for "Serverless Jobs" and "Serverless
  Endpoints" (or whatever they're currently named) and see what shared
  storage options they offer to a Job and an Endpoint running together.
- The Job's entrypoint would be roughly:
  ```bash
  python -c "from core.orchestrator import run_simulation; run_simulation()"
  ```
- The Endpoint's entrypoint is the existing `server.py`, minus the
  `start_simulation_background()` call (since the Job is now doing that
  separately) — it would just read whatever shared store you set up.

## Environment variables to set on whichever platform you use

Same as local development (see `.env.example`):
```
NEBIUS_API_KEY=your_real_key
NEBIUS_BASE_URL=...      # confirm against Nebius docs
NANO_MODEL=...           # confirm against Nebius docs
ULTRA_MODEL=...          # confirm against Nebius docs
USE_FAKE_MODEL=0
LOG_TRAINING_DATA=1
```

## Before you rely on this for the actual demo

1. Build the image locally and confirm it runs: `docker build -t proof-of-work . && docker run -p 5000:5000 --env-file .env proof-of-work`, then open `http://localhost:5000`.
2. Deploy it to wherever you're hosting it, then open the public URL from a different network (like your phone on cellular data) to confirm it's actually reachable from outside — a container that only responds on the deploy platform's internal network is a common last-mile mistake.
3. Do at least one full run-through on the live public URL, the same way a judge would encounter it cold, before submission.
