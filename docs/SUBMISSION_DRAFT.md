# Devpost Submission Draft

Fill in the bracketed parts with real specifics once you've actually run
this against Nebius Token Factory — everything else is ready to paste in
as-is or lightly edit. Written in first person plural ("we") — change to
"I" if you'd rather, since you're solo.

---

## Project name
Proof of Work

## Tagline
An economy where thinking costs money.

## Inspiration

Most agentic AI demos give a model unlimited compute and ask it to
complete a task. We wanted to see what happens when AI agents have to
*pay* for their own thinking, out of a real, scarce, shared budget — the
same way people do. So we built a small economy: five Nemotron-powered
agents with distinct personalities, sharing a compute budget, where
every decision — bidding on a task, lending to another agent, escalating
to a more expensive model when they're unsure — costs real money they
can run out of.

Nothing about the outcome is scripted. We don't know which personality
will dominate a given run, whether a lending market will form, or who
goes bankrupt, until we watch it happen.

## What it does

Five AI agents (Specialist, Generalist, LoanShark, Hoarder, Gambler)
compete in a live economy:

- A stream of short logic/trivia tasks arrives with a compute bounty attached
- Each agent decides, via a real Nemotron call, whether to bid on it,
  skip it, lend spare balance to another agent, or borrow from one
- The winning bidder answers the task — using cheap, fast Nemotron Nano
  by default, escalating to the more expensive Nemotron Ultra only when
  their own confidence is too low to risk it
- Getting it right pays the bounty; running out of compute means
  bankruptcy — and Nemotron writes the bankrupt agent a short, in-character epitaph
- A live dashboard shows the whole thing: a real-time balance history
  chart, a scrolling transaction ticker, live sportscaster commentary
  every few rounds, a Gini coefficient tracking how unequal the economy
  currently is, and a cross-run leaderboard tracking win rates across
  every run we've ever done — not just the one you're watching
- Anyone can type a question into "Ask the economy" and watch it get
  answered live, by a real model call, in the next round

## How we built it

- **Nebius Token Factory** serves both model tiers used throughout:
  **NVIDIA Nemotron Nano** for every bid/skip/lend decision and most
  task-solving, **NVIDIA Nemotron Ultra** for the (costlier) escalation
  path when an agent's confidence is low
- Backend: Python, Flask, a plain JSONL event log as the single source
  of truth shared between the simulation and the dashboard
- Frontend: a single self-contained HTML dashboard — no framework,
  hand-drawn canvas chart, synthesized audio cues (no external sound
  files), polling a `/state` endpoint
- [If you ran it]: **Self-play fine-tuning** — every real model call is
  logged, later joined back against whether its round actually worked
  out, filtered down to a dataset of the agents' own successful
  decisions, and used to fine-tune Nemotron Nano on its own history

## Challenges we ran into

- [Fill in once you've actually integrated: anything about the Token
  Factory API shape, rate limits, latency between Nano and Ultra, etc.]
- Making sure a single flaky API call couldn't take down an entire live
  demo run — we built retry/backoff logic and round-level error recovery
  so the simulation survives real network failures gracefully instead of
  freezing mid-demo. We stress-tested this against both a total simulated
  API outage and a 30% random failure rate.
- Deciding how to grade a live, user-submitted question with no known
  correct answer ("Ask the economy") without breaking the integrity of
  the economy — we settled on treating ungraded questions as exhibition
  rounds: agents still reason and answer for real, but no bounty changes
  hands, so a mischievous or unanswerable question can't be gamed for
  free money.

## Accomplishments we're proud of

- A genuinely emergent system — we don't know what will happen in any
  given run before we watch it
- [Fill in a real number once you have one]: a cross-run leaderboard
  showing e.g. "[Agent] wins X% of Y runs" as actual evidence rather than
  cherry-picking one dramatic run
- [If you ran it] closing the loop with self-play fine-tuning: measuring
  a real before/after success-rate improvement from training Nemotron on
  its own agents' successful decisions

## What we learned

[Fill in honestly — something about prompt design for economic reasoning,
something about the Nano/Ultra cost tradeoff, something about Nebius
Token Factory specifically]

## What's next for Proof of Work

- Per-personality fine-tuned adapters instead of one shared model, so
  each agent's strategy sharpens independently over time
- A Physical AI extension: the same escalation-under-cost mechanic
  applied to a robot deciding when a cheap local model is enough versus
  when a task needs a heavier model
- Splitting the simulation loop and the dashboard across a proper Nebius
  Serverless Job + Serverless Endpoint (currently run together as one
  process for simplicity — see `docs/DEPLOYMENT.md`)

## Built with

Python, Flask, Nebius Token Factory, NVIDIA Nemotron (Nano + Ultra),
HTML/CSS/JavaScript (vanilla, no framework), Docker

---

## Feedback on Nebius / NVIDIA tools

[This section is read by judges — fill it in honestly once you've
actually built against the real API. Things worth commenting on either
way: how clear the Token Factory docs were, whether the OpenAI-SDK
compatibility worked as expected out of the box, latency differences
between Nano and Ultra, anything about the fine-tuning API if you used
it, anything that was confusing or took longer than expected to figure
out. Specific, honest feedback is worth far more here than generic
praise — a couple of the hackathon's award categories are specifically
about the quality of this feedback.]
