# Demo Video Script (3 minutes max)

Written as a shot list, not a word-for-word script — say it in your own
voice, but hit these beats in this order and roughly this timing. Record
several live runs first and pick the best window rather than trying to
get one perfect take live (see "Before you record" below).

## Before you record

**Fastest reliable option (no API key):** `python -m training.demo --install-replay` installs a
pinned, un-staged 50-round story into the Replay menu (a strategy overriding the model, an
escalation, a loan made and repaid, a refused escalation, a default and a bankruptcy), and prints
the round-by-round timeline to narrate from. **It is a mock run** — the banner says MOCK and the
fake model plays — so say so in the video; do not present it as Nemotron behaviour. Use a real
(LIVE) run for any claim about the models.


1. Run the system for real (not fake mode) for at least 20-30 minutes,
   at a fast-but-legible speed (2-3 seconds/round via the Speed control).
2. Watch for a run that produces a genuinely interesting story: a
   bankruptcy, a lending relationship forming, one personality clearly
   pulling ahead. Use **Save this run** the moment something good
   happens, so you have it in the Replays dropdown even if the run
   later flattens out.
3. Record your screen for the actual video using the **Replay dropdown**
   to play back your best saved run — this means you're never at the
   mercy of live randomness cooperating during the one take you record.
   Playback happens at the same pace it was recorded, so speed it up
   with screen-recording software if needed, or re-save a run captured
   at a faster round interval.

## Shot list

**0:00 -- 0:15 — Cold open, no explanation yet**
Cut straight to the dashboard mid-run: the ledger bars mid-swing, the
balance history chart with visibly diverging lines, the ticker
scrolling. Let it breathe for a few seconds before any voiceover. The
goal is "wait, what is this?" not "let me explain what you're about to
see."

**0:15 -- 0:40 — The rule, in one breath**
Voiceover: something like *"Five AI agents share a real compute budget.
Every decision they make — bid on a task, lend money, borrow from each
other — costs them their own money. If they run out, they go bankrupt.
Nothing here is scripted; every one of these decisions is a live call to
NVIDIA Nemotron running on Nebius Token Factory."*
On screen: point at the agent ledger while saying "compute budget," then
the ticker while saying "nothing scripted."

**0:40 -- 1:10 — The actual story from your best run**
Narrate what actually happened in the run you're replaying — be specific
and use real names/numbers from your own data, not a generic description.
Example shape: *"Here, LoanShark started cornering the lending market
around round 12 — see these loans stacking up — while Hoarder barely
spent anything the whole run. Then Gambler went all-in on a hard question,
got it wrong, and by round 20..."*
On screen: let the bankruptcy banner and its epitaph play if one happened
in this run — it's a strong, funny, memorable beat, don't cut past it.

**1:10 -- 1:35 — The two-tier model system, explicitly**
This is where you directly answer "how did you use Nebius/NVIDIA" for
anyone judging technically. Voiceover: *"Every agent's decision uses
Nemotron Nano — fast and cheap. But when an agent isn't confident in its
own answer, it escalates to Nemotron Ultra, which costs more compute —
so the choice to escalate is itself part of the economy."*
On screen: point at the stats panel's Nano/Ultra call counts and the
"tier" label in a ticker line where an Ultra escalation happened.

**1:35 -- 2:00 — Live interactivity (if you built "ask the economy")**
Type a real question into the ask box on camera, live. *"Let's ask the
economy something nobody planned for."* Wait for it to actually get
picked up and answered in the ticker. This is a strong "this is real,
not a canned demo" moment — keep it in even if it takes a few seconds of
dead air waiting for the next round.

**2:00 -- 2:25 — Rigor, not just vibes**
Voiceover: *"This isn't one lucky run — here's the win rate across every
run we've done."* Show the Leaderboard section. *"And here's the Gini
coefficient — a live measure of how unequal the economy currently is."*
This is your answer to "Quality of the Idea" and "Technological
Implementation" — a single dramatic run is a demo, but a leaderboard
across dozens of runs is evidence.

**2:25 -- 2:50 — The stretch goal, if you did it**
If you actually ran the self-play fine-tuning pipeline: *"We even closed
the loop — collected the agents' own successful decisions, fine-tuned
Nemotron on them, and here's the before/after success rate."* Show the
stats panel's success-rate number from before and after, side by side if
you can (two screen recordings cut together, or a simple on-screen
comparison graphic).

**2:50 -- 3:00 — End on the most dramatic single moment**
Cut back to whatever the single best 2-3 seconds of your best run was —
a bankruptcy banner, a monopoly banner, whatever hit hardest. End on
that, cut to black or your project name card. Don't add a "thank you for
watching" slide — you don't have the seconds to spare.

## Things to double check before submitting the video

- [ ] Under 3 minutes total (Devpost's stated limit)
- [ ] Shows a real demo URL or the local dashboard working, not just slides
- [ ] Mentions Nebius Token Factory and NVIDIA Nemotron by name at least once, explicitly
- [ ] No dead air longer than ~2 seconds anywhere
- [ ] Uploaded to YouTube (per the hackathon's submission requirements) and the link works in an incognito window before you submit it
