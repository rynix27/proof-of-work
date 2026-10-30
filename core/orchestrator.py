"""
The round loop. Plain readable functions rather than a class hierarchy.

ROUND RULES (also documented in README "Economic model")
  1. A task arrives (an injected "ask the economy" question first, else a random
     bank task).
  2. Every solvent agent makes ONE decision: bid | skip | lend | borrow. Model
     output is validated (core.validation); invalid output => the agent skips
     and the failure is recorded. Deciding is free.
  3. AUCTION: bids are ranked by self-reported confidence (higher wins). Bids
     from agents who cannot afford a Nano solve are rejected. Ties are broken
     deterministically by a roster order that rotates every round. The bid
     `amount` field is currently ignored. Losing bids cost nothing. No bids =>
     task unclaimed.
  4. The winner solves. Escalation to Ultra happens only if bid confidence <
     escalation_cutoff AND the winner can afford ULTRA_COST.
  5. ACCOUNTING for failed calls (see core/model_client.py): rejected-before-
     processing => free; timeout (outcome unknown) or empty answer from a
     provider that processed the request => charged once; success => charged.
     An unusable answer from a successful call is charged and graded wrong.
  6. Correct => bounty minted to the winner. Wrong => only the compute is lost.
     Ungraded exhibition tasks pay nothing.
  7. End of round: loans due this round are collected/defaulted, then every
     agent at/below the bankruptcy line goes bankrupt (its loans default).
"""
import logging
import random
import threading
import time

from core import config, events, tasks
from core.agents import fresh_roster, record_result
from core.control import control
from core.economy import Ledger, LedgerError
from core.injected_tasks import injected_tasks
from core import model_client as _mc
from core.model_client import call_model_json, ModelCallError
from core.simconfig import SimulationConfig
from core.strategies import BASELINE, decide_escalation, evaluate_decision, lend_cap, resolve_strategies
from core.validation import ValidationError, validate_answer, validate_decision

log = logging.getLogger("proof_of_work.orchestrator")

DECISION_SYSTEM_PROMPT = (
    "You are a reasoning agent participating in a compute economy. Deciding is free, "
    "but if you win a task you must solve it, which costs compute-seconds from your own "
    "balance (a cheap tier costs less than the expensive deep-reasoning tier). A correct "
    "answer pays the task bounty; a wrong answer pays nothing. The most confident bidder "
    "wins the task. Respond ONLY with a JSON object, no other text, no markdown fences: "
    '{"action": "bid" | "skip" | "lend" | "borrow", "amount": <int>, '
    '"confidence": <float 0-1>, "estimated_success_probability": <float 0-1>, '
    '"estimated_difficulty": "easy" | "medium" | "hard", "reason": "<short reason>"}. '
    'If action is "lend", amount is how much to offer (loans are repaid with interest). '
    'If action is "borrow", amount is how much you want (you must repay with interest).'
)

SOLVE_SYSTEM_PROMPT = (
    "Answer the question as briefly and precisely as possible. "
    "Respond ONLY with a JSON object, no other text: "
    '{"answer": "<your answer>", "confidence": <float 0-1>}'
)

EPITAPH_SYSTEM_PROMPT = (
    "You write a single short, darkly funny one-line epitaph, in character, for an "
    "AI agent that has just gone bankrupt in a compute economy. Under 20 words. "
    "Plain text only -- no quotation marks, no markdown, nothing but the epitaph itself."
)

COMMENTARY_SYSTEM_PROMPT = (
    "You are a witty sportscaster providing live color commentary on an AI agent "
    "economy, the way an announcer calls a horse race. One punchy sentence, under "
    "25 words. Plain text only -- no quotation marks, no markdown."
)

# Per-thread run context: the ledger/config/rng owned by the simulation thread.
# (A thread-local keeps run_round(agents, round_num) callable with two arguments.)
_ctx = threading.local()
flavor_failures = 0


class RunContext:
    """What one simulation thread owns: ledger, config, rng and per-agent policies."""

    def __init__(self, ledger, cfg, rng, policies):
        self.ledger, self.cfg, self.rng, self.policies = ledger, cfg, rng, policies

    def policy(self, agent):
        return self.policies.get(agent.name, BASELINE)


def _context(agents) -> RunContext:
    """The running simulation's context, or an ad-hoc one (tests / one-off calls) built
    from the current defaults."""
    ledger = getattr(_ctx, "ledger", None)
    if ledger is not None and ledger.agents is agents:
        return RunContext(ledger, _ctx.cfg, _ctx.rng, _ctx.policies)
    cfg = SimulationConfig.defaults()
    policies = resolve_strategies(cfg.strategies, [a.name for a in agents])
    for a in agents:
        a.strategy = policies[a.name].name
    return RunContext(Ledger(agents, cfg), cfg, getattr(_ctx, "rng", None) or random.Random(), policies)


def _flavor_call(prompt: str, system: str, purpose: str, round_num: int = None,
                 agent_name: str = None) -> str | None:
    """Narrative-flavor calls (epitaphs, commentary) must never take down a round,
    so failures return None -- but they are logged and counted, not swallowed
    silently. Excluded from training datasets (purpose != decision/solve)."""
    global flavor_failures
    from core.model_client import call_model
    try:
        text, _latency = call_model(prompt, tier="nano", system=system,
                                    meta={"purpose": purpose, "round": round_num, "agent": agent_name})
    except Exception as e:  # noqa: BLE001 -- deliberate boundary, logged below
        flavor_failures += 1
        log.warning("flavor call (%s) failed: %s", purpose, e)
        return None
    cleaned = text.strip().strip('"').strip()
    return cleaned[:200] if cleaned else None


def _track_record(agent) -> str:
    """The agent's own results so far -- information it legitimately has."""
    if not agent.by_difficulty:
        return "Your track record: no graded results yet."
    parts = [f"{d}: {c}/{n} correct" for d, (n, c) in sorted(agent.by_difficulty.items())]
    return "Your track record: " + ", ".join(parts) + "."


def decide_action(agent, task, round_num: int) -> dict:
    ledger_cfg = getattr(_ctx, "cfg", None) or SimulationConfig.defaults()
    prompt = (
        f"{agent.seed_prompt}\n\n"
        f"Round {round_num}. Your current balance: {agent.balance} compute-seconds. {_track_record(agent)}\n"
        f"Solving costs {ledger_cfg.nano_cost} (cheap tier) or {ledger_cfg.ultra_cost} "
        f"(deep tier). Loans carry {int(round(ledger_cfg.interest_rate * 100))}% interest "
        f"and fall due after {ledger_cfg.loan_term_rounds} rounds.\n"
        f"Incoming task (difficulty {task.difficulty}, bounty {task.bounty}): "
        f"{task.prompt!r}\n"
        "Decide your action for this round."
    )
    meta = {"agent": agent.name, "round": round_num, "purpose": "decision", "task_id": task.id,
            "injected": task.injected}
    decision = call_model_json(prompt, tier="nano", system=DECISION_SYSTEM_PROMPT, meta=meta)
    agent.nano_calls += 1
    return decision


def solve_task(agent, task, escalate: bool, round_num: int) -> dict:
    """Calls the model only. Charging is applied by run_round per the policy."""
    tier = "ultra" if escalate else "nano"
    prompt = f"Question: {task.prompt}"
    meta = {"agent": agent.name, "round": round_num, "purpose": "solve", "task_id": task.id,
            "injected": task.injected}
    return call_model_json(prompt, tier=tier, system=SOLVE_SYSTEM_PROMPT, meta=meta)


def _bankruptcy_line(cfg) -> int:
    # an agent that can no longer afford a Nano solve is effectively dead too
    return max(cfg.bankruptcy_threshold, cfg.nano_cost - 1)


def _error_fields(e: Exception) -> dict:
    return {"error": str(e), "kind": getattr(e, "kind", "unexpected")}


def run_round(agents: list, round_num: int):
    ctx = _context(agents)
    ledger, cfg, rng = ctx.ledger, ctx.cfg, ctx.rng
    # bounties are passed only when customised, keeping the 2-argument call shape that
    # existing callers/fakes of generate_task rely on
    extra = {} if tuple(cfg.bounties) == (8, 20, 50) else {"bounties": cfg.bounties}
    task = injected_tasks.pop_next() or tasks.generate_task(rng, cfg.task_weights, **extra)
    events.emit("task_arrived", round=round_num, task_id=task.id,
                prompt=task.prompt, difficulty=task.difficulty, bounty=task.bounty,
                injected=task.injected)

    bids = []
    for agent in [a for a in agents if not a.bankrupt]:
        try:
            raw = decide_action(agent, task, round_num)
        except Exception as e:  # noqa: BLE001 -- one agent's failure must not end the round
            log.warning("decision call failed for %s: %s", agent.name, e)
            events.emit("call_error", round=round_num, agent=agent.name, purpose="decision",
                        charged=0, **_error_fields(e))
            continue
        try:
            if raw.get("_parse_error"):
                raise ValidationError(raw.get("_error", "malformed_json"), "unparseable response")
            proposal = validate_decision(raw)
        except ValidationError as e:
            agent.invalid_decisions += 1
            events.emit("decision_invalid", round=round_num, agent=agent.name, code=e.code,
                        detail=str(e)[:200], fallback="skip")
            continue

        # The model proposed; the agent's strategy decides. Pure function of observable numbers.
        out = evaluate_decision(
            agent, proposal, task, ctx.policy(agent), balance=agent.balance,
            start=agent.starting_balance, debt=ledger.debt_of(agent), cfg=cfg,
            others=[o for o in agents if o is not agent])
        decision = out.decision
        events.emit("decision", round=round_num, agent=agent.name, action=decision.action,
                    model_action=out.model_action, amount=decision.amount,
                    confidence=decision.confidence, strategy=out.strategy,
                    p_est=out.p_est, ev=out.ev, est_cost=out.cost, reward=out.reward,
                    balance=agent.balance, debt=ledger.debt_of(agent),
                    policy_reasons=list(out.reasons), explanation=out.explanation,
                    model_explanation=proposal.reason,          # the model's own words, not verified
                    model_est_probability=proposal.est_probability,
                    model_est_difficulty=proposal.est_difficulty,
                    reason=proposal.reason, warnings=list(proposal.warnings))

        if decision.action == "bid":
            bids.append((agent, decision, out))
        elif decision.action == "lend":
            _handle_lend_offer(agent, decision, agents, round_num, ledger, cfg, rng, ctx.policy(agent))
        elif decision.action == "borrow":
            _handle_borrow_request(agent, decision, agents, round_num, ledger, cfg, ctx.policy)

    # ---- auction: validate bids, then highest confidence wins ----
    valid = []
    for agent, decision, out in bids:
        if agent.bankrupt or agent.balance < cfg.nano_cost:
            events.emit("bid_rejected", round=round_num, agent=agent.name,
                        reason="insufficient_funds")
            _check_bankruptcy(agent, round_num, ledger)
        else:
            valid.append((agent, decision, out))

    if not valid:
        events.emit("task_unclaimed", round=round_num, task_id=task.id,
                    reason="no_valid_bids" if bids else "no_bids")
        return

    n = len(agents)
    order = {a.name: (i - round_num) % n for i, a in enumerate(agents)}
    valid.sort(key=lambda p: (-p[1].confidence, order[p[0].name]))
    winner, decision, win_out = valid[0]
    events.emit("auction_resolved", round=round_num, task_id=task.id, winner=winner.name,
                bids=[{"agent": a.name, "confidence": d.confidence, "p_est": o.p_est, "ev": o.ev,
                       "strategy": o.strategy} for a, d, o in valid],
                rule="highest_confidence; ties by rotating roster order")

    escalate, esc_reasons = decide_escalation(
        winner, ctx.policy(winner), decision.confidence, task, balance=winner.balance,
        start=winner.starting_balance, cfg=cfg)
    tier = "ultra" if escalate else "nano"
    cost = cfg.ultra_cost if escalate else cfg.nano_cost
    if escalate:
        winner.escalations += 1
        events.emit("escalation", round=round_num, agent=winner.name,
                    confidence=decision.confidence,
                    cutoff=(cfg.escalation_cutoff if ctx.policy(winner).escalation_cutoff is None
                            else ctx.policy(winner).escalation_cutoff), cost=cost,
                    strategy=ctx.policy(winner).name)
    elif esc_reasons:
        events.emit("escalation_denied", round=round_num, agent=winner.name,
                    confidence=decision.confidence, reasons=list(esc_reasons),
                    strategy=ctx.policy(winner).name)

    try:
        result = solve_task(winner, task, escalate=escalate, round_num=round_num)
    except Exception as e:  # noqa: BLE001 -- the task goes unresolved, the run continues
        charged = bool(getattr(e, "charged", False))
        log.warning("solve call failed for %s (%s): %s", winner.name, tier, e)
        txn = ledger.charge_compute(winner, cost, round_num,
                                    ref=f"solve:{round_num}:{task.id}:{winner.name}") if charged else None
        if charged and escalate:
            winner.ultra_spent += cost
        events.emit("call_error", round=round_num, agent=winner.name, purpose="solve",
                    tier=tier, charged=cost if charged else 0, new_balance=winner.balance,
                    cost_txn=txn, **_error_fields(e))
        events.emit("task_unclaimed", round=round_num, task_id=task.id, reason="solve_failed")
        _check_bankruptcy(winner, round_num, ledger)
        return

    # the provider answered: the simulated compute is spent either way
    cost_txn = ledger.charge_compute(winner, cost, round_num,
                                     ref=f"solve:{round_num}:{task.id}:{winner.name}")
    if escalate:
        winner.ultra_calls += 1
        winner.ultra_spent += cost
    else:
        winner.nano_calls += 1
    winner.tasks_won += 1

    valid_response = True
    try:
        if result.get("_parse_error"):
            raise ValidationError(result.get("_error", "malformed_json"))
        answer, _conf = validate_answer(result)
    except ValidationError as e:
        valid_response = False
        winner.invalid_answers += 1
        answer = ""
        events.emit("answer_invalid", round=round_num, agent=winner.name, code=e.code)

    correct = tasks.grade(task, answer)
    if correct is not None:
        winner.graded_attempts += 1
        record_result(winner, task.difficulty, tier, bool(correct), cost)
    reward_txn = None
    if correct is True:
        reward_txn = ledger.reward(winner, task.bounty, round_num,
                                   ref=f"bounty:{round_num}:{task.id}:{winner.name}")
        winner.tasks_correct += 1

    events.emit(
        "task_resolved", round=round_num, task_id=task.id, agent=winner.name,
        tier=tier, correct=correct, exhibition=(correct is None), answer=answer[:200],
        bounty=task.bounty if correct is True else 0, cost=cost, escalated=escalate,
        valid_response=valid_response, injected=task.injected, new_balance=winner.balance,
        cost_txn=cost_txn, reward_txn=reward_txn, bid_confidence=decision.confidence,
        p_est=win_out.p_est, ev_est=win_out.ev, strategy=win_out.strategy, difficulty=task.difficulty,
    )
    _check_bankruptcy(winner, round_num, ledger)


def _handle_lend_offer(lender, decision, agents, round_num, ledger=None, cfg=None, rng=None, policy=None):
    if ledger is None:
        c = _context(agents)
        ledger, cfg, rng = c.ledger, c.cfg, c.rng
        policy = policy or c.policy(lender)
    policy = policy or BASELINE
    rng = rng or random.Random()
    amount = min(decision.amount, cfg.max_loan_amount, lender.balance - cfg.lender_reserve)
    if amount <= 0:
        events.emit("loan_rejected", round=round_num, lender=lender.name, code="lender_reserve",
                    requested=decision.amount)
        return
    able = [a for a in agents if a is not lender
            and len(ledger.active_loans(a)) < cfg.max_active_loans]
    bankrupt_agents = [a for a in able if a.bankrupt]
    target = bankrupt_agents[0] if bankrupt_agents else None
    if target is None:
        needy = [a for a in able if not a.bankrupt
                 and a.balance < a.starting_balance * policy.lend_target_below_fraction]
        target = rng.choice(needy) if needy else None
    if target is None:
        return
    try:
        ledger.create_loan(lender, target, amount, round_num, initiated_by="lender",
                           request_key=f"lend:{round_num}:{lender.name}")
    except LedgerError as e:
        events.emit("loan_rejected", round=round_num, lender=lender.name, borrower=target.name,
                    code=e.code, requested=amount)


def _handle_borrow_request(borrower, decision, agents, round_num, ledger=None, cfg=None, policy_of=None):
    if ledger is None:
        c = _context(agents)
        ledger, cfg, policy_of = c.ledger, c.cfg, policy_of or c.policy
    policy_of = policy_of or (lambda a: BASELINE)
    amount = min(decision.amount, cfg.max_loan_amount)
    def willing(a):
        """The LENDER's own strategy must allow this loan (can_lend, reserve, per-loan cap)."""
        pol = policy_of(a)
        return (pol.can_lend and a.balance - amount >= max(cfg.lender_reserve,
                                                           int(pol.lend_reserve_fraction * a.starting_balance))
                and amount <= lend_cap(pol, a.balance))
    lenders = sorted(
        [a for a in agents if a is not borrower and not a.bankrupt
         and a.balance - amount >= cfg.lender_reserve and willing(a)],
        key=lambda a: -a.balance,
    )
    if not lenders:
        events.emit("loan_rejected", round=round_num, borrower=borrower.name,
                    code="no_lender_available", requested=amount)
        return
    try:
        ledger.create_loan(lenders[0], borrower, amount, round_num, initiated_by="borrower",
                           request_key=f"borrow:{round_num}:{borrower.name}")
    except LedgerError as e:
        events.emit("loan_rejected", round=round_num, lender=lenders[0].name,
                    borrower=borrower.name, code=e.code, requested=amount)


def _check_bankruptcy(agent, round_num, ledger=None):
    line = _bankruptcy_line(ledger.cfg) if ledger else config.BANKRUPTCY_THRESHOLD
    if agent.balance <= line and not agent.bankrupt:
        agent.go_bankrupt()
        if ledger:
            ledger.default_loans_of(agent, round_num)
        epitaph_prompt = (
            f"{agent.seed_prompt}\n\n"
            f"You just went bankrupt on round {round_num} after starting with "
            f"{agent.starting_balance} compute-seconds. Write your epitaph."
        )
        epitaph = _flavor_call(epitaph_prompt, EPITAPH_SYSTEM_PROMPT, "epitaph",
                               round_num=round_num, agent_name=agent.name)
        events.emit("bankruptcy", round=round_num, agent=agent.name, epitaph=epitaph)


def _maybe_emit_commentary(agents: list, round_num: int):
    if round_num % config.COMMENTARY_INTERVAL_ROUNDS != 0:
        return
    standings = ", ".join(
        f"{a.name}: {a.balance}{' (bankrupt)' if a.bankrupt else ''}"
        for a in sorted(agents, key=lambda a: -a.balance)
    )
    prompt = f"Round {round_num}. Current standings: {standings}. Give your commentary."
    line = _flavor_call(prompt, COMMENTARY_SYSTEM_PROMPT, "commentary", round_num=round_num)
    if line:
        events.emit("commentary", round=round_num, text=line)


def agent_stats(agents, ledger=None) -> dict:
    out = {}
    for a in agents:
        graded = a.graded_attempts
        out[a.name] = {
            "final_balance": a.balance, "net_pl": a.balance - a.starting_balance,
            "bankrupt": a.bankrupt, "attempts": a.tasks_won, "graded_attempts": graded,
            "correct": a.tasks_correct,
            "accuracy": round(a.tasks_correct / graded, 3) if graded else None,
            "compute_spent": a.compute_spent, "rewards_earned": a.rewards_earned,
            "invalid_decisions": a.invalid_decisions, "invalid_answers": a.invalid_answers,
            "escalations": a.escalations, "nano_calls": a.nano_calls, "ultra_calls": a.ultra_calls,
            "strategy": a.strategy, "ultra_spent": a.ultra_spent,
            "economic_profit": a.rewards_earned - a.compute_spent,
            "loans_issued": ledger.loans_issued(a) if ledger else 0,
            "outstanding_debt": ledger.debt_of(a) if ledger else 0,
        }
    return out


def _end_of_round(agents, round_num, ledger):
    ledger.settle_due_loans(round_num)
    for a in agents:
        _check_bankruptcy(a, round_num, ledger)


def run_simulation(max_rounds: int = None, round_delay: float = None, verbose=True, *,
                   cfg: SimulationConfig = None, config_overrides: dict = None, seed: int = None,
                   generation: int = None, run_id: str = None):
    """
    Entry point for a full run. Normally started via core.runner.manager (which
    allocates `generation` and `run_id` before the thread starts). Called
    directly (CLI, tests) it allocates both itself.
    """
    from core import model_client

    if cfg is None:
        overrides = dict(config_overrides or {})
        if max_rounds:
            overrides["max_rounds"] = max_rounds
        if seed is not None:
            overrides["seed"] = seed
        cfg = SimulationConfig.from_overrides(overrides)
    if round_delay is not None:
        # programmatic callers (tests, CLI, experiments) may use 0; the HTTP
        # endpoint still clamps to >= 0.1s via control.set_speed
        control.round_delay = max(0.0, min(30.0, float(round_delay)))
    my_generation = control.start_new_generation() if generation is None else generation
    if run_id is None:
        events.reset_log()
    events.bind_run(events.current_run_id() if run_id is None else run_id)

    try:
        _run(cfg, my_generation, verbose, model_client)
    finally:
        _ctx.ledger = None
        events.bind_run(None)


def _run(cfg, my_generation, verbose, model_client):
    run_seed = cfg.seed if cfg.seed is not None else random.SystemRandom().randrange(2**31)
    rng = random.Random(run_seed)
    model_client.seed_fake(run_seed)
    tasks.reset_ids()
    injected_tasks.clear() if my_generation is None else None

    agents = fresh_roster(cfg.num_agents, cfg.starting_balance)
    ledger = Ledger(agents, cfg)
    policies = resolve_strategies(cfg.strategies, [a.name for a in agents])
    for a in agents:
        a.strategy = policies[a.name].name
    _ctx.ledger, _ctx.cfg, _ctx.rng, _ctx.policies = ledger, cfg, rng, policies

    resolved = {t: model_client._resolve_model(t) for t in ("nano", "ultra")}
    events.emit("simulation_started", agents=[a.name for a in agents],
                strategies={a.name: a.strategy for a in agents},
                starting_balance=cfg.starting_balance, mode=config.mode(),
                config=cfg.to_dict(), seed=run_seed, models=resolved)

    ended_reason = "max_rounds_reached"
    final_round = 0

    for round_num in range(1, cfg.max_rounds + 1):
        if control.should_stop(my_generation):
            ended_reason = "stopped"
            break
        control.wait_while_paused(my_generation)
        if control.should_stop(my_generation):
            ended_reason = "stopped"
            break

        try:
            run_round(agents, round_num)
            _end_of_round(agents, round_num, ledger)
        except Exception as e:  # noqa: BLE001 -- last-resort net so the dashboard never freezes
            log.exception("round %s failed", round_num)
            events.emit("round_error", round=round_num, error=str(e))
            if verbose:
                print(f"[round {round_num}] ERROR: {e} -- continuing to next round")
        final_round = round_num

        usage = _mc.take_usage()
        if usage:                                   # live mode only: provider-reported, not simulated
            events.emit("usage", round=round_num, usage=usage, estimated_cost_usd=_mc.estimate_cost_usd(usage),
                        basis="provider-reported tokens x configured price (None if prices unset)")
        events.emit(
            "round_snapshot", round=round_num,
            balances={a.name: (0 if a.bankrupt else a.balance) for a in agents},
            bankrupt=[a.name for a in agents if a.bankrupt],
            outstanding_debt=ledger.outstanding_debt(),
        )
        _maybe_emit_commentary(agents, round_num)

        if verbose:
            state = ", ".join(f"{a.name}={a.balance}{'(BANKRUPT)' if a.bankrupt else ''}"
                              for a in agents)
            print(f"[round {round_num}] {state}")

        solvent = [a for a in agents if not a.bankrupt]
        if len(solvent) <= 1:
            ended_reason = "monopoly"
            events.emit("monopoly", round=round_num,
                        winner=solvent[0].name if solvent else None)
            break

        slept = 0.0
        while slept < control.round_delay:
            if control.should_stop(my_generation) or control.is_paused():
                break
            step = min(0.2, control.round_delay - slept)
            time.sleep(step)
            slept += step

    final_balances = {a.name: a.balance for a in agents}
    winner = (max(final_balances, key=final_balances.get)
              if ended_reason in ("monopoly", "max_rounds_reached") else None)
    stats = agent_stats(agents, ledger)
    recon = ledger.reconcile()
    if not recon["ok"]:
        log.error("ledger reconciliation FAILED: %s", recon)
    events.emit("simulation_ended", final_balances=final_balances, reason=ended_reason,
                rounds_played=final_round, winner=winner, agent_stats=stats,
                reconciliation=recon, loans=ledger.loan_summary(), seed=run_seed)

    if winner is not None:
        events.record_leaderboard_entry(winner, final_balances, final_round, ended_reason,
                                        agent_stats=stats,
                                        extra={"seed": run_seed, "mode": config.mode()})
        try:
            events.save_replay(label=winner)
        except OSError as e:
            log.error("replay save failed: %s", e)
            events.emit("replay_save_failed", error=str(e))


if __name__ == "__main__":
    control.set_speed(0.1)  # fast for local testing
    run_simulation()
