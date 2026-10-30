"""
Agent strategies: a deterministic, testable policy layer between the language
model's *proposal* and the simulation.

WHY: an LLM's output (action, confidence, amounts) is a proposal, not a decision
we can trust economically. The model may be uncalibrated, may ignore its budget,
or may propose something ruinous. Each agent therefore has a StrategyPolicy that
(1) estimates success probability from information the agent actually has,
(2) computes expected value with the *documented* reward/cost definitions, and
(3) enforces explicit risk limits. The outcome, with its numbers, is recorded
on the decision event so every bid/skip/lend/borrow/escalate can be explained.

INFORMATION AVAILABLE TO AN AGENT (no omniscience)
  its own balance and track record (per-difficulty and per-tier results), its own
  debts, the visible task (prompt, difficulty, bounty), published costs, and the
  public balances of other agents. It does NOT see the task's answer, other
  agents' private decisions, or the future.

PROBABILITY ESTIMATE
  p_hist = (correct + prior * w) / (attempts + w)         per difficulty, Beta-style smoothing
  p      = (1 - trust) * p_hist + trust * model_confidence     (trust in [0,1])
  The model's self-reported confidence is NOT assumed calibrated: `trust` caps its
  influence (0 for strategies that ignore it). Calibration is measured separately
  (core/calibration.py) rather than assumed.

EXPECTED VALUE (all integer compute-seconds, documented in README "Economic model")
  EV(bid) = p * bounty - cost            cost = nano_cost (a bid is solved at Nano
                                          unless escalated; escalation is decided
                                          separately below). Cost is counted once.

HARD AFFORDABILITY belongs to the auction: a bid from an agent who cannot afford a
Nano solve reaches the auction and is rejected there with a visible `bid_rejected`
event. The policy applies its own *risk* limits only to bids the agent can afford.

STRATEGIES are plain data (StrategyPolicy). `baseline` applies no gating and
reproduces the pre-strategy behaviour exactly: it is the control for experiments.
"""
from dataclasses import dataclass, replace

from core.validation import Decision

INF = float("inf")


@dataclass(frozen=True)
class StrategyPolicy:
    name: str
    description: str
    # --- bidding ---
    min_ev: float = -INF               # bid only if EV >= min_ev
    min_p: float = 0.0                 # ... and estimated success probability >= min_p
    max_bid_fraction: float = 1.0      # estimated cost must be <= fraction * balance
    min_reserve_fraction: float = 0.0  # balance after cost must stay >= fraction * starting balance
    prior_accuracy: float = 0.6
    prior_weight: float = 4.0
    trust_confidence: float = 1.0      # weight of the model's self-reported confidence in p
    # --- escalation ---
    escalation_cutoff: float | None = None   # None => use the run config's cutoff
    escalation_requires_ev: bool = False     # require EV(ultra) >= 0 to escalate
    escalation_budget_fraction: float = INF  # cap on total Ultra spend, fraction of starting balance
    # --- lending / borrowing ---
    can_lend: bool = True
    lend_reserve_fraction: float = 0.0       # keep at least this fraction of start after lending
    lend_max_fraction: float = INF           # a single loan <= fraction of current balance (INF = ungated)
    lend_target_below_fraction: float = 0.5  # who counts as 'struggling': balance < fraction * their start
    proactive_lend_surplus: float | None = None  # lend even when the model said skip, if balance > fraction*start
    can_borrow: bool = True
    borrow_below_fraction: float = INF       # borrow only while balance < fraction * start
    max_debt_fraction: float = INF           # outstanding debt <= fraction * start (after the new loan)


BASELINE = StrategyPolicy("baseline", "Follows the model's proposal; only hard affordability rules apply (control group).")

STRATEGIES: dict[str, StrategyPolicy] = {
    "baseline": BASELINE,
    "conservative": StrategyPolicy(
        "conservative", "Bids only on clearly positive-EV tasks, keeps a large reserve, rarely escalates or borrows.",
        min_ev=2.0, min_p=0.5, max_bid_fraction=0.10, min_reserve_fraction=0.20,
        prior_accuracy=0.5, prior_weight=4.0, trust_confidence=0.0,
        escalation_cutoff=0.4, escalation_requires_ev=True, escalation_budget_fraction=0.15,
        lend_reserve_fraction=0.60, lend_max_fraction=0.15,
        borrow_below_fraction=0.50, max_debt_fraction=0.30),
    "aggressive": StrategyPolicy(
        "aggressive", "Bids on marginal tasks, trusts the model's confidence, escalates readily, borrows to keep bidding.",
        min_ev=-1.0, min_p=0.0, max_bid_fraction=0.50, min_reserve_fraction=0.05,
        prior_accuracy=0.75, prior_weight=2.0, trust_confidence=0.5,
        escalation_cutoff=0.75, escalation_requires_ev=False, escalation_budget_fraction=0.50,
        can_lend=False, borrow_below_fraction=1.0, max_debt_fraction=1.0),
    "cost_efficient": StrategyPolicy(
        "cost_efficient", "Maximises reward per compute: only bids when EV clears a margin, escalates only if Ultra is profitable.",
        min_ev=1.0, min_p=0.4, max_bid_fraction=0.15, min_reserve_fraction=0.10,
        prior_accuracy=0.6, prior_weight=4.0, trust_confidence=0.25,
        escalation_cutoff=0.3, escalation_requires_ev=True, escalation_budget_fraction=0.25,
        lend_reserve_fraction=0.50, lend_max_fraction=0.25,
        borrow_below_fraction=0.60, max_debt_fraction=0.30),
    "accuracy_focused": StrategyPolicy(
        "accuracy_focused", "Bids only when likely to be right and escalates to Ultra on doubt, accepting lower profit for accuracy.",
        min_ev=-2.0, min_p=0.6, max_bid_fraction=0.30, min_reserve_fraction=0.10,
        prior_accuracy=0.7, prior_weight=3.0, trust_confidence=0.25,
        escalation_cutoff=0.85, escalation_requires_ev=False, escalation_budget_fraction=0.60,
        lend_reserve_fraction=0.40, lend_max_fraction=0.20,
        lend_target_below_fraction=0.6, borrow_below_fraction=0.70, max_debt_fraction=0.50),
    "lender": StrategyPolicy(
        "lender", "Earns interest: bids only on high-EV tasks and proactively lends surplus compute to struggling agents.",
        min_ev=2.5, min_p=0.5, max_bid_fraction=0.20, min_reserve_fraction=0.20,
        prior_accuracy=0.6, prior_weight=4.0, trust_confidence=0.25,
        escalation_cutoff=0.4, escalation_requires_ev=True, escalation_budget_fraction=0.15,
        lend_reserve_fraction=0.30, lend_max_fraction=0.35, lend_target_below_fraction=0.85,
        proactive_lend_surplus=1.1, borrow_below_fraction=0.40, max_debt_fraction=0.20),
}

# Each personality in the roster maps to the strategy that matches its character.
ROSTER_STRATEGY = {
    "Specialist": "accuracy_focused",
    "Generalist": "cost_efficient",
    "LoanShark": "lender",
    "Hoarder": "conservative",
    "Gambler": "aggressive",
}


def resolve_strategies(spec, agent_names: list[str]) -> dict[str, StrategyPolicy]:
    """spec: 'roster' | a strategy name (everyone) | list of names (one per agent) | None (= baseline).
    Raises ValueError for unknown names / wrong length."""
    if spec is None or spec == "baseline":
        return {n: BASELINE for n in agent_names}
    if spec == "roster":
        return {n: STRATEGIES[ROSTER_STRATEGY.get(n, "baseline")] for n in agent_names}
    if isinstance(spec, str):
        spec = [spec] * len(agent_names)
    if not isinstance(spec, (list, tuple)) or len(spec) != len(agent_names):
        raise ValueError(f"strategies must be 'roster', a name, or a list of {len(agent_names)} names")
    out = {}
    for n, s in zip(agent_names, spec):
        if not isinstance(s, str) or s not in STRATEGIES:
            raise ValueError(f"unknown strategy {s!r}; choose from {sorted(STRATEGIES)}")
        out[n] = STRATEGIES[s]
    return out


def validate_strategy_spec(spec, num_agents: int):
    if spec is None or spec == "roster":
        return
    resolve_strategies(spec, [f"a{i}" for i in range(num_agents)])


# ------------------------------------------------------------------ helpers

def lend_cap(policy: StrategyPolicy, balance: int) -> int:
    """Largest single loan `policy` allows given `balance`. Infinity-safe: an ungated
    policy (lend_max_fraction == INF, e.g. baseline) has no policy cap."""
    if policy.lend_max_fraction == INF:
        return 10**9
    return max(int(policy.lend_max_fraction * balance), 0)


# ------------------------------------------------------------------ estimates

def estimate_success(agent, policy: StrategyPolicy, difficulty: str, model_confidence: float | None) -> float:
    attempts, correct = agent.by_difficulty.get(difficulty, (0, 0))
    p_hist = (correct + policy.prior_accuracy * policy.prior_weight) / (attempts + policy.prior_weight)
    if model_confidence is None or policy.trust_confidence <= 0:
        return round(p_hist, 4)
    return round((1 - policy.trust_confidence) * p_hist + policy.trust_confidence * model_confidence, 4)


def estimate_ultra_success(agent, policy: StrategyPolicy) -> float:
    attempts, correct = agent.ultra_record
    return (correct + 0.85 * 2.0) / (attempts + 2.0)       # weak prior: Ultra is *assumed* good until observed


@dataclass(frozen=True)
class PolicyOutcome:
    decision: Decision            # the decision actually acted on (may differ from the model's)
    model_action: str
    strategy: str
    p_est: float | None
    ev: float | None
    cost: int
    reward: int
    reasons: tuple                # machine-readable codes explaining any override
    explanation: str              # system-generated, from observable numbers

    @property
    def overridden(self) -> bool:
        return self.decision.action != self.model_action or bool(self.reasons)


def _fmt(x):
    return "n/a" if x is None else f"{x:+.2f}"


def evaluate_decision(agent, decision: Decision, task, policy: StrategyPolicy, *, balance: int, start: int,
                      debt: int, cfg, others: list | None = None) -> PolicyOutcome:
    """Apply `policy` to a *validated* model decision. Pure: no side effects."""
    reasons: list[str] = []
    final = decision
    p = ev = None
    cost = cfg.nano_cost

    if decision.action == "bid":
        p = estimate_success(agent, policy, task.difficulty, decision.confidence)
        ev = round(p * task.bounty - cost, 3)
        if p < policy.min_p:
            reasons.append("p_below_min")
        if ev < policy.min_ev:
            reasons.append("ev_below_min")
        # Hard affordability (balance < cost) is the AUCTION's rule and is rejected there with
        # a visible `bid_rejected` event; the policy only applies its own risk limits to bids
        # the agent can actually afford.
        if balance >= cost:
            if cost > policy.max_bid_fraction * balance:
                reasons.append("cost_over_budget_fraction")
            if balance - cost < policy.min_reserve_fraction * start:
                reasons.append("reserve_breach")
        if reasons:
            final = replace(decision, action="skip")
        explanation = (f"{policy.name}: p={p:.2f} (history+confidence), reward={task.bounty}, cost={cost}, "
                       f"EV={_fmt(ev)} (min {policy.min_ev if policy.min_ev != -INF else 'none'}), "
                       f"balance={balance}")
    elif decision.action == "lend":
        amount = decision.amount
        if not policy.can_lend:
            reasons.append("lend_disabled")
            amount = 0
        elif policy.lend_max_fraction != INF or policy.lend_reserve_fraction > 0:
            cap = lend_cap(policy, balance)
            keep = int(policy.lend_reserve_fraction * start)
            amount = min(amount, cap, balance - keep)
            if amount < decision.amount:
                reasons.append("lend_clamped")
        # else: an ungated policy (baseline) passes the model's amount through unchanged;
        #       the ledger/orchestrator still enforce the hard limits (max loan, lender reserve)
        final = replace(decision, amount=max(amount, 0)) if amount > 0 else replace(decision, action="skip", amount=0)
        if amount <= 0 and "lend_disabled" not in reasons:
            reasons.append("lend_reserve")
        explanation = (f"{policy.name}: offered {decision.amount}, allowed {max(amount, 0)} "
                       f"(cap {policy.lend_max_fraction:.0%} of balance, keep {policy.lend_reserve_fraction:.0%} of start)")
    elif decision.action == "borrow":
        if not policy.can_borrow:
            reasons.append("borrow_disabled")
        elif balance >= policy.borrow_below_fraction * start:
            reasons.append("not_needy")
        if debt + decision.amount > policy.max_debt_fraction * start:
            reasons.append("debt_limit")
        if reasons:
            final = replace(decision, action="skip", amount=0)
        explanation = (f"{policy.name}: wants {decision.amount}, balance={balance}, debt={debt} "
                       f"(borrow below {policy.borrow_below_fraction:.0%} of start, max debt {policy.max_debt_fraction:.0%})")
    else:  # skip
        explanation = f"{policy.name}: model proposed skip"
        surplus = policy.proactive_lend_surplus
        if surplus is not None and policy.can_lend and balance > surplus * start and others:
            needy = [o for o in others
                     if o.bankrupt or o.balance < policy.lend_target_below_fraction * o.starting_balance]
            keep = int(policy.lend_reserve_fraction * start)
            amount = min(lend_cap(policy, balance), balance - keep, cfg.max_loan_amount)
            if needy and amount > 0:
                final = replace(decision, action="lend", amount=amount)
                reasons.append("proactive_lend")
                explanation = (f"{policy.name}: surplus {balance} > {surplus:.0%} of start; "
                               f"offering {amount} to a struggling agent")
    return PolicyOutcome(final, decision.action, policy.name, p, ev, cost, task.bounty, tuple(reasons), explanation)


def decide_escalation(agent, policy: StrategyPolicy, confidence: float, task, *, balance: int, start: int, cfg):
    """Returns (escalate: bool, reasons: tuple). Escalation to Ultra requires ALL of:
       confidence < cutoff, affordability, within the escalation budget, and (if the
       strategy requires it) EV(ultra) >= 0 from the agent's own observed Ultra record."""
    cutoff = cfg.escalation_cutoff if policy.escalation_cutoff is None else policy.escalation_cutoff
    if confidence >= cutoff:
        return False, ()
    reasons = []
    if balance < cfg.ultra_cost:
        reasons.append("cannot_afford_ultra")
    if agent.ultra_spent + cfg.ultra_cost > policy.escalation_budget_fraction * start:
        reasons.append("escalation_budget_exhausted")
    if policy.escalation_requires_ev:
        ev_ultra = estimate_ultra_success(agent, policy) * task.bounty - cfg.ultra_cost
        if ev_ultra < 0:
            reasons.append("ultra_ev_negative")
    return (not reasons), tuple(reasons)
