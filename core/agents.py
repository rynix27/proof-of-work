"""
Agents are personality-seeded, not hard-coded strategies. Each agent's
seed_prompt shapes how it reasons about bidding/lending, but the actual
decision comes from a real Nano call each round -- the personalities bias
behavior, they don't script it. This is what makes the emergent behavior
real rather than a fixed animation.
"""
from dataclasses import dataclass, field

from core import config


@dataclass
class Agent:
    name: str
    seed_prompt: str
    balance: int = field(default_factory=lambda: config.STARTING_BALANCE)
    bankrupt: bool = False
    tasks_won: int = 0
    tasks_correct: int = 0
    ultra_calls: int = 0
    nano_calls: int = 0
    # --- accounting / evaluation counters (all integers) ---
    starting_balance: int = field(default_factory=lambda: config.STARTING_BALANCE)
    graded_attempts: int = 0     # solves on tasks that have a ground truth
    invalid_decisions: int = 0   # model output rejected by core.validation
    invalid_answers: int = 0     # successful call but unusable answer
    compute_spent: int = 0       # simulated compute-seconds charged
    rewards_earned: int = 0      # bounties received
    escalations: int = 0
    strategy: str = "baseline"                                   # name of the StrategyPolicy in force
    ultra_spent: int = 0                                         # simulated compute spent on Ultra
    by_difficulty: dict = field(default_factory=dict)            # difficulty -> [graded attempts, correct]
    ultra_record: list = field(default_factory=lambda: [0, 0])   # [graded Ultra attempts, correct]

    def net_worth(self) -> int:
        return self.balance

    def go_bankrupt(self):
        self.bankrupt = True

    def revive(self, amount: int):
        self.balance = amount
        self.bankrupt = False


# Five personalities, deliberately different enough that they should
# converge on different strategies over a run. Rewrite these based on what
# you observe in tuning (Week 2 / Phase 2) -- a personality that never does
# anything interesting should be cut or rewritten, not kept out of loyalty.
AGENT_ROSTER = [
    Agent(
        name="Specialist",
        seed_prompt=(
            "You are Specialist, a careful reasoning agent with a compute budget. "
            "You only bid on tasks you are genuinely confident you can solve correctly. "
            "You skip tasks outside your strength rather than bidding recklessly. "
            "You rarely lend money to others, preferring to hoard for high-value tasks."
        ),
    ),
    Agent(
        name="Generalist",
        seed_prompt=(
            "You are Generalist, a reasoning agent with a compute budget. "
            "You bid often and cheaply on almost every task, betting on volume "
            "over precision. You are willing to borrow small amounts if it lets "
            "you take a bid you'd otherwise have to skip."
        ),
    ),
    Agent(
        name="LoanShark",
        seed_prompt=(
            "You are LoanShark, a reasoning agent with a compute budget. "
            "You rarely solve tasks yourself. Instead you look for agents who are "
            "low on balance or about to go bankrupt and lend to them at interest. "
            "Your goal is to grow your balance primarily through lending, not solving."
        ),
    ),
    Agent(
        name="Hoarder",
        seed_prompt=(
            "You are Hoarder, a reasoning agent with a compute budget. "
            "You almost never spend. You skip most tasks, saving your balance for "
            "rare, very high-bounty tasks. You never lend and rarely borrow."
        ),
    ),
    Agent(
        name="Gambler",
        seed_prompt=(
            "You are Gambler, a reasoning agent with a compute budget. "
            "You take risks other agents avoid: bidding on hard tasks even with low "
            "confidence, escalating to your expensive deep-reasoning tier often, and "
            "borrowing aggressively when you think a task is worth it."
        ),
    ),
]


def record_result(agent, difficulty: str, tier: str, correct: bool, cost: int):
    """Update the agent's own track record after a GRADED solve (information the agent
    legitimately has when it makes later decisions)."""
    rec = agent.by_difficulty.setdefault(difficulty, [0, 0])
    rec[0] += 1
    rec[1] += int(correct)
    if tier == "ultra":
        agent.ultra_record[0] += 1
        agent.ultra_record[1] += int(correct)


def fresh_roster(num_agents: int | None = None, starting_balance: int | None = None) -> list[Agent]:
    """Brand-new agents so each run starts clean. `num_agents` takes the first N
    personalities of the roster (default: all)."""
    sb = config.STARTING_BALANCE if starting_balance is None else starting_balance
    roster = AGENT_ROSTER if num_agents is None else AGENT_ROSTER[:num_agents]
    return [Agent(name=a.name, seed_prompt=a.seed_prompt, balance=sb, starting_balance=sb)
            for a in roster]
