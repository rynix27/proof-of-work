"""
Per-run simulation configuration. Defaults come from core/config.py (read at
call time), every override is validated, unknown keys are rejected. The full
resolved config is written into the `simulation_started` event, so any run can
be reproduced (same config + same seed + mock model => same result).
"""
import math
from dataclasses import asdict, dataclass

from core import config


class ConfigError(ValueError):
    pass


def _default_strategies():
    spec = config.STRATEGY_SET
    return tuple(x.strip() for x in spec.split(",")) if "," in spec else spec


@dataclass(frozen=True)
class SimulationConfig:
    num_agents: int
    max_rounds: int
    starting_balance: int
    nano_cost: int
    ultra_cost: int
    interest_rate: float
    loan_term_rounds: int
    escalation_cutoff: float
    max_loan_amount: int
    max_active_loans: int
    lender_reserve: int
    bankruptcy_threshold: int
    seed: int | None = None
    task_weights: tuple | None = None  # (easy, medium, hard) relative weights
    bounties: tuple = (8, 20, 50)      # reward by difficulty (easy, medium, hard)
    strategies: object = "roster"       # 'roster' | 'baseline' | name | tuple of names (one per agent)

    @property
    def rate_bp(self) -> int:
        """Interest rate in integer basis points (0.15 -> 1500)."""
        return int(round(self.interest_rate * 10000))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task_weights"] = list(self.task_weights) if self.task_weights else None
        d["bounties"] = list(self.bounties)
        d["strategies"] = list(self.strategies) if isinstance(self.strategies, tuple) else self.strategies
        return d

    @classmethod
    def defaults(cls, **overrides) -> "SimulationConfig":
        return cls.from_overrides(overrides)

    @classmethod
    def from_overrides(cls, overrides: dict | None = None) -> "SimulationConfig":
        o = dict(overrides or {})
        base = dict(
            num_agents=5,
            max_rounds=config.MAX_ROUNDS,
            starting_balance=config.STARTING_BALANCE,
            nano_cost=config.NANO_COST,
            ultra_cost=config.ULTRA_COST,
            interest_rate=config.LOAN_INTEREST_RATE,
            loan_term_rounds=config.LOAN_TERM_ROUNDS,
            escalation_cutoff=config.CONFIDENCE_ESCALATION_CUTOFF,
            max_loan_amount=config.MAX_LOAN_AMOUNT,
            max_active_loans=config.MAX_ACTIVE_LOANS_PER_BORROWER,
            lender_reserve=config.MIN_LENDER_RESERVE,
            bankruptcy_threshold=config.BANKRUPTCY_THRESHOLD,
            seed=config.SIMULATION_SEED,
            task_weights=None,
            bounties=(8, 20, 50),
            strategies=_default_strategies(),
        )
        unknown = set(o) - set(base)
        if unknown:
            raise ConfigError(f"unknown config key(s): {sorted(unknown)}")
        for k, v in o.items():
            if v is not None or k in ("seed", "task_weights"):
                base[k] = v

        def integer(name, lo, hi):
            v = base[name]
            if isinstance(v, bool) or not isinstance(v, int):
                raise ConfigError(f"{name} must be an integer")
            if not lo <= v <= hi:
                raise ConfigError(f"{name} must be between {lo} and {hi}")

        def real(name, lo, hi):
            v = base[name]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ConfigError(f"{name} must be a finite number")
            if not lo <= v <= hi:
                raise ConfigError(f"{name} must be between {lo} and {hi}")
            base[name] = float(v)

        integer("num_agents", 2, 5)
        integer("max_rounds", 1, 1000)
        integer("starting_balance", 10, 10_000)
        integer("nano_cost", 1, 100)
        integer("ultra_cost", 1, 1000)
        real("interest_rate", 0.0, 1.0)
        integer("loan_term_rounds", 1, 100)
        real("escalation_cutoff", 0.0, 1.0)
        integer("max_loan_amount", 1, 10_000)
        integer("max_active_loans", 1, 20)
        integer("lender_reserve", 0, 1000)
        integer("bankruptcy_threshold", 0, 1000)
        if base["ultra_cost"] < base["nano_cost"]:
            raise ConfigError("ultra_cost must be >= nano_cost")
        if base["seed"] is not None:
            integer("seed", 0, 2**31 - 1)
        tw = base["task_weights"]
        if tw is not None:
            if (not isinstance(tw, (list, tuple)) or len(tw) != 3
                    or any(isinstance(w, bool) or not isinstance(w, (int, float))
                           or not math.isfinite(w) or w < 0 for w in tw)
                    or sum(tw) <= 0):
                raise ConfigError("task_weights must be [easy, medium, hard], non-negative, not all zero")
            base["task_weights"] = tuple(float(w) for w in tw)
        bt = base["bounties"]
        if (not isinstance(bt, (list, tuple)) or len(bt) != 3
                or any(isinstance(b, bool) or not isinstance(b, int) or not 0 <= b <= 1000 for b in bt)):
            raise ConfigError("bounties must be [easy, medium, hard], integers between 0 and 1000")
        base["bounties"] = tuple(bt)
        from core.strategies import validate_strategy_spec
        st = base["strategies"]
        if isinstance(st, list):
            st = base["strategies"] = tuple(st)
        try:
            validate_strategy_spec(st, base["num_agents"])
        except (ValueError, KeyError, TypeError) as e:
            raise ConfigError(str(e))
        return cls(**base)
