"""
Central config. Everything else in the codebase reads from here so that
swapping API keys, base URLs, or model slugs never means hunting through
multiple files.

All values come from environment variables (optionally via a local .env).
Numeric variables are parsed defensively: a malformed value falls back to
the documented default and is reported by `config_warnings()` instead of
crashing the process at import time.
"""
import os

from dotenv import load_dotenv

load_dotenv()

_WARNINGS: list[str] = []


def _env_int(name: str, default: int, lo: int | None = None, hi: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        _WARNINGS.append(f"{name}={raw!r} is not an integer; using default {default}")
        return default
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        _WARNINGS.append(f"{name}={value} outside [{lo}, {hi}]; using default {default}")
        return default
    return value


def _env_float(name: str, default: float, lo: float | None = None, hi: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError:
        _WARNINGS.append(f"{name}={raw!r} is not a number; using default {default}")
        return default
    if value != value or value in (float("inf"), float("-inf")):
        _WARNINGS.append(f"{name}={raw!r} is not finite; using default {default}")
        return default
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        _WARNINGS.append(f"{name}={value} outside [{lo}, {hi}]; using default {default}")
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def config_warnings() -> list[str]:
    return list(_WARNINGS)


# --- Provider -------------------------------------------------------------
NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY", "")
# Verified against docs.tokenfactory.nebius.com (quickstart / api-reference).
# The older api.studio.nebius.* host belongs to the previous "AI Studio" brand.
NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/")

# Model IDs as listed in the Nebius token-factory-cookbook Nemotron page
# (playground links). Verify against `GET {NEBIUS_BASE_URL}models` with your
# own key before a live run: `python -m training.check_provider`.
NANO_MODEL = os.getenv("NANO_MODEL", "nvidia/nvidia-nemotron-3-nano-30b-a3b")
ULTRA_MODEL = os.getenv("ULTRA_MODEL", "nvidia/Nemotron-3-Ultra-550b-a55b")

# Once you've fine-tuned *and deployed* a model (see docs/DEPLOYMENT.md and
# training/finetune.py), put its served model name here and every call for that
# tier switches to it. Leave blank to keep using the base model.
FINE_TUNED_NANO_MODEL = os.getenv("FINE_TUNED_NANO_MODEL", "")
FINE_TUNED_ULTRA_MODEL = os.getenv("FINE_TUNED_ULTRA_MODEL", "")

# Mock mode: no network, deterministic (seeded) fake model. Defaults to ON so a
# fresh clone can never spend money by accident. Set USE_FAKE_MODEL=0 for live.
USE_FAKE_MODEL = os.getenv("USE_FAKE_MODEL", "1") == "1"

# Append every real (non-mock) call to data/training/calls.jsonl.
LOG_TRAINING_DATA = os.getenv("LOG_TRAINING_DATA", "1") == "1"

# --- Model call bounds ------------------------------------------------------
# Nemotron 3 models on Token Factory are *reasoning* models; a small token budget
# can be consumed entirely by the reasoning trace, leaving empty `content`
# (see github.com/nebius/api/issues/211). 300 was too small; 1024 is a starting
# point that MUST be validated against the live provider.
MODEL_MAX_TOKENS = _env_int("MODEL_MAX_TOKENS", 1024, 16, 32768)
MODEL_TIMEOUT_SECONDS = _env_float("MODEL_TIMEOUT_SECONDS", 20.0, 1.0, 300.0)
MODEL_MAX_RETRIES = _env_int("MODEL_MAX_RETRIES", 3, 1, 8)
# Hard wall-clock bound for one logical call including every retry and sleep.
MODEL_DEADLINE_SECONDS = _env_float("MODEL_DEADLINE_SECONDS", 60.0, 1.0, 600.0)
MAX_RESPONSE_CHARS = _env_int("MAX_RESPONSE_CHARS", 4000, 100, 100000)

# --- Economy tuning knobs ----------------------------------------------------
# All balances, costs, bounties and loans are INTEGER compute-seconds. Using
# integers everywhere means no floating-point drift in the ledger.
STARTING_BALANCE = 100          # compute-seconds each agent starts with
ROUND_INTERVAL_SECONDS = 8      # how often a new task appears
NANO_COST = 1                   # simulated cost of a Nano solve
ULTRA_COST = 12                 # simulated cost of an Ultra solve (escalation)
BANKRUPTCY_THRESHOLD = 0        # balance at/below this = frozen
CONFIDENCE_ESCALATION_CUTOFF = 0.6  # bid confidence below this => escalate to Ultra
MAX_ROUNDS = 200                # safety cap so a run can't loop forever

# Lending: simple interest, single period.
#   interest = ceil(principal * LOAN_INTEREST_RATE)   (integer basis-point math)
#   total owed = principal + interest   (20 @ 15% -> 23)
LOAN_INTEREST_RATE = 0.15
LOAN_TERM_ROUNDS = 5            # a loan falls due this many rounds after creation
MAX_LOAN_AMOUNT = 50            # per-loan cap
MAX_ACTIVE_LOANS_PER_BORROWER = 3
# A lender must keep at least this much after lending, so lending cannot
# silently bankrupt the lender.
MIN_LENDER_RESERVE = NANO_COST

# Which strategy each agent plays: "roster" (each personality gets the strategy that
# matches it -- the default), "baseline" (no policy gating: the control group), a single
# strategy name for everyone, or a comma-separated list with one name per agent.
STRATEGY_SET = os.getenv("STRATEGY_SET", "roster").strip() or "roster"


def _price(name):
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else None
    except ValueError:
        _WARNINGS.append(f"{name}={raw!r} is not a number; ignoring")
        return None


# Optional provider pricing (USD per 1M tokens) used ONLY to turn provider-reported token
# usage into an *estimated* cost. Unset by default: pricing is not hardcoded because it
# changes and could not be verified; copy it from Nebius's pricing page if you want estimates.
PRICE_IN_PER_MTOK = {"nano": _price("NANO_PRICE_IN_PER_MTOK"), "ultra": _price("ULTRA_PRICE_IN_PER_MTOK")}
PRICE_OUT_PER_MTOK = {"nano": _price("NANO_PRICE_OUT_PER_MTOK"), "ultra": _price("ULTRA_PRICE_OUT_PER_MTOK")}

# --- Flavor / narrative features ---
COMMENTARY_INTERVAL_ROUNDS = 3

# --- Reproducibility ------------------------------------------------------
# None => a fresh random seed per run (recorded in the simulation_started event).
_seed_raw = os.getenv("SIMULATION_SEED", "").strip()
try:
    SIMULATION_SEED = int(_seed_raw) if _seed_raw else None
except ValueError:
    _WARNINGS.append(f"SIMULATION_SEED={_seed_raw!r} is not an integer; ignoring")
    SIMULATION_SEED = None

# --- Server / security ------------------------------------------------------
# Local development: loopback bind, open controls.
# Public deployment: PUBLIC_MODE=1 fails CLOSED -- control endpoints are
# disabled unless CONTROL_TOKEN is set, in which case they require it.
HOST = os.getenv("HOST", "127.0.0.1")
PORT = _env_int("PORT", 5000, 1, 65535)
PUBLIC_MODE = _env_bool("PUBLIC_MODE", False)
CONTROL_TOKEN = os.getenv("CONTROL_TOKEN", "")
TRUST_PROXY = _env_bool("TRUST_PROXY", False)
MAX_REQUEST_BYTES = _env_int("MAX_REQUEST_BYTES", 16 * 1024, 256, 10 * 1024 * 1024)
ENABLE_INJECT = _env_bool("ENABLE_INJECT", True)
# Extra origins allowed to POST (comma-separated, e.g. "https://demo.example.com").
# The request's own Host is always allowed; any other Origin is rejected (CSRF guard).
ALLOWED_ORIGINS = [o.strip().rstrip("/") for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
# Wrong control-token attempts allowed per client per minute before 429.
AUTH_FAIL_MAX_PER_MINUTE = _env_int("AUTH_FAIL_MAX_PER_MINUTE", 10, 1, 1000)
# Per-client cap on read endpoints (/state etc.) per minute: resource-exhaustion guard.
READ_MAX_PER_MINUTE = _env_int("READ_MAX_PER_MINUTE", 600, 10, 100000)
# "Ask the economy" spends real model calls in live mode, so it is rate limited.
INJECT_MAX_PER_MINUTE_PER_IP = _env_int("INJECT_MAX_PER_MINUTE_PER_IP", 5, 1, 1000)
INJECT_MAX_PER_HOUR_GLOBAL = _env_int("INJECT_MAX_PER_HOUR_GLOBAL", 60, 1, 100000)
INJECT_MAX_PENDING = _env_int("INJECT_MAX_PENDING", 10, 1, 1000)


def mode() -> str:
    """'mock' when the fake model is active, otherwise 'live'. Read at call
    time (not import time) so tests and the CLI can flip it."""
    return "mock" if USE_FAKE_MODEL else "live"
