"""
Thin wrapper around Nebius Token Factory (OpenAI-compatible chat completions),
using the official `openai` SDK pointed at NEBIUS_BASE_URL.

ACCOUNTING POLICY FOR FAILED CALLS (simulated compute, NOT real provider billing)
  Every failure raises ModelCallError carrying `.kind` and `.charged`:
    kind        meaning                                   charged
    ----------  ----------------------------------------  -------
    config      missing key / bad setup, nothing sent      no
    auth        401/403 rejected before processing         no
    rejected    other 4xx rejected before processing       no
    rate_limit  429 after all retries                      no
    server      5xx after all retries                      no
    connection  could not reach provider after retries     no
    timeout     request sent, outcome unknown              YES (once, not per retry)
    empty       provider answered with no usable content   YES (it processed it)
  "charged" means the orchestrator debits the simulated compute cost once. Real
  provider billing is never inspected; the dashboard labels the numbers as
  simulated compute-seconds.

BOUNDS
  The SDK's own retries are disabled (max_retries=0) so ours are the only ones;
  every logical call is bounded by MODEL_DEADLINE_SECONDS wall clock.

MOCK MODE
  USE_FAKE_MODEL=1 uses a seeded local fake (no network). It is never logged as
  training data and the app labels itself "MOCK" everywhere.
"""
import json
import logging
import random
import re
import threading
import time
from pathlib import Path

from core import config, events
from core.security import redact
from core.validation import ValidationError, parse_json_object

log = logging.getLogger("proof_of_work.model")

_client = None
_client_lock = threading.Lock()

TRAINING_LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "training" / "calls.jsonl"
_training_lock = threading.Lock()

# Provider-reported token usage, recorded per thread so concurrent runs never mix.
# This is REAL measurement (from the API response) and is kept separate from the
# *simulated* compute-seconds the economy charges. The mock model records nothing.
_usage_local = threading.local()

_fake_rng = random.Random()
_fake_lock = threading.Lock()
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class ModelCallError(RuntimeError):
    def __init__(self, message: str, kind: str, charged: bool = False):
        super().__init__(message)
        self.kind = kind
        self.charged = charged


def _record_usage(response, model: str, tier: str):
    """Capture token counts from a provider response when present and well-formed."""
    items = getattr(_usage_local, "items", None)
    if items is None:
        items = _usage_local.items = []
    usage = getattr(response, "usage", None)
    pt, ct = getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
    ok = all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (pt, ct))
    items.append({"model": model, "tier": tier, "input_tokens": pt if ok else None,
                  "output_tokens": ct if ok else None, "reported": ok, "latency_s": None})


def take_usage() -> dict | None:
    """Aggregate and clear the calling thread's usage since the last call. None if no live
    calls were made (always the case in mock mode)."""
    items = getattr(_usage_local, "items", None)
    if not items:
        return None
    _usage_local.items = []
    out = {"calls": len(items), "calls_with_usage": sum(1 for i in items if i["reported"]),
           "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0, "by_tier": {}}
    for i in items:
        t = out["by_tier"].setdefault(i["tier"], {"model": i["model"], "calls": 0, "input_tokens": 0,
                                                  "output_tokens": 0, "reported": 0})
        t["calls"] += 1
        if i["reported"]:
            t["reported"] += 1
            t["input_tokens"] += i["input_tokens"]
            t["output_tokens"] += i["output_tokens"]
            out["input_tokens"] += i["input_tokens"]
            out["output_tokens"] += i["output_tokens"]
        out["latency_s"] += i["latency_s"] or 0.0
    out["latency_s"] = round(out["latency_s"], 3)
    return out


def estimate_cost_usd(usage: dict | None) -> float | None:
    """Estimated USD cost = provider-reported tokens x the prices YOU configured
    (NANO/ULTRA_PRICE_IN/OUT_PER_MTOK). Returns None -- never a guess -- if there is no
    usage or a needed price is not configured."""
    if not usage:
        return None
    total = 0.0
    for tier, t in usage["by_tier"].items():
        pin, pout = config.PRICE_IN_PER_MTOK.get(tier), config.PRICE_OUT_PER_MTOK.get(tier)
        if pin is None or pout is None:
            return None
        total += t["input_tokens"] / 1e6 * pin + t["output_tokens"] / 1e6 * pout
    return round(total, 6)


def seed_fake(seed: int | None):
    """(Re)seed the mock model. Same seed => same mock behaviour."""
    with _fake_lock:
        _fake_rng.seed(seed)


def _get_client():
    global _client
    with _client_lock:
        if _client is None:
            if not config.NEBIUS_API_KEY:
                raise ModelCallError(
                    "NEBIUS_API_KEY is not set. Set it in .env, or use USE_FAKE_MODEL=1 for mock mode.",
                    kind="config")
            from openai import OpenAI
            # max_retries=0: the SDK defaults to 2, which would multiply with our own retries.
            _client = OpenAI(api_key=config.NEBIUS_API_KEY, base_url=config.NEBIUS_BASE_URL,
                             max_retries=0)
        return _client


def _fake_completion(full_text: str, tier: str) -> str:
    """Seeded fake model for offline dev/tests. `full_text` is system+user prompt."""
    time.sleep(0.0 if config.MODEL_TIMEOUT_SECONDS <= 0 else 0.001)
    with _fake_lock:
        r = _fake_rng
        if '"action"' in full_text:
            action = r.choices(["bid", "skip", "lend", "borrow"], weights=[55, 25, 10, 10])[0]
            return json.dumps({
                "action": action,
                "amount": r.choice([5, 8, 12, 20]),
                "confidence": round(r.uniform(0.3, 0.95), 2),
                "reason": "fake-mode heuristic decision",
            })
        if '"answer"' in full_text:
            from core.tasks import answer_for_prompt
            real_answer = None
            for line in full_text.splitlines():
                if line.startswith("Question: "):
                    real_answer = answer_for_prompt(line[len("Question: "):].strip())
                    break
            answer = real_answer if (real_answer and r.random() < 0.7) else "wrong-guess"
            return json.dumps({"answer": answer, "confidence": round(r.uniform(0.3, 0.95), 2)})
        if "epitaph" in full_text.lower():
            return r.choice(_FAKE_EPITAPHS)
        if "sportscaster" in full_text.lower():
            return r.choice(_FAKE_COMMENTARY)
    return "fake-mode response"


_FAKE_EPITAPHS = [
    "Never spent a compute-second it didn't have to. Died rich in principle, poor in practice.",
    "Bid big, lost bigger. At least it went down swinging.",
    "Bankrupted by its own confidence interval.",
    "It lent to everyone and was repaid by no one.",
    "Optimized for the wrong objective function, right to the end.",
]

_FAKE_COMMENTARY = [
    "And there it is folks, a bold bid from the back of the pack!",
    "The lending market is heating up -- somebody's about to regret this.",
    "This economy is more volatile than a meme stock on a Friday.",
    "We may be witnessing the beginning of a monopoly here, ladies and gentlemen.",
    "Confidence is high, funds are low -- a classic setup for disaster.",
]


def _resolve_model(tier: str) -> str:
    """Model slug for a tier, preferring a fine-tuned override when configured."""
    if tier == "nano":
        return config.FINE_TUNED_NANO_MODEL or config.NANO_MODEL
    return config.FINE_TUNED_ULTRA_MODEL or config.ULTRA_MODEL


def _log_training_call(tier: str, system: str, prompt: str, response_text: str, meta: dict,
                       model: str | None = None):
    if not config.LOG_TRAINING_DATA:
        return
    record = {
        "ts": time.time(),
        "run_id": events.thread_run_id() or events.current_run_id(),
        "tier": tier,
        "model": model or _resolve_model(tier),
        "system": system,
        "prompt": prompt,
        "response": response_text,
        **meta,
    }
    try:
        with _training_lock:
            TRAINING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TRAINING_LOG_PATH, "a") as f:
                f.write(json.dumps(record) + "\n")
    except OSError as e:
        log.error("training-log write failed: %s", redact(e))


def call_model(prompt: str, tier: str, system: str = "", meta: dict = None,
               model: str | None = None) -> tuple[str, float]:
    """tier: "nano" or "ultra". `model` pins an explicit model id (used by
    training/evaluate.py to compare base vs fine-tuned on identical inputs).
    Returns (response_text, latency_seconds).
    Raises ModelCallError on failure (see policy table in the module docstring)."""
    model = model or _resolve_model(tier)
    start = time.time()

    if config.USE_FAKE_MODEL:
        text = _fake_completion(f"{system}\n{prompt}", tier)
        return text, time.time() - start

    client = _get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    text = _call_with_retry(client, model, messages, tier)
    latency = time.time() - start
    if getattr(_usage_local, "items", None):
        _usage_local.items[-1]["latency_s"] = round(latency, 3)
    _log_training_call(tier, system, prompt, text, meta or {}, model=model)
    return text, latency


def _call_with_retry(client, model: str, messages: list, tier: str,
                     max_retries: int | None = None, timeout_seconds: float | None = None) -> str:
    """One logical model call with bounded, classified retries."""
    import openai

    max_retries = max_retries or config.MODEL_MAX_RETRIES
    timeout_seconds = timeout_seconds or config.MODEL_TIMEOUT_SECONDS
    deadline = time.monotonic() + config.MODEL_DEADLINE_SECONDS

    last_error = None
    last_kind = "unknown"
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.4 if tier == "nano" else 0.2,
                max_tokens=config.MODEL_MAX_TOKENS,
                timeout=timeout_seconds,
            )
            text = _extract_text(response)
            _record_usage(response, model, tier)
            return text
        except ModelCallError:
            raise
        except openai.RateLimitError as e:
            last_error, last_kind = e, "rate_limit"
            delay = 2 ** attempt * 2
        except openai.APITimeoutError as e:
            last_error, last_kind = e, "timeout"
            delay = 1 + attempt
        except openai.APIConnectionError as e:
            last_error, last_kind = e, "connection"
            delay = 1 + attempt
        except openai.APIStatusError as e:
            if e.status_code >= 500:
                last_error, last_kind = e, "server"
                delay = 1 + attempt
            else:
                kind = "auth" if e.status_code in (401, 403) else "rejected"
                raise ModelCallError(
                    f"Token Factory returned {e.status_code}: {redact(e.message)}. "
                    "Check NEBIUS_API_KEY, NEBIUS_BASE_URL, and the model slug in .env.",
                    kind=kind) from e
        # transient failure: sleep and retry, unless that would blow the deadline
        if attempt < max_retries - 1:
            if time.monotonic() + delay > deadline:
                break
            time.sleep(delay)

    raise ModelCallError(
        f"Token Factory call failed after {attempt + 1} attempt(s) ({last_kind}): {redact(last_error)}",
        kind=last_kind, charged=(last_kind == "timeout"))


def _extract_text(response) -> str:
    try:
        choice = response.choices[0]
        content = choice.message.content
    except (AttributeError, IndexError, TypeError) as e:
        raise ModelCallError(f"unexpected response shape from provider: {e}", kind="empty", charged=True)
    if not isinstance(content, str) or not _THINK_RE.sub("", content).strip():
        finish = getattr(choice, "finish_reason", None)
        hint = (" The reasoning trace probably used the whole token budget; raise MODEL_MAX_TOKENS."
                if finish == "length" else "")
        raise ModelCallError(f"provider returned empty content (finish_reason={finish}).{hint}",
                             kind="empty", charged=True)
    return content


def call_model_json(prompt: str, tier: str, system: str = "", meta: dict = None,
                    model: str | None = None) -> dict:
    """Call the model and parse a JSON object. Provider failures raise
    ModelCallError (so the caller can apply the charging policy); an
    unparseable *successful* response returns {"_parse_error": True, ...}."""
    text, latency = call_model(prompt, tier, system=system, meta=meta, model=model)
    try:
        return {**parse_json_object(text), "_latency": latency, "_tier": tier}
    except ValidationError as e:
        return {"_parse_error": True, "_error": e.code, "_raw": text[:500],
                "_latency": latency, "_tier": tier}
