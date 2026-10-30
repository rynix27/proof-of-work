"""
Strict validation of model output. Syntactically valid JSON is NOT trusted:
every field is type-, range- and finiteness-checked before it can touch the
simulation. Invalid output raises ValidationError; callers fall back to a safe
action (skip) and record the failure.
"""
import json
import math
import re
from dataclasses import dataclass

from core import config

ALLOWED_ACTIONS = ("bid", "skip", "lend", "borrow")
MAX_AMOUNT = 10_000
MAX_REASON_CHARS = 200
MAX_ANSWER_CHARS = 200

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")
_INT_STR_RE = re.compile(r"^\d{1,9}$")


class ValidationError(ValueError):
    """Model output is unusable. `.code` is a short machine-readable reason."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True)
class Decision:
    action: str
    amount: int
    confidence: float | None
    reason: str                              # the model's own explanation: text, not ground truth
    est_probability: float | None = None     # model's estimate of success (optional)
    est_difficulty: str | None = None        # model's estimate: easy | medium | hard (optional)
    warnings: tuple = ()                     # optional fields that were present but invalid (dropped)


def parse_json_object(text) -> dict:
    """Extract one JSON object from raw model text.
    Handles <think> blocks and markdown fences; rejects oversize / non-object /
    malformed output. Never returns a non-dict."""
    if not isinstance(text, str):
        raise ValidationError("not_text", "model returned no text")
    if len(text) > config.MAX_RESPONSE_CHARS:
        raise ValidationError("too_large", f"response is {len(text)} chars (max {config.MAX_RESPONSE_CHARS})")
    cleaned = _THINK_RE.sub("", text).strip()
    cleaned = _FENCE_RE.sub("", cleaned).strip()
    if not cleaned:
        raise ValidationError("empty", "empty response")
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        # tolerate prose around a single object: take the outermost braces
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValidationError("malformed_json", "no JSON object found")
        try:
            obj = json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            raise ValidationError("malformed_json", "invalid JSON")
    if not isinstance(obj, dict):
        raise ValidationError("not_object", "top-level JSON is not an object")
    return obj


def _strict_int(value, field: str, lo: int = 0, hi: int = MAX_AMOUNT) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"bad_{field}", f"{field} must be a number")
    if isinstance(value, int):
        n = value
    elif isinstance(value, float):
        if not math.isfinite(value) or value != int(value):
            raise ValidationError(f"bad_{field}", f"{field} must be a finite whole number")
        n = int(value)
    elif isinstance(value, str):
        if not _INT_STR_RE.match(value.strip()):
            raise ValidationError(f"bad_{field}", f"{field} is not a valid whole number")
        n = int(value.strip())
    else:
        raise ValidationError(f"bad_{field}", f"{field} must be a number")
    if n < lo or n > hi:
        raise ValidationError(f"bad_{field}", f"{field} {n} outside [{lo}, {hi}]")
    return n


def _confidence(value, required: bool):
    if value is None:
        if required:
            raise ValidationError("missing_confidence", "confidence is required")
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("bad_confidence", "confidence must be a number")
    value = float(value)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValidationError("bad_confidence", "confidence must be within [0, 1]")
    return value


def validate_decision(raw) -> Decision:
    """Validate an agent decision dict. Rules:
      action      required, one of ALLOWED_ACTIONS
      confidence  required for 'bid' (finite, 0..1); optional otherwise
      amount      required (> 0) for 'lend'/'borrow'; ignored for bid/skip
      reason      optional string, truncated to MAX_REASON_CHARS
    """
    if not isinstance(raw, dict):
        raise ValidationError("not_object", "decision must be an object")
    action = raw.get("action")
    if not isinstance(action, str) or action.strip().lower() not in ALLOWED_ACTIONS:
        raise ValidationError("bad_action", f"unknown action {str(action)[:30]!r}")
    action = action.strip().lower()

    conf = _confidence(raw.get("confidence"), required=(action == "bid"))

    amount = 0
    if action in ("lend", "borrow"):
        if "amount" not in raw:
            raise ValidationError("missing_amount", f"{action} requires an amount")
        amount = _strict_int(raw["amount"], "amount", lo=1)
    elif "amount" in raw and raw["amount"] is not None:
        # ignored for bid/skip, but garbage is still garbage
        try:
            amount = _strict_int(raw["amount"], "amount", lo=0)
        except ValidationError:
            amount = 0

    reason = raw.get("reason", "")
    if not isinstance(reason, str):
        reason = ""

    # Optional estimate fields: validated like everything else, but an invalid optional
    # field is dropped (and recorded) rather than discarding an otherwise usable decision.
    warnings, est_p, est_d = [], None, None
    if raw.get("estimated_success_probability") is not None:
        try:
            est_p = _confidence(raw["estimated_success_probability"], required=False)
        except ValidationError:
            warnings.append("bad_estimated_success_probability")
    if raw.get("estimated_difficulty") is not None:
        d = raw["estimated_difficulty"]
        if isinstance(d, str) and d.strip().lower() in ("easy", "medium", "hard"):
            est_d = d.strip().lower()
        else:
            warnings.append("bad_estimated_difficulty")
    return Decision(action=action, amount=amount, confidence=conf, reason=reason[:MAX_REASON_CHARS],
                    est_probability=est_p, est_difficulty=est_d, warnings=tuple(warnings))


def validate_answer(raw) -> tuple[str, float | None]:
    """Validate a solve response. Returns (answer_text, confidence|None).
    Numbers are accepted and stringified; lists/objects/None are rejected."""
    if not isinstance(raw, dict):
        raise ValidationError("not_object", "answer response must be an object")
    ans = raw.get("answer")
    if isinstance(ans, bool) or ans is None:
        raise ValidationError("missing_answer", "answer missing")
    if isinstance(ans, (int, float)):
        if isinstance(ans, float) and not math.isfinite(ans):
            raise ValidationError("bad_answer", "answer is not finite")
        ans = str(ans)
    if not isinstance(ans, str):
        raise ValidationError("bad_answer", "answer must be text or a number")
    ans = ans.strip()
    if not ans:
        raise ValidationError("missing_answer", "answer is empty")
    if len(ans) > MAX_ANSWER_CHARS:
        raise ValidationError("bad_answer", "answer too long")
    conf = None
    try:
        conf = _confidence(raw.get("confidence"), required=False)
    except ValidationError:
        conf = None  # a bad self-reported confidence does not invalidate the answer itself
    return ans, conf
