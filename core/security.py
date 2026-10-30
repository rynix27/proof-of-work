"""Small helpers shared by the logging/event paths so secrets never leak."""
import re

from core import config

_KEY_LIKE = re.compile(r"(?i)\b(?:bearer\s+)?[A-Za-z0-9_\-]{32,}\b")


def redact(text) -> str:
    """Return `text` as a string with the configured API key (and any long
    key-looking token following 'Bearer') removed. Used for every error string
    that is written to the event log, logs, or an API response."""
    if text is None:
        return ""
    s = str(text)
    key = config.NEBIUS_API_KEY
    if key and len(key) >= 8:
        s = s.replace(key, "[REDACTED]")
    s = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9_\-\.]{16,}", r"\1[REDACTED]", s)
    return s
