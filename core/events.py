"""
Structured event log. Every meaningful thing that happens in a round
(a bid, a task result, a loan, a bankruptcy, a bailout) is appended here
as one JSON object per line. The dashboard tails this file -- it's the single
source of truth shared between the simulation and the UI.

SCHEMA (v2, backward compatible with v1 -- v1 events simply lack the new keys):
    ts        float   unix time
    type      str     event type
    run_id    str     one id per simulation run (new on every reset)
    seq       int     1-based, strictly increasing within a run
    event_id  str     f"{run_id}-{seq}"
    v         int     schema version
    ...       type-specific fields

CONCURRENCY MODEL
  * One process-wide RLock guards the log file, the run id and the sequence
    counter. Readers take the same lock, so a reader never observes a
    half-written line from a writer in this process.
  * Each simulation thread *binds* the run id it owns (bind_run). Once a reset
    has replaced the current run id, anything that stale thread still tries to
    write (events, replays, leaderboard rows) is silently dropped. This is what
    prevents events from an old run leaking into a new one without needing to
    join (and possibly hang on) the old thread.
  * Disk-write failures are logged, counted (`write_failures`) and surfaced via
    /healthz; they never raise into the simulation thread.
"""
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path

from core.security import redact

log = logging.getLogger("proof_of_work.events")

SCHEMA_VERSION = 2

LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "events.jsonl"
REPLAY_DIR = Path(__file__).resolve().parent.parent / "data" / "replays"
LEADERBOARD_PATH = Path(__file__).resolve().parent.parent / "data" / "leaderboard.jsonl"

_lock = threading.RLock()
_local = threading.local()
_current_run_id = None
_seq = 0
write_failures = 0

_REPLAY_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}\.jsonl$")


# --- run binding ------------------------------------------------------------

def bind_run(run_id: str | None):
    """Mark the calling thread as owned by `run_id`. Pass None to unbind."""
    _local.run_id = run_id


def thread_run_id() -> str | None:
    return getattr(_local, "run_id", None)


def is_stale() -> bool:
    """True when the calling thread is bound to a run that has since been
    replaced by a reset. Unbound threads (tests, CLI, server) are never stale."""
    bound = thread_run_id()
    return bound is not None and bound != _current_run_id


# --- run lifecycle ------------------------------------------------------------

def reset_log() -> str:
    """Start a new run: new run id, sequence restarts, log truncated."""
    global _current_run_id, _seq, write_failures
    with _lock:
        _current_run_id = uuid.uuid4().hex[:12]
        _seq = 0
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            LOG_PATH.write_text("")
        except OSError as e:
            write_failures += 1
            log.error("could not truncate event log %s: %s", LOG_PATH, redact(e))
        return _current_run_id


def current_run_id() -> str:
    global _current_run_id
    with _lock:
        if _current_run_id is None:
            _current_run_id = uuid.uuid4().hex[:12]
        return _current_run_id


# --- writing ------------------------------------------------------------------

def emit(event_type: str, **fields):
    """Append one event. Returns the event dict, or None if the calling thread
    belongs to a superseded run (the event is dropped on purpose)."""
    global _seq, write_failures
    with _lock:
        if is_stale():
            return None
        _seq += 1
        run_id = current_run_id()
        if "error" in fields and fields["error"] is not None:
            fields["error"] = redact(fields["error"])
        event = {
            "ts": time.time(), "type": event_type, "run_id": run_id,
            "seq": _seq, "event_id": f"{run_id}-{_seq}", "v": SCHEMA_VERSION,
            **fields,
        }
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
        except OSError as e:
            write_failures += 1
            log.error("event write failed (%s): %s", event_type, redact(e))
        return event


# --- reading ------------------------------------------------------------------

def read_jsonl_tolerant(path: Path) -> tuple[list[dict], int]:
    """Parse a JSONL file, skipping blank/corrupt/non-object lines.
    Returns (records, skipped_count). Never raises on malformed content."""
    records, skipped = [], 0
    try:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    skipped += 1
                    continue
                if isinstance(obj, dict):
                    records.append(obj)
                else:
                    skipped += 1
    except FileNotFoundError:
        return [], 0
    except OSError as e:
        log.error("could not read %s: %s", path, redact(e))
        return [], 0
    if skipped:
        log.warning("%s: skipped %d malformed line(s)", path, skipped)
    return records, skipped


def read_all() -> list[dict]:
    with _lock:
        records, _ = read_jsonl_tolerant(LOG_PATH)
    return records


def read_recent(n=200):
    return read_all()[-n:]


# --- replays ------------------------------------------------------------------

def replay_path(filename: str) -> Path:
    """Validate a replay filename and return its path inside REPLAY_DIR.
    Raises ValueError for anything that isn't a plain *.jsonl name."""
    if not isinstance(filename, str) or not _REPLAY_NAME_RE.fullmatch(filename):
        raise ValueError("invalid replay path")
    path = REPLAY_DIR / filename
    if path.resolve().parent != REPLAY_DIR.resolve():
        raise ValueError("invalid replay path")
    return path


def save_replay(label: str = None):
    """Snapshot the current event log into data/replays/ (atomic write).
    Returns the saved filename, or None if the caller's run was superseded.
    Raises OSError if the disk write fails."""
    with _lock:
        if is_stale():
            return None
        REPLAY_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe_label = re.sub(r"[^a-zA-Z0-9_-]", "-", str(label))[:40] if label else "run"
        filename = f"{stamp}-{safe_label}.jsonl"
        n = 1
        while (REPLAY_DIR / filename).exists():
            n += 1
            filename = f"{stamp}-{safe_label}-{n}.jsonl"
        content = LOG_PATH.read_text() if LOG_PATH.exists() else ""
        tmp = REPLAY_DIR / (filename + ".tmp")
        tmp.write_text(content)
        os.replace(tmp, REPLAY_DIR / filename)
        return filename


def list_replays() -> list[dict]:
    if not REPLAY_DIR.exists():
        return []
    out = []
    for path in sorted(REPLAY_DIR.glob("*.jsonl"), reverse=True):
        if not _REPLAY_NAME_RE.fullmatch(path.name):
            continue
        lines, skipped = read_jsonl_tolerant(path)
        summary = None
        ended = next((e for e in reversed(lines)
                      if e.get("type") in ("simulation_ended", "monopoly")), None)
        if ended:
            summary = ended.get("winner") or None
        out.append({"filename": path.name, "event_count": len(lines),
                    "summary": summary, "skipped_lines": skipped})
    return out


def read_replay(filename: str) -> list[dict]:
    """Read-only view of a saved run. Raises ValueError on an invalid name;
    returns [] if the file doesn't exist; skips corrupt lines."""
    path = replay_path(filename)
    if not path.exists():
        return []
    records, _ = read_jsonl_tolerant(path)
    return records


# --- leaderboard ----------------------------------------------------------------

def record_leaderboard_entry(winner: str, final_balances: dict, rounds_played: int,
                             reason: str, agent_stats: dict | None = None,
                             extra: dict | None = None):
    """
    Appends one line per finished run to data/leaderboard.jsonl -- unlike
    events.jsonl this file is NEVER reset. `winner` is the agent with the
    highest final balance (a *wealth* ranking). `agent_stats` carries the data
    needed for the other rankings (accuracy, profit, compute efficiency);
    entries written before it existed simply lack it.
    """
    global write_failures
    with _lock:
        if is_stale():
            return
        record = {
            "ts": time.time(), "run_id": current_run_id(), "winner": winner,
            "final_balances": final_balances, "rounds_played": rounds_played,
            "reason": reason,
        }
        if agent_stats is not None:
            record["agent_stats"] = agent_stats
        if extra:
            record.update(extra)
        try:
            LEADERBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(LEADERBOARD_PATH, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as e:
            write_failures += 1
            log.error("leaderboard write failed: %s", redact(e))


def read_leaderboard() -> list[dict]:
    with _lock:
        records, _ = read_jsonl_tolerant(LEADERBOARD_PATH)
    return records


def _read_jsonl_file(path: Path) -> list[dict]:  # kept for backward compatibility
    return read_jsonl_tolerant(path)[0]
