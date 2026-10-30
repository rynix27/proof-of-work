"""
Simulation lifecycle: the single place that starts, replaces and stops the
background simulation thread.

Guarantees
  * start() is serialized by a lock, so concurrent/rapid resets cannot
    interleave.
  * The new generation is allocated BEFORE anything else, which immediately
    invalidates the previous thread; the previous thread is then joined with a
    BOUNDED timeout (never indefinitely -- it may be inside a slow API call).
  * The event log is reset only after that, in the caller's thread, so the
    dashboard sees an empty, consistent state the moment start() returns.
  * If the old thread is still alive after the timeout it is harmless: its run
    id is stale, so core.events drops every event/replay/leaderboard write it
    attempts, and it owns its own Ledger/agents (no shared mutable state).
  * Pause/resume/speed never spawn threads, so they cannot create duplicates.
"""
import logging
import threading

from core import events
from core.control import control
from core.injected_tasks import injected_tasks
from core.simconfig import SimulationConfig

log = logging.getLogger("proof_of_work.runner")


class SimulationManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self, overrides: dict | None = None, join_timeout: float = 2.0) -> dict:
        """Start a fresh run, replacing any current one. Raises
        core.simconfig.ConfigError (before touching any state) on bad config."""
        from core.orchestrator import run_simulation  # late import: avoids a cycle

        cfg = SimulationConfig.from_overrides(overrides)
        with self._lock:
            old = self._thread
            generation = control.start_new_generation()
            if old is not None and old.is_alive():
                old.join(timeout=join_timeout)
                if old.is_alive():
                    log.warning("previous simulation still finishing an in-flight call; "
                                "its output is discarded")
            injected_tasks.clear()
            run_id = events.reset_log()
            thread = threading.Thread(
                target=run_simulation, name=f"simulation-{generation}", daemon=True,
                kwargs={"verbose": True, "cfg": cfg, "generation": generation, "run_id": run_id},
            )
            self._thread = thread
            thread.start()
            return {"run_id": run_id, "generation": generation}

    def is_running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def shutdown(self, timeout: float = 5.0) -> bool:
        """Ask the current run to stop and wait (bounded). True if it stopped."""
        with self._lock:
            control.request_stop()
            t = self._thread
            if t is not None and t.is_alive():
                t.join(timeout=timeout)
            return not (t and t.is_alive())


manager = SimulationManager()
