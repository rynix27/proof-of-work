"""
Thread-safe queue for the "ask the economy" feature: a viewer types a
question on the dashboard, it lands here, and the next round pulls it
instead of generating a random task from the bank.

A single process-wide instance (`injected_tasks`) is imported by both
server.py (to add) and core/orchestrator.py (to pop).
"""
import threading
from collections import deque


class InjectedTaskQueue:
    def __init__(self, max_pending: int = 10):
        self._queue = deque(maxlen=max_pending)
        self._lock = threading.Lock()

    def push(self, task):
        with self._lock:
            self._queue.append(task)

    def try_push(self, task) -> bool:
        """Like push, but refuses (returns False) when full instead of silently
        dropping the oldest queued question."""
        with self._lock:
            if len(self._queue) >= self._queue.maxlen:
                return False
            self._queue.append(task)
            return True

    def clear(self):
        with self._lock:
            self._queue.clear()

    def pop_next(self):
        with self._lock:
            return self._queue.popleft() if self._queue else None

    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)


def _default_max_pending() -> int:
    from core import config
    return config.INJECT_MAX_PENDING


injected_tasks = InjectedTaskQueue(max_pending=_default_max_pending())
