"""
Shared control state between the simulation thread and the Flask control
endpoints. A single process-wide instance (`control`) is imported by both
core/orchestrator.py and server.py.

Generation tokens: every (re)start allocates a new generation number under a
lock. A simulation thread owns exactly one generation and exits at its next
checkpoint once the global generation moves on, so at most one thread is ever
"current". See core/runner.py for the start/stop/shutdown lifecycle, and
core/events.py for how a superseded thread is prevented from writing.
"""
import threading
import time

from core import config


class SimulationControl:
    def __init__(self):
        self._lock = threading.Lock()
        self._pause_event = threading.Event()  # set = paused
        self._stop_event = threading.Event()   # set = stop
        self.round_delay = config.ROUND_INTERVAL_SECONDS
        self.generation = 0

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def set_speed(self, seconds: float):
        try:
            seconds = float(seconds)
        except (TypeError, ValueError):
            raise ValueError("seconds must be a number")
        if seconds != seconds:  # NaN
            raise ValueError("seconds must be a finite number")
        self.round_delay = max(0.1, min(30.0, seconds))

    def request_stop(self):
        self._stop_event.set()

    def should_stop(self, my_generation: int) -> bool:
        return self._stop_event.is_set() or my_generation != self.generation

    def start_new_generation(self) -> int:
        with self._lock:
            self._stop_event.clear()
            self._pause_event.clear()
            self.generation += 1
            return self.generation

    def wait_while_paused(self, my_generation: int):
        while self.is_paused() and not self.should_stop(my_generation):
            time.sleep(0.1)


control = SimulationControl()
