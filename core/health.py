"""Per-engine health metrics and stall detection.

Each AI engine (detection / depth / face-rec / human-tracker) runs its
inference in a background worker thread. When one of those threads dies
silently — common failure modes: a CUDA OOM that the kernel never throws,
a model-input shape mismatch, a downstream tensor-copy hang — the engine
simply stops producing results. The UI keeps showing "ON" because
``is_ready`` is still True, but no boxes update. From the ground crew's
perspective the drone has gone blind without warning.

This module gives the GCS a uniform way to:

  * Stamp a heartbeat from every worker loop iteration.
  * Record per-inference latency so the status bar can show real FPS
    and a p95 instead of a static "Loaded" badge.
  * Detect a stalled worker (no heartbeat for N seconds) and surface a
    structured event the UI / watchdog can hook into.

Auto-restart is left to callers: each engine has different shutdown
semantics, and a blind ``thread.start()`` could re-trigger the same
crash. ``EngineHealth.attach_restart_callback`` lets the watchdog ask
the engine to recover itself.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

from core.logging import get_logger


log = get_logger("Health")


class EngineStatus(str, Enum):
    UNINITIALIZED = "uninitialized"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"       # producing results but slow / dropping
    STALLED = "stalled"         # no heartbeat past threshold
    FAILED = "failed"           # caller-marked terminal failure


@dataclass
class HealthSnapshot:
    """A point-in-time view of an engine. Cheap to construct, safe to
    hand to the UI thread."""
    name: str
    status: EngineStatus
    fps: float
    mean_ms: float
    p95_ms: float
    last_heartbeat_age_s: float
    inferences_total: int


class EngineHealth:
    """Tracks heartbeat + latency for a single engine.

    Thread-safe: ``record_inference`` and ``heartbeat`` are called from
    the worker thread; ``snapshot`` is read from the GUI thread.
    """

    def __init__(self, name: str, window: int = 60,
                 stall_threshold_s: float = 5.0):
        self.name = name
        self._lock = threading.Lock()
        self._latencies_ms: deque = deque(maxlen=window)
        self._status: EngineStatus = EngineStatus.UNINITIALIZED
        self._last_heartbeat: float = 0.0
        self._inferences_total: int = 0
        self._stall_threshold_s = stall_threshold_s
        self._restart_cb: Optional[Callable[[], None]] = None
        self._stall_acked: bool = False

    # ── Worker-side API ───────────────────────────────────────────
    def set_status(self, status: EngineStatus) -> None:
        with self._lock:
            old = self._status
            self._status = status
            if old != status:
                log.info("%s: status %s -> %s", self.name, old.value, status.value)
                if status == EngineStatus.READY:
                    self._stall_acked = False

    def heartbeat(self) -> None:
        with self._lock:
            self._last_heartbeat = time.monotonic()
            if self._status == EngineStatus.STALLED:
                # The worker came back to life on its own — clear the flag
                # so the watchdog stops nagging.
                self._status = EngineStatus.READY
                self._stall_acked = False
                log.info("%s: recovered from STALLED", self.name)

    def record_inference(self, ms: float) -> None:
        with self._lock:
            self._latencies_ms.append(ms)
            self._inferences_total += 1
            self._last_heartbeat = time.monotonic()

    # ── Watchdog / UI side ───────────────────────────────────────
    def attach_restart_callback(self, cb: Callable[[], None]) -> None:
        """Provide a recovery hook the watchdog can call on stall.

        The callback runs on the watchdog thread. It should be quick —
        either kick off a recovery thread itself or schedule the actual
        recovery via QTimer.singleShot to land it on the GUI thread.
        """
        with self._lock:
            self._restart_cb = cb

    def snapshot(self) -> HealthSnapshot:
        with self._lock:
            latencies = list(self._latencies_ms)
            now = time.monotonic()
            age = max(0.0, now - self._last_heartbeat) if self._last_heartbeat else float("inf")
            status = self._status
            total = self._inferences_total

        if latencies:
            mean = sum(latencies) / len(latencies)
            # Pick the worst 5% — for a 20-sample window the last 1 sample.
            # ceil((1-0.95) * n) = number of items in the top 5%; we want
            # the smallest of those, which is at index n - that_count.
            n = len(latencies)
            tail = max(1, int(round(0.05 * n)))
            p95 = sorted(latencies)[n - tail]
            fps = 1000.0 / mean if mean > 0 else 0.0
        else:
            mean = 0.0
            p95 = 0.0
            fps = 0.0
        return HealthSnapshot(
            name=self.name,
            status=status,
            fps=fps,
            mean_ms=mean,
            p95_ms=p95,
            last_heartbeat_age_s=age if age != float("inf") else -1.0,
            inferences_total=total,
        )

    def check_stall(self, now: Optional[float] = None) -> bool:
        """Return True iff the engine should be considered stalled
        right now. Idempotent — calling repeatedly is cheap."""
        with self._lock:
            if self._status not in (EngineStatus.READY, EngineStatus.DEGRADED):
                return False
            if self._last_heartbeat == 0.0:
                return False
            t = now if now is not None else time.monotonic()
            return (t - self._last_heartbeat) >= self._stall_threshold_s

    def mark_stalled(self) -> bool:
        """Transition to STALLED if not already. Returns True the first
        time it transitions (used by the watchdog to fire exactly one
        restart attempt per stall episode)."""
        with self._lock:
            if self._status == EngineStatus.STALLED:
                return False
            self._status = EngineStatus.STALLED
            first = not self._stall_acked
            self._stall_acked = True
            return first

    def get_restart_callback(self) -> Optional[Callable[[], None]]:
        with self._lock:
            return self._restart_cb


class HealthRegistry:
    """Container that owns every :class:`EngineHealth` instance in the
    process. The UI status panel polls this; the watchdog walks it."""

    def __init__(self):
        self._lock = threading.Lock()
        self._engines: Dict[str, EngineHealth] = {}

    def register(self, health: EngineHealth) -> None:
        with self._lock:
            self._engines[health.name] = health

    def get(self, name: str) -> Optional[EngineHealth]:
        with self._lock:
            return self._engines.get(name)

    def snapshot_all(self) -> List[HealthSnapshot]:
        with self._lock:
            engines = list(self._engines.values())
        return [e.snapshot() for e in engines]


# Process-wide singleton — same pattern as core.logging.
_REGISTRY = HealthRegistry()


def get_registry() -> HealthRegistry:
    return _REGISTRY


class EngineWatchdog:
    """Background thread that polls every registered engine and:

      * Marks any engine without a heartbeat for ``stall_threshold_s``
        as STALLED (logged).
      * Invokes the engine's restart callback once per stall episode.

    Start with :meth:`start` after engines are registered. Stop on
    application shutdown with :meth:`stop`.
    """

    def __init__(self, registry: HealthRegistry, poll_s: float = 2.0):
        self._registry = registry
        self._poll_s = poll_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="EngineWatchdog", daemon=True)
        self._thread.start()
        log.info("watchdog started (poll=%.1fs)", self._poll_s)

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            for snap in self._registry.snapshot_all():
                eng = self._registry.get(snap.name)
                if eng is None:
                    continue
                if eng.check_stall():
                    fresh = eng.mark_stalled()
                    if fresh:
                        log.warning(
                            "%s: no heartbeat for >=%.1fs — marking STALLED",
                            snap.name, eng._stall_threshold_s)
                        cb = eng.get_restart_callback()
                        if cb is not None:
                            try:
                                cb()
                            except Exception as e:
                                log.error("%s: restart callback raised: %s",
                                          snap.name, e)
            self._stop.wait(self._poll_s)
