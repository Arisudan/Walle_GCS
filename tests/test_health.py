"""Tests for the per-engine health + watchdog module.

These don't require Qt / CUDA — pure threading + time-based logic. Run
inside the headless suite::

    python -m unittest discover tests
"""
import threading
import time
import unittest

import _env  # noqa: F401  -- sets sys.path + offscreen Qt; must import first

from core.health import (
    EngineHealth,
    EngineStatus,
    EngineWatchdog,
    HealthRegistry,
)


class HealthBasicTest(unittest.TestCase):
    def test_initial_state(self):
        h = EngineHealth("X")
        snap = h.snapshot()
        self.assertEqual(snap.status, EngineStatus.UNINITIALIZED)
        self.assertEqual(snap.inferences_total, 0)
        self.assertEqual(snap.fps, 0.0)

    def test_record_inference_updates_fps_and_total(self):
        h = EngineHealth("X")
        h.set_status(EngineStatus.READY)
        for _ in range(10):
            h.record_inference(50.0)            # 50 ms -> 20 fps
        snap = h.snapshot()
        self.assertEqual(snap.inferences_total, 10)
        self.assertAlmostEqual(snap.mean_ms, 50.0, places=4)
        self.assertAlmostEqual(snap.fps, 20.0, places=3)

    def test_p95_latency(self):
        h = EngineHealth("X")
        h.set_status(EngineStatus.READY)
        latencies = [10.0] * 19 + [200.0]       # one slow frame in 20
        for ms in latencies:
            h.record_inference(ms)
        snap = h.snapshot()
        # 95th percentile across the window picks the slow frame
        self.assertGreaterEqual(snap.p95_ms, 199.0)

    def test_heartbeat_clears_stalled(self):
        h = EngineHealth("X", stall_threshold_s=0.05)
        h.set_status(EngineStatus.READY)
        h.heartbeat()
        time.sleep(0.1)
        self.assertTrue(h.check_stall())
        h.mark_stalled()
        h.heartbeat()                            # worker resumed
        self.assertEqual(h.snapshot().status, EngineStatus.READY)


class WatchdogTest(unittest.TestCase):
    def test_stall_triggers_restart_callback(self):
        reg = HealthRegistry()
        h = EngineHealth("X", stall_threshold_s=0.05)
        h.set_status(EngineStatus.READY)
        h.heartbeat()
        reg.register(h)

        called = threading.Event()
        h.attach_restart_callback(called.set)

        wd = EngineWatchdog(reg, poll_s=0.02)
        wd.start()
        try:
            self.assertTrue(called.wait(timeout=1.0))
            self.assertEqual(h.snapshot().status, EngineStatus.STALLED)
        finally:
            wd.stop()

    def test_loading_engine_is_not_flagged(self):
        reg = HealthRegistry()
        h = EngineHealth("X", stall_threshold_s=0.05)
        h.set_status(EngineStatus.LOADING)
        reg.register(h)

        called = threading.Event()
        h.attach_restart_callback(called.set)

        wd = EngineWatchdog(reg, poll_s=0.02)
        wd.start()
        try:
            self.assertFalse(called.wait(timeout=0.3))
            self.assertEqual(h.snapshot().status, EngineStatus.LOADING)
        finally:
            wd.stop()


if __name__ == "__main__":
    unittest.main()
