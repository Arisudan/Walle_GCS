"""Unit tests for the hardware-independent, safety-relevant logic.

These cover the pure functions that decode telemetry, authenticate users,
estimate distance, and make the depth-zone decision -- the parts most likely
to be touched during the refactor and most damaging if silently broken.

Tested against walle_gcs; the logic is identical in walle_gcs_max (the builds
differ only in model selection / config paths).
"""
import math
import unittest

import numpy as np

import _env  # noqa: F401  -- sets sys.path + offscreen Qt; must import first
import walle_gcs as g


class AuthTest(unittest.TestCase):
    def test_hash_is_deterministic_with_same_salt(self):
        salt, hashed = g._hash_password("hunter2")
        salt2, hashed2 = g._hash_password("hunter2", salt)
        self.assertEqual(salt, salt2)
        self.assertEqual(hashed, hashed2)

    def test_wrong_password_produces_different_hash(self):
        salt, hashed = g._hash_password("hunter2")
        _, wrong = g._hash_password("nope", salt)
        self.assertNotEqual(hashed, wrong)

    def test_fresh_salt_each_call(self):
        s1, _ = g._hash_password("pw")
        s2, _ = g._hash_password("pw")
        self.assertNotEqual(s1, s2)


class TelemetryTest(unittest.TestCase):
    """The MAVLink telemetry holder starts in a safe, zeroed state and
    exposes the field names the GUI renders."""

    def test_defaults_are_safe(self):
        tel = g.Telemetry()
        self.assertFalse(tel.armed)
        self.assertEqual(tel.link_quality, 0)
        self.assertEqual(tel.battery_voltage, 0.0)
        self.assertEqual(tel.flight_mode, "")

    def test_has_gui_consumed_fields(self):
        tel = g.Telemetry()
        for field in ("battery_voltage", "battery_current", "battery_percent",
                      "link_quality", "rssi_1", "snr", "pitch", "roll", "yaw",
                      "heading", "altitude", "ground_speed", "satellites",
                      "latitude", "longitude", "vertical_speed", "flight_mode",
                      "armed"):
            self.assertTrue(hasattr(tel, field), f"missing field {field!r}")


class DepthTest(unittest.TestCase):
    def test_normalize_to_0_255(self):
        raw = np.random.rand(60, 80).astype(np.float32) * 5.0
        norm = g._normalize_depth(raw)
        self.assertEqual(norm.shape, raw.shape)
        self.assertGreaterEqual(norm.min(), 0.0)
        self.assertLessEqual(norm.max(), 255.0001)

    def test_colorize_shape(self):
        norm = g._normalize_depth(np.random.rand(60, 80).astype(np.float32))
        col = g._colorize_depth(norm)
        self.assertEqual(col.shape, (60, 80, 3))

    def test_decision_stop_when_center_close(self):
        near = np.full((60, 80), 255.0, dtype=np.float32)  # everything close
        mode, _, _, _, _ = g._depth_decision(near)
        self.assertEqual(mode, "STOP")

    def test_decision_clear_when_far(self):
        far = np.zeros((60, 80), dtype=np.float32)  # everything far
        mode, _, _, _, _ = g._depth_decision(far)
        self.assertEqual(mode, "CLEAR")


class DistanceGeometryTest(unittest.TestCase):
    def test_no_altitude_correction_below_threshold(self):
        eng = g.ObjectDetectionEngine(conf_thresh=0.5)
        eng.depth_scale = 1.0
        out = eng._compute_distance(raw_depth=4.0, altitude=0.0,
                                    pitch_deg=-30.0, obj_cy_norm=0.5)
        self.assertAlmostEqual(out["los_m"], 4.0, places=3)
        self.assertAlmostEqual(out["scaled_los_m"], 4.0, places=3)
        self.assertAlmostEqual(out["ground_m"], 4.0 * math.cos(math.radians(-30)), places=3)
        self.assertAlmostEqual(out["vert_m"], 4.0 * abs(math.sin(math.radians(-30))), places=3)


if __name__ == "__main__":
    unittest.main()
