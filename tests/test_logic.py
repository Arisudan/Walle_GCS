"""Unit tests for the hardware-independent, safety-relevant logic.

These cover the pure functions that decode telemetry, authenticate users,
estimate distance, and make the depth-zone decision -- the parts most likely
to be touched during the refactor and most damaging if silently broken.

Tested against walle_gcs; the logic is identical in walle_gcs_max (the builds
differ only in model selection / config paths).
"""
import math
import struct
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


class CRSFTelemetryTest(unittest.TestCase):
    def test_battery_frame(self):
        tel = g.CRSFTelemetry()
        # voltage(2) current(2) capacity(3) remaining(1), big-endian, deci-units
        payload = struct.pack(">H", 168) + struct.pack(">H", 50) + b"\x00\x00\x00" + bytes([75])
        tel.parse_frame(g.CRSF_FRAMETYPE_BATTERY, payload)
        self.assertAlmostEqual(tel.battery_voltage, 16.8, places=3)
        self.assertAlmostEqual(tel.battery_current, 5.0, places=3)

    def test_gps_frame(self):
        tel = g.CRSFTelemetry()
        payload = (struct.pack(">i", 377749000)    # lat * 1e7
                   + struct.pack(">i", -1224194000)  # lon * 1e7
                   + struct.pack(">H", 360)          # ground speed dm/s -> 10 m/s
                   + struct.pack(">H", 18000)        # heading cdeg -> 180.0
                   + struct.pack(">H", 1100)         # altitude, offset +1000 -> 100 m
                   + bytes([9]))                     # satellites
        tel.parse_frame(g.CRSF_FRAMETYPE_GPS, payload)
        self.assertAlmostEqual(tel.latitude, 37.7749, places=4)
        self.assertAlmostEqual(tel.longitude, -122.4194, places=4)
        self.assertAlmostEqual(tel.ground_speed, 10.0, places=3)
        self.assertAlmostEqual(tel.heading, 180.0, places=3)
        self.assertEqual(tel.altitude, 100)
        self.assertEqual(tel.satellites, 9)

    def test_short_payload_is_ignored(self):
        tel = g.CRSFTelemetry()
        tel.parse_frame(g.CRSF_FRAMETYPE_BATTERY, b"\x01\x02")  # too short
        self.assertEqual(tel.battery_voltage, 0.0)  # untouched default


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


class JoystickCalibrationTest(unittest.TestCase):
    def test_passthrough_when_uncalibrated(self):
        cal = g.JoystickCalibration()
        self.assertEqual(cal.apply(0, 0.42), 0.42)

    def test_symmetric_mapping_when_calibrated(self):
        cal = g.JoystickCalibration(num_axes=4)
        cal.mins = [-1.0] * 4
        cal.centers = [0.0] * 4
        cal.maxs = [1.0] * 4
        cal.calibrated = True
        self.assertAlmostEqual(cal.apply(0, 0.5), 0.5, places=6)
        self.assertAlmostEqual(cal.apply(0, -0.5), -0.5, places=6)
        self.assertAlmostEqual(cal.apply(0, 0.0), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
