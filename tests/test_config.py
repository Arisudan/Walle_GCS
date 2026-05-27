"""Tests for the typed walle_max_config module.

These don't depend on Qt or CUDA — they exercise JSON layering, default
fallback, and validation. Run alongside the rest of the headless suite::

    python -m unittest discover tests
"""
import json
import tempfile
import unittest
from pathlib import Path

import _env  # noqa: F401  -- sets sys.path + offscreen Qt; must import first

from config.walle_max_config import (
    DepthZoneConfig,
    DetectionConfig,
    FaceRecConfig,
    WalleMaxConfig,
    load_config,
)


class DefaultsTest(unittest.TestCase):
    def test_defaults_are_valid(self):
        cfg = WalleMaxConfig()
        cfg.validate()                       # raises on failure
        # Sanity: defaults match the legacy hard-coded values
        self.assertEqual(cfg.depth_zone.stop_depth_rel, 180.0)
        self.assertEqual(cfg.depth_zone.slow_depth_rel, 140.0)
        self.assertAlmostEqual(cfg.depth_zone.center_zone_frac, 0.34)
        self.assertEqual(cfg.detection.depth_variant, "Base")
        self.assertEqual(cfg.face_rec.scales, (1.0, 2.0))


class ValidationTest(unittest.TestCase):
    def test_slow_must_be_less_than_stop(self):
        c = DepthZoneConfig(stop_depth_rel=100.0, slow_depth_rel=150.0)
        with self.assertRaises(ValueError):
            c.validate()

    def test_conf_thresh_out_of_range(self):
        c = DetectionConfig(conf_thresh=1.5)
        with self.assertRaises(ValueError):
            c.validate()

    def test_unknown_depth_variant_rejected(self):
        c = DetectionConfig(depth_variant="Huge")
        with self.assertRaises(ValueError):
            c.validate()

    def test_empty_face_scales_rejected(self):
        c = FaceRecConfig(scales=())
        with self.assertRaises(ValueError):
            c.validate()


class JSONLayeringTest(unittest.TestCase):
    def test_legacy_layout_loads(self):
        payload = {
            "alerts": {"low_batt_warn": 25, "low_batt_crit": 10},
            "video_streaming": {"ip": "10.0.0.5", "port": 6000},
        }
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "settings.json"
            p.write_text(json.dumps(payload))
            cfg = load_config(p)
        self.assertEqual(cfg.alerts.low_batt_warn, 25)
        self.assertEqual(cfg.alerts.low_batt_crit, 10)
        self.assertEqual(cfg.video.ip, "10.0.0.5")
        self.assertEqual(cfg.video.port, 6000)

    def test_unknown_keys_silently_ignored(self):
        payload = {"alerts": {"low_batt_warn": 50, "nonsense_key": 999}}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "settings.json"
            p.write_text(json.dumps(payload))
            cfg = load_config(p)
        self.assertEqual(cfg.alerts.low_batt_warn, 50)

    def test_bad_json_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "settings.json"
            p.write_text("{ this is not valid json")
            cfg = load_config(p)
        # Defaults still applied
        self.assertEqual(cfg.alerts.low_batt_warn, 30)

    def test_missing_file_uses_defaults(self):
        cfg = load_config(Path("/nonexistent/path/to/settings.json"))
        self.assertEqual(cfg.alerts.low_batt_warn, 30)


if __name__ == "__main__":
    unittest.main()
