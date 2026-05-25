"""Smoke tests: every GCS build must import and boot its full UI headless.

These are the regression net for the upcoming refactor (twin-file merge,
monolith decomposition). If a change breaks module loading, a tab, or a
feature toggle, one of these fails before it reaches a drone.

No AI models are loaded here — MainWindow construction does not call
load_models() (that happens on toggle), so this stays fast and offline.
"""
import unittest

import _env  # noqa: F401  -- sets sys.path + offscreen Qt; must import first

# The shipping builds and their expected version strings. After the twin-file
# merge (Phase 1) this collapses to one module exercised with/without --max;
# update this table at that point.
BUILDS = [
    ("walle_gcs", "2.0.0"),
    ("walle_gcs_max", "2.0.0-MAX"),
]

EXPECTED_TABS = ["control_tab", "detection_tab", "logs_tab",
                 "settings_tab", "slam_tab", "recon_tab"]

EXPECTED_TOGGLES = ["_toggle_video", "_toggle_serial", "_toggle_esp32",
                    "_toggle_detection", "_toggle_depth_overlay",
                    "_toggle_face_rec", "_toggle_track_human"]


class SmokeTest(unittest.TestCase):
    def test_imports_and_version(self):
        import importlib
        for mod_name, version in BUILDS:
            with self.subTest(build=mod_name):
                mod = importlib.import_module(mod_name)
                self.assertEqual(mod.APP_VERSION, version)

    def test_ui_boots_with_all_tabs_and_toggles(self):
        import importlib
        app = _env.qapp()
        for mod_name, _ in BUILDS:
            with self.subTest(build=mod_name):
                mod = importlib.import_module(mod_name)
                win = mod.MainWindow(username="admin", role="admin")
                try:
                    for _ in range(5):
                        app.processEvents()
                    for tab in EXPECTED_TABS:
                        self.assertTrue(hasattr(win, tab),
                                        f"{mod_name}: missing tab '{tab}'")
                    for tog in EXPECTED_TOGGLES:
                        self.assertTrue(callable(getattr(win, tog, None)),
                                        f"{mod_name}: missing toggle '{tog}'")
                finally:
                    win.close()


if __name__ == "__main__":
    unittest.main()
