"""Smoke tests: every GCS build must import and boot its full UI headless.

These are the regression net for the upcoming refactor (twin-file merge,
monolith decomposition). If a change breaks module loading, a tab, or a
feature toggle, one of these fails before it reaches a drone.

No AI models are loaded here — MainWindow construction does not call
load_models() (that happens on toggle), so this stays fast and offline.
"""
import unittest

import _env  # noqa: F401  -- sets sys.path + offscreen Qt; must import first

# Post twin-file merge: one module, two editions selected by --edition.
# EDITION is resolved from argv once at import, so a single process boots
# the default (base) edition. That exercises all shared UI; the only
# edition-specific code (the RF-DETR model class) loads on toggle, not at
# MainWindow construction, so it is out of smoke scope.
BUILDS = [
    ("walle_gcs", "2.0.0"),
]

EXPECTED_TABS = ["control_tab", "detection_tab", "logs_tab",
                 "settings_tab", "slam_tab", "recon_tab"]

EXPECTED_TOGGLES = ["_toggle_video", "_toggle_mav",
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
