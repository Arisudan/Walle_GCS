"""Shared test setup: import-path + headless-Qt configuration.

Importing this module (do it first, before any walle_gcs import) makes the
GCS modules importable from the tests/ folder and forces Qt to the offscreen
platform so the UI can be built without a display server.

This mirrors what the ./walle-gcs launcher does at runtime: it runs from the
project root (rosetta/) so that slam_engine.py and the bare relative model
paths resolve.
"""
import os
import sys

# Headless Qt — must be set before PyQt5 is imported anywhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../walle_gcs
ROOT = os.path.dirname(HERE)                                        # .../rosetta

for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def qapp():
    """Return the singleton QApplication, creating it if needed."""
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication(sys.argv[:1])
