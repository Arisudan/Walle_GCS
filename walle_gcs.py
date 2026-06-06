
#!/usr/bin/env python3
"""
Walle Drone Ground Control Station.

Single GCS application with two editions, selected at launch via
``--edition base|max`` (default: base). The edition picks the AI model
sizes and the config/credential files; everything else is shared:

  base : RF-DETR Base   + Depth-Anything-V2 Small  (walle_settings.json,
         walle_users.json)
  max  : RF-DETR Medium + Depth-Anything-V2 Base   (walle_settings_max.json,
         walle_users_max.json) — bigger models, higher accuracy.

The two editions use separate config files so MAX settings never
overwrite the regular build's. See the EDITIONS table below.
"""

import sys
import struct
import time
import math
import threading
import json
import csv
import os
import hashlib

# Force Qt through XCB (XWayland) instead of native Wayland — COSMIC's
# Wayland compositor crashes on Pop!_OS 24.04 under sustained NVIDIA driver
# load (heavy ONNX/CUDA inference). XCB is the rock-solid X11 path.
# Set BEFORE any PyQt5 import. ``setdefault`` so an explicit env var still wins.
# Linux only — macOS has no XCB plugin and Windows uses its own ``windows`` QPA.
if sys.platform.startswith("linux"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
from datetime import datetime, timedelta
from collections import deque

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from rfdetr import RFDETRBase, RFDETRMedium


# ─── PyTorch CPU/GPU configuration ──────────────────────────────────
# By default PyTorch grabs one intra-op worker per physical core and
# pegs them all even when the heavy work is queued on the GPU — that
# alone pushed walle_gcs to ~99% CPU on multi-core hosts. Two threads
# are enough to stage frames into the GPU and run the cv2 preproc;
# anything more just oversubscribes. `WALLE_TORCH_THREADS` overrides.
try:
    _n_threads = int(os.environ.get("WALLE_TORCH_THREADS", "0")) or 2
    torch.set_num_threads(max(1, _n_threads))
except (RuntimeError, ValueError):
    pass
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    # set_num_interop_threads must run before any parallel work; calling
    # it after the pool has spun up raises. Safe to ignore — the cap is
    # a one-time tuning knob, not a correctness requirement.
    pass


def _select_device():
    """Pick the best inference device available on this host.

    Order: CUDA (NVIDIA, Windows + Linux) → MPS (Apple Silicon) → CPU.
    Returns (torch.device, model_dtype). fp16 is only safe on CUDA/MPS;
    on CPU we stay in fp32 (the HF DPT head misbehaves in fp16 on CPU).
    """
    if torch.cuda.is_available():
        # cuDNN benchmark + TF32 are the two biggest single-line wins
        # for a CNN/ViT inference workload with mostly-static shapes.
        torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except AttributeError:
            pass
        return torch.device("cuda"), torch.float16
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32

# This script lives in walle_gcs/. Its shared sibling module ``slam_engine``
# (and the model/data files it expects) stay in the rosetta project root, so
# put the project root on sys.path for that import to resolve.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from config.walle_max_config import load_config, get_config
from core.logging import configure_logging, get_logger
from core.health import (
    EngineHealth, EngineStatus, EngineWatchdog, get_registry as get_health_registry,
)

# ─── Edition selection ──────────────────────────────────────────────
# One application, two editions. ``--edition`` is parsed here at import
# time (not in main()) because WALLE_CONFIG below loads at module level,
# before main() runs. parse_known_args ignores Qt's own argv flags.
import argparse
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path


@_dataclass(frozen=True)
class Edition:
    name: str
    detect_model: type      # RF-DETR class to instantiate
    depth_variant: str      # Depth-Anything variant key (see DEPTH_MODELS)
    settings_file: str
    users_file: str
    version: str
    title_tag: str          # suffix shown in window title / login


EDITIONS = {
    "base": Edition("base", RFDETRBase,   "Small",
                    "walle_settings.json",     "walle_users.json",
                    "2.0.0",     ""),
    "max":  Edition("max",  RFDETRMedium, "Base",
                    "walle_settings_max.json", "walle_users_max.json",
                    "2.0.0-MAX", "  [MAX]"),
}


def _resolve_edition() -> Edition:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--edition", choices=tuple(EDITIONS), default="base")
    args, _ = p.parse_known_args()
    return EDITIONS[args.edition]


EDITION = _resolve_edition()
_HERE = _Path(__file__).resolve().parent

# Load typed config FIRST so subsequent module-level constants
# (STOP_DEPTH_REL etc.) see JSON overrides. Bad config aborts startup —
# loud failure beats silently flying on stale thresholds. The edition
# picks which settings file to layer over the typed defaults.
WALLE_CONFIG = load_config(path=_HERE / EDITION.settings_file)
# The settings file may not carry depth_variant; the edition is the
# source of truth for which model sizes this build runs.
WALLE_CONFIG.detection.depth_variant = EDITION.depth_variant

from face_recognition_engine import FaceRecognitionEngine
from human_tracker_engine import HumanTrackerEngine
from slam_engine import SlamEngine, SlamTab
from reconstruction_engine import ReconstructionTab

log_det = get_logger("Detection")
log_cam = get_logger("CameraThread")
log_mav = get_logger("MAVLink")
log_main = get_logger("MainWindow")
log_settings = get_logger("Settings")
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QGroupBox, QGridLayout, QFrame,
    QProgressBar, QSplitter, QStatusBar, QCheckBox, QSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QLineEdit, QSlider, QFormLayout, QMessageBox, QFileDialog,
    QListWidget, QListWidgetItem, QScrollArea, QSizePolicy,
    QAbstractItemView, QDateEdit, QDialog, QStackedWidget,
    QToolButton, QButtonGroup, QGraphicsDropShadowEffect,
    QDoubleSpinBox
)
from PyQt5.QtCore import (
    Qt, QTimer, pyqtSignal, QThread, QObject, QSize, QDate, QPropertyAnimation,
    QEasingCurve, QPoint
)
from PyQt5.QtGui import (
    QImage, QPixmap, QPainter, QPen, QColor, QFont, QBrush, QIcon,
    QLinearGradient, QPainterPath, QPolygonF, QFontMetrics
)
from PyQt5.QtCore import QPointF


# ─── MAVLink link — see protocol/mavlink.py ─────────────────────────
# The vehicle link is plain MAVLink over UDP (QGroundControl-style).
# Telemetry parsing and command sending live in protocol/mavlink.py so
# this file isn't responsible for the wire protocol.
from protocol.mavlink import MAVLinkConnection, Telemetry

# ─── Application Version ────────────────────────────────────────────
APP_VERSION = EDITION.version

# Qt 5.14+ can wrap a BGR byte buffer directly (no per-frame BGR->RGB
# conversion in the video hot path). Fall back gracefully on older Qt.
_HAS_BGR888 = hasattr(QImage, "Format_BGR888")

# ─── Dark Theme Stylesheet ──────────────────────────────────────────
DARK_STYLESHEET = """
QMainWindow { background-color: #0a0a0a; }
QWidget { background-color: transparent; color: #cccccc; font-family: 'Segoe UI', 'Roboto', sans-serif; }

QLabel { color: #b0b0b0; }

QGroupBox {
    color: #00cc00;
    border: 1px solid #1e1e1e;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    font-size: 11px;
    background-color: #0f0f0f;
}
QGroupBox::title { subcontrol-position: top left; padding: 2px 12px; }

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QDateEdit {
    background-color: #141414;
    color: #cccccc;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 5px 10px;
    font-size: 12px;
    min-height: 26px;
}
QComboBox:hover, QLineEdit:hover { border-color: #444; }
QComboBox:focus, QLineEdit:focus { border-color: #00aa00; }
QComboBox::drop-down { border: none; padding-right: 8px; }
QComboBox QAbstractItemView {
    background-color: #141414; color: #ccc; selection-background-color: #002200;
    border: 1px solid #2a2a2a; border-radius: 4px;
}

QPushButton {
    background-color: #141414;
    color: #cccccc;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 12px;
    min-height: 26px;
}
QPushButton:hover { background-color: #1c1c1c; border-color: #444; }
QPushButton:pressed { background-color: #252525; }
QPushButton:checked { background-color: #002200; border-color: #00aa00; color: #00ff00; }

QTabWidget::pane {
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    background-color: #0d0d0d;
    top: -1px;
}
QTabBar::tab {
    background-color: #111;
    color: #666;
    border: 1px solid #1e1e1e;
    border-bottom: none;
    padding: 10px 20px;
    margin-right: 2px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-size: 11px;
    font-weight: bold;
}
QTabBar::tab:selected {
    background-color: #0d0d0d;
    color: #00cc00;
    border-bottom: 2px solid #00cc00;
}
QTabBar::tab:hover { color: #999; background-color: #151515; }

QTableWidget {
    background-color: #0d0d0d;
    alternate-background-color: #111;
    gridline-color: #1a1a1a;
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    font-size: 11px;
    selection-background-color: #002200;
}
QTableWidget::item { padding: 4px 8px; border: none; }
QHeaderView::section {
    background-color: #141414;
    color: #666;
    border: none;
    border-bottom: 1px solid #2a2a2a;
    padding: 8px 8px;
    font-weight: bold;
    font-size: 10px;
    text-transform: uppercase;
}

QScrollBar:vertical {
    background: #0a0a0a; width: 6px; margin: 0; border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #2a2a2a; border-radius: 3px; min-height: 30px;
}
QScrollBar::handle:vertical:hover { background: #444; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #0a0a0a; height: 6px; margin: 0; border-radius: 3px;
}
QScrollBar::handle:horizontal {
    background: #2a2a2a; border-radius: 3px; min-width: 30px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

QSlider::groove:horizontal {
    background: #1a1a1a; height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #00cc00; width: 14px; height: 14px;
    margin: -5px 0; border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #005500; border-radius: 2px; }

QProgressBar {
    background-color: #141414; border: 1px solid #2a2a2a;
    border-radius: 4px; text-align: center; color: #aaa;
    font-size: 10px; min-height: 16px;
}
QProgressBar::chunk { background-color: #00aa00; border-radius: 3px; }

QStatusBar { background-color: #080808; color: #555; font-size: 11px; border-top: 1px solid #1a1a1a; }
QStatusBar QLabel { color: #555; font-size: 11px; }

QScrollArea { border: none; background: transparent; }

QCheckBox { color: #aaa; spacing: 6px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 3px; border: 1px solid #333; background: #141414; }
QCheckBox::indicator:checked { background: #00aa00; border-color: #00cc00; }
"""


# ─── Sidebar Stylesheet ─────────────────────────────────────────────
SIDEBAR_STYLE = """
QFrame#sidebar {
    background-color: #0d0d0d;
    border-right: 1px solid #1a1a1a;
}
"""

NAV_BTN_STYLE = """
QToolButton {
    background: transparent;
    border: none;
    border-radius: 8px;
    padding: 6px 4px;
    margin: 2px 4px;
    color: #555;
    font-size: 10px;
    font-weight: bold;
}
QToolButton:hover {
    background-color: #151515;
    color: #aaa;
}
QToolButton:checked {
    background-color: #001a00;
    color: #00cc00;
    border-left: 3px solid #00cc00;
}
"""




# ─── Object Detection + Depth Engine ────────────────────────────────

DEPTH_MODELS = {
    "Small": "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "Base":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
}

COCO_NAMES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow",
    22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe", 27: "backpack",
    28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase", 34: "frisbee",
    35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard",
    42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass",
    47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli",
    57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
    67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
    75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave",
    79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator", 84: "book",
    85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear",
    89: "hair drier", 90: "toothbrush",
}

DANGER_CLASSES = {"person", "stairs", "obstacle", "car", "truck"}

DETECT_COLORS = {
    "person": (0, 220, 0), "chair": (255, 165, 0), "table": (255, 200, 0),
    "door": (0, 220, 220), "car": (255, 100, 100), "truck": (255, 50, 50),
    "bicycle": (100, 255, 100), "dog": (200, 100, 255), "cat": (255, 100, 200),
}
DETECT_DEFAULT_COLOR = (255, 255, 255)

# ─── Depth Avoidance (from drone_depth_avoidance.py) ────────────────
#
# Thresholds are pulled from the typed config at startup
# (config/walle_max_config.py :: DepthZoneConfig). These module-level
# names remain so the rest of the file can keep referencing
# STOP_DEPTH_REL / SLOW_DEPTH_REL / CENTER_ZONE_FRAC without each call
# site having to fetch the config.

_zone_cfg = get_config().depth_zone
STOP_DEPTH_REL = _zone_cfg.stop_depth_rel    # normalized 0..255; above -> emergency
SLOW_DEPTH_REL = _zone_cfg.slow_depth_rel    # above -> side-step
CENTER_ZONE_FRAC = _zone_cfg.center_zone_frac  # fraction of width that is "forward"


def _normalize_depth(depth_raw: np.ndarray) -> np.ndarray:
    """Normalize raw depth map to 0-255 range (larger = closer)."""
    d = depth_raw.astype(np.float32)
    d = (d - d.min()) / (d.max() - d.min() + 1e-8) * 255.0
    return d


# Precomputed Spectral_r LUT: built once at first use to avoid per-frame
# matplotlib colormap lookups. cv2.applyColorMap with this LUT is ~10x faster.
_SPECTRAL_R_LUT = None


def _spectral_r_lut() -> np.ndarray:
    global _SPECTRAL_R_LUT
    if _SPECTRAL_R_LUT is None:
        cmap = matplotlib.colormaps.get_cmap("Spectral_r")
        rgb = np.array([cmap(i)[:3] for i in range(256)]) * 255.0
        _SPECTRAL_R_LUT = rgb[:, ::-1].astype(np.uint8).reshape(256, 1, 3)
    return _SPECTRAL_R_LUT


def _colorize_depth(depth_norm: np.ndarray) -> np.ndarray:
    """Apply Spectral_r colormap to a 0-255 depth map -> BGR image."""
    d8 = depth_norm.astype(np.uint8)
    return cv2.applyColorMap(d8, _spectral_r_lut())


def _zone_closeness(depth_norm: np.ndarray, zone_frac: float = CENTER_ZONE_FRAC):
    """Return (left, center, right) closeness scores (95th percentile)."""
    h, w = depth_norm.shape
    cw = int(w * zone_frac)
    cx = w // 2
    left   = depth_norm[:, : cx - cw // 2]
    center = depth_norm[:, cx - cw // 2 : cx + cw // 2]
    right  = depth_norm[:, cx + cw // 2 :]
    pct = lambda a: float(np.percentile(a, 95)) if a.size else 0.0
    return pct(left), pct(center), pct(right)


def _depth_decision(depth_norm: np.ndarray):
    """Return (mode, reason, left, center, right) avoidance decision."""
    l, c, r = _zone_closeness(depth_norm)
    if c >= STOP_DEPTH_REL:
        return "STOP", f"center={c:.0f}", l, c, r
    if c >= SLOW_DEPTH_REL:
        if l < r:
            return "AVOID_L", f"center={c:.0f}, left clearer", l, c, r
        return "AVOID_R", f"center={c:.0f}, right clearer", l, c, r
    return "CLEAR", f"L/C/R={l:.0f}/{c:.0f}/{r:.0f}", l, c, r


class ObjectDetectionEngine(QObject):
    """
    Runs RF-DETR object detection + Depth Anything V2 monocular depth in
    a background thread. Call process_frame() from the main thread —
    returns annotated frame + detection list.
    """

    models_loaded = pyqtSignal(bool)

    def __init__(self, conf_thresh=0.5, depth_variant="Base", camera_pitch=-30.0):
        super().__init__()
        self.conf_thresh = conf_thresh
        self.camera_pitch = camera_pitch
        self.depth_scale = 1.0
        self._loaded = False
        self._loading = False
        self._lock = threading.Lock()

        # Health tracking — registered with the global registry so the
        # status panel and watchdog see both workers.
        self._depth_health = EngineHealth("Depth")
        self._detect_health = EngineHealth("Detection")
        _hreg = get_health_registry()
        _hreg.register(self._depth_health)
        _hreg.register(self._detect_health)
        self._det_model = None
        self._depth_processor = None
        self._depth_model = None
        self._device = None
        self._depth_map = None
        # Bumped each time the depth worker stores a new map, so the display
        # can colorize/stack the depth panel only when it actually changes
        # (the map updates at depth_target_fps, far below display FPS).
        self._depth_version = 0
        self._depth_ms = 0.0
        self._det_ms = 0.0
        self._dist_history = {}

        # Async depth
        self._depth_frame = None
        self._depth_pending = threading.Event()
        self._depth_stop = threading.Event()
        self._depth_thread = None

        # Async detection (non-blocking process_frame)
        self._det_input_frame = None
        self._det_altitude = 0.0
        self._det_pitch = camera_pitch
        self._det_pending = threading.Event()
        self._det_stop = threading.Event()
        self._det_thread = None
        self._latest_detections = []

        # Worker cadence caps (see DetectionConfig.target_fps). The display
        # still draws every camera frame from the cached result; these only
        # limit how often the heavy models actually run — the main CPU lever.
        _dcfg = get_config().detection
        self._det_min_interval = (1.0 / _dcfg.target_fps
                                  if _dcfg.target_fps > 0 else 0.0)
        self._depth_min_interval = (1.0 / _dcfg.depth_target_fps
                                    if _dcfg.depth_target_fps > 0 else 0.0)
        self._last_det_proc = 0.0
        self._last_depth_proc = 0.0

        # ── Lightweight IoU tracker ──
        # Assigns a stable id per physical object so the detection table can
        # log each object once (with a "times seen" count) instead of adding a
        # new row every frame. Touched only by the detection worker thread.
        self._tracks = {}            # id -> {class, x1,y1,x2,y2, last_seen, hits}
        self._next_track_id = 1
        self._track_iou_thresh = 0.3
        self._track_ttl = 2.0        # seconds unseen before a track is dropped

        # ── DeepSort re-ID tracker (preferred over the IoU tracker) ──
        # Adds a MobileNet appearance embedding + Kalman motion so an object
        # keeps its id through short occlusions and re-entry at a different
        # position. Created in load_models(); stays None if the package is
        # missing, in which case the IoU tracker above is used as fallback.
        self._deepsort = None
        self._ds_max_age = 120       # detection-frames a track survives unseen
        self._ds_n_init = 2          # frames before a track is confirmed/logged
        self._track_hits = {}        # track id -> times seen (the "Count" column)

        self.depth_variant = depth_variant

        # ── Overlay tuning (less-cluttered rendering) ──
        # Cap how many boxes get drawn per frame. None = no cap (depth-only).
        self.max_drawn = None
        # IoU threshold for suppressing duplicate same-class boxes.
        # Set >= 1.0 to disable — keep every same-class instance visible.
        self.iou_dedupe_thresh = 1.0
        # Hide detections farther than this (meters, from Depth Anything V2).
        # None disables the cutoff. Detections with no depth info are kept.
        self.max_distance_m = 6.0
        # Also drop boxes that fall in the far end of THIS frame's relative
        # depth: 0.0 keeps everything, 1.0 keeps only the absolute closest.
        # Robust to depth_scale drift since it uses in-frame ranking.
        self.rel_depth_drop_quantile = 0.35
        # Optional whitelist of class names; None means "show all".
        # Default to classes that actually matter for drone safety/ops.
        self.class_filter = {
            "person", "car", "truck", "bus", "bicycle", "motorcycle",
            "dog", "cat", "chair", "couch", "bed", "dining table",
        }

    def load_models(self):
        """Load models in background thread. Emits models_loaded(bool) when done.

        The signal is delivered to the main (GUI) thread automatically via Qt's
        queued-connection mechanism, so connected slots can safely touch widgets.
        """
        if self._loaded or self._loading:
            return
        self._loading = True
        self._depth_health.set_status(EngineStatus.LOADING)
        self._detect_health.set_status(EngineStatus.LOADING)

        def _load():
            try:
                self._device, self._model_dtype = _select_device()
                log_det.info("Device: %s (dtype: %s)",
                             self._device, self._model_dtype)

                log_det.info("Loading %s (%s edition) ...",
                             EDITION.detect_model.__name__, EDITION.name)
                self._det_model = EDITION.detect_model()
                # torch.compile (TorchInductor) gives ~1.5-2x steady-state
                # speedup but triggers heavy GPU kernel compilation that can
                # stress the NVIDIA + Wayland compositor combo on Pop!_OS
                # (we've seen it crash the Wayland connection on first run).
                # Off by default; flip on via RFDETR_COMPILE=1 once verified.
                # optimize_for_inference assumes an accelerator (fp16 + GPU
                # kernels); on CPU it just errors, so we only call it when
                # CUDA/MPS is in play.
                _use_compile = os.environ.get("RFDETR_COMPILE", "0") == "1"
                if self._device.type in ("cuda", "mps"):
                    try:
                        self._det_model.optimize_for_inference(
                            compile=_use_compile, dtype=self._model_dtype
                        )
                        tag = "fp16, compiled" if _use_compile else "fp16"
                        log_det.info("RF-DETR optimized for inference (%s)", tag)
                    except Exception as e:
                        log_det.warning("optimize_for_inference skipped: %s", e)
                else:
                    log_det.info("RF-DETR running on CPU (fp32) — install "
                                 "CUDA/MPS PyTorch for GPU acceleration")

                model_id = DEPTH_MODELS[self.depth_variant]
                log_det.info("Loading depth model: %s", model_id)
                self._depth_processor = AutoImageProcessor.from_pretrained(model_id)
                self._depth_model = AutoModelForDepthEstimation.from_pretrained(
                    model_id, torch_dtype=self._model_dtype
                ).to(self._device)
                self._depth_model.eval()
                self._cache_depth_preproc()

                # ── Object re-ID tracker: DeepSort + MobileNet appearance ──
                try:
                    from deep_sort_realtime.deepsort_tracker import DeepSort
                    self._deepsort = DeepSort(
                        max_age=self._ds_max_age,
                        n_init=self._ds_n_init,
                        max_iou_distance=0.70,
                        nn_budget=100,
                        embedder="mobilenet",
                        half=(self._device.type == "cuda"),
                    )
                    log_det.info("DeepSort re-ID tracker ready")
                except Exception as e:
                    self._deepsort = None
                    log_det.warning("DeepSort unavailable, IoU tracker used: %s", e)

                # Warmup pass: a single dummy forward through each model to
                # absorb cuDNN algo-selection and torch.compile graph capture
                # before the first real frame arrives. Without this the first
                # toggle-on click can take 10-30 s to produce its first box.
                try:
                    self._warmup()
                    log_det.info("Warmup complete")
                except Exception as e:
                    log_det.warning("Warmup skipped: %s", e)

                # Start depth worker thread
                self._depth_thread = threading.Thread(target=self._depth_worker, daemon=True)
                self._depth_thread.start()

                # Start async detection worker so process_frame() never blocks.
                self._det_thread = threading.Thread(
                    target=self._detection_worker, daemon=True)
                self._det_thread.start()

                self._loaded = True
                self._loading = False
                self._depth_health.set_status(EngineStatus.READY)
                self._detect_health.set_status(EngineStatus.READY)
                log_det.info("Models loaded OK")
                self.models_loaded.emit(True)
            except Exception as e:
                log_det.exception("Load failed: %s", e)
                self._loading = False
                self._depth_health.set_status(EngineStatus.FAILED)
                self._detect_health.set_status(EngineStatus.FAILED)
                self.models_loaded.emit(False)

        threading.Thread(target=_load, daemon=True).start()

    @property
    def is_ready(self):
        return self._loaded

    def _depth_worker(self):
        while not self._depth_stop.is_set():
            self._depth_pending.wait(timeout=0.5)
            self._depth_pending.clear()
            # Heartbeat even when no frame is available so the watchdog
            # doesn't misread an idle worker as a stalled one.
            self._depth_health.heartbeat()
            # Cadence cap (see _detection_worker). Depth is the heaviest
            # model; capping it to depth_target_fps is the biggest CPU win.
            if self._depth_min_interval:
                now = time.perf_counter()
                if (now - self._last_depth_proc) < self._depth_min_interval:
                    continue
                self._last_depth_proc = now
            with self._lock:
                frame = self._depth_frame
            if frame is None:
                continue
            t0 = time.perf_counter()
            try:
                depth = self._predict_depth(frame)
            except Exception as e:
                log_det.error("depth worker error: %s", e)
                self._depth_health.set_status(EngineStatus.DEGRADED)
                continue
            ms = (time.perf_counter() - t0) * 1000
            self._depth_health.record_inference(ms)
            with self._lock:
                self._depth_map = depth
                self._depth_version += 1
                self._depth_ms = ms

    def _cache_depth_preproc(self):
        """Pull resize size + mean/std out of the HF processor once at load
        time so per-frame inference can skip the PIL roundtrip entirely.

        The HF AutoImageProcessor does its resize and normalize on CPU
        via PIL — that single call dominates depth latency on a fast GPU.
        Doing the same work with cv2.resize + a precomputed GPU tensor
        cuts per-frame preproc from ~15-25 ms down to ~1-2 ms.
        """
        proc = self._depth_processor
        # Depth-Anything-V2 uses DPTImageProcessor: 'size' is {'height','width'}.
        size = getattr(proc, "size", None) or {}
        h = size.get("height") or size.get("shortest_edge") or 518
        w = size.get("width") or size.get("shortest_edge") or 518
        self._depth_in_hw = (int(h), int(w))

        mean = np.asarray(getattr(proc, "image_mean", [0.5, 0.5, 0.5]),
                          dtype=np.float32)
        std = np.asarray(getattr(proc, "image_std", [0.5, 0.5, 0.5]),
                         dtype=np.float32)
        rescale = float(getattr(proc, "rescale_factor", 1.0 / 255.0))
        # Preconvert to (1, 3, 1, 1) tensors on the model device using the
        # same dtype as the model so broadcast-normalize stays a single
        # fused op (no implicit cast). On CPU this is fp32, on CUDA/MPS fp16.
        self._depth_mean_t = torch.from_numpy(mean).to(
            self._device, dtype=self._model_dtype).view(1, 3, 1, 1)
        self._depth_std_t = torch.from_numpy(std).to(
            self._device, dtype=self._model_dtype).view(1, 3, 1, 1)
        self._depth_rescale = rescale

    @torch.inference_mode()
    def _predict_depth(self, bgr_frame):
        th, tw = self._depth_in_hw
        # cv2 resize is much faster than PIL.Image.resize and matches HF's
        # bilinear default closely enough for monocular depth.
        resized = cv2.resize(bgr_frame, (tw, th), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # HWC uint8 -> 1xCxHxW on device in the model dtype, normalize on GPU.
        # non_blocking is a CUDA pinned-memory optimization; harmless elsewhere.
        t = torch.from_numpy(rgb).to(self._device, non_blocking=True)
        t = t.permute(2, 0, 1).unsqueeze(0).to(self._model_dtype)
        t = t * self._depth_rescale
        t = (t - self._depth_mean_t) / self._depth_std_t
        out = self._depth_model(pixel_values=t)
        depth = out.predicted_depth.squeeze().float().cpu().numpy()
        return cv2.resize(depth, (bgr_frame.shape[1], bgr_frame.shape[0]))

    def _warmup(self):
        """Run one dummy forward through both models to absorb cuDNN algo
        selection and torch.compile graph capture before live frames arrive."""
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        # Depth warmup
        try:
            _ = self._predict_depth(dummy)
        except Exception as e:
            log_det.error("depth warmup error: %s", e)
        # Detection warmup — RF-DETR.predict accepts a BGR ndarray.
        try:
            _ = self._det_model.predict(dummy, threshold=self.conf_thresh)
        except Exception as e:
            log_det.error("detect warmup error: %s", e)

    def _sample_depth(self, depth_map, x1, y1, x2, y2):
        h, w = depth_map.shape
        bw, bh = x2 - x1, y2 - y1
        if bw < 4 or bh < 4:
            return None
        cx1 = max(0, x1 + bw // 4)
        cy1 = max(0, y1 + bh // 4)
        cx2 = min(w, x2 - bw // 4)
        cy2 = min(h, y2 - bh // 4)
        region = depth_map[cy1:cy2, cx1:cx2]
        if region.size == 0:
            return None
        return float(np.percentile(region, 25))

    def _compute_distance(self, raw_depth, altitude, pitch_deg, obj_cy_norm):
        scaled = raw_depth * self.depth_scale
        theta = math.radians(pitch_deg)
        sin_abs = abs(math.sin(theta))
        if altitude > 0.5 and sin_abs > 0.1:
            geo_los = altitude / sin_abs
            geo_w = 0.3 + 0.4 * obj_cy_norm
            scaled = min(scaled, geo_los * 1.2)
            scaled = scaled * (1 - geo_w) + geo_los * geo_w
        ground = abs(scaled * math.cos(theta))
        vert = abs(scaled * math.sin(theta))
        return {"los_m": raw_depth, "scaled_los_m": scaled,
                "ground_m": ground, "vert_m": vert}

    def process_frame(self, frame, altitude=0.0, pitch_deg=None):
        """Non-blocking: submit frame to async workers, draw latest cached result.

        Returns ``(annotated_frame, detection_list)``. The detection list may
        come from a frame or two ago (whatever the detection worker has
        most-recently produced), but drawing always happens on the *current*
        frame so video FPS stays at camera FPS.
        """
        if not self._loaded:
            return frame, []

        pitch = pitch_deg if pitch_deg is not None else self.camera_pitch

        # Submit current frame to BOTH async workers. Latest-frame-wins.
        # Hand the workers their own clean snapshot: the GUI thread is about
        # to draw detection boxes / depth zones onto `frame` below, and the
        # workers must run inference on the un-annotated pixels (and not race
        # the in-place drawing). One copy is shared by both read-only workers.
        clean = frame.copy()
        with self._lock:
            self._depth_frame = clean
            self._det_input_frame = clean
            self._det_altitude = altitude
            self._det_pitch = pitch
        self._depth_pending.set()
        self._det_pending.set()

        # Read latest cached detections + draw them on the current frame.
        with self._lock:
            det_list = list(self._latest_detections)
        display = self._draw_detections(frame, det_list)
        return display, det_list

    def _detection_worker(self):
        """Background loop: detect + compute distances using the latest depth."""
        while not self._det_stop.is_set():
            self._det_pending.wait(timeout=0.5)
            self._det_pending.clear()
            self._detect_health.heartbeat()

            # Cadence cap: if we ran too recently, skip this wake. The newest
            # frame stays cached (latest-frame-wins) and the next wake within
            # the interval processes it; the display keeps drawing the prior
            # result every frame, so video FPS is unaffected.
            if self._det_min_interval:
                now = time.perf_counter()
                if (now - self._last_det_proc) < self._det_min_interval:
                    continue
                self._last_det_proc = now

            with self._lock:
                frame = self._det_input_frame
                altitude = self._det_altitude
                pitch = self._det_pitch
                self._det_input_frame = None
            if frame is None:
                continue

            try:
                t0 = time.perf_counter()
                detections_sv = self._det_model.predict(
                    frame, threshold=self.conf_thresh)
                det_ms = (time.perf_counter() - t0) * 1000

                with self._lock:
                    depth_map = self._depth_map

                h_frame, w_frame = frame.shape[:2]

                if depth_map is not None and altitude > 0.5:
                    ground_strip = depth_map[int(depth_map.shape[0] * 0.7):, :]
                    if ground_strip.size > 0:
                        gd = float(np.percentile(ground_strip, 25))
                        if gd > 0.1:
                            theta = math.radians(pitch)
                            sin_abs = abs(math.sin(theta))
                            if sin_abs > 0.15:
                                new_scale = max(0.2,
                                                min(5.0, (altitude / sin_abs) / gd))
                                self.depth_scale = (self.depth_scale * 0.75
                                                    + new_scale * 0.25)

                new_dets = []
                seen_keys = set()
                boxes = detections_sv.xyxy
                scores = detections_sv.confidence
                class_ids = detections_sv.class_id

                for idx in range(len(detections_sv)):
                    x1, y1, x2, y2 = map(int, boxes[idx])
                    conf = float(scores[idx])
                    name = COCO_NAMES.get(int(class_ids[idx]), str(class_ids[idx]))

                    dist_m = None
                    if depth_map is not None:
                        raw_d = self._sample_depth(depth_map, x1, y1, x2, y2)
                        if raw_d is not None:
                            obj_cy = ((y1 + y2) / 2.0) / h_frame
                            dists = self._compute_distance(
                                raw_d, altitude, pitch, obj_cy)

                            gx = int(((x1 + x2) / 2.0) / w_frame * 8)
                            gy = int(obj_cy * 8)
                            skey = f"{name}_{gx}_{gy}"
                            seen_keys.add(skey)
                            if skey not in self._dist_history:
                                self._dist_history[skey] = deque(maxlen=5)
                            self._dist_history[skey].append(dists["scaled_los_m"])
                            smoothed = float(np.mean(self._dist_history[skey]))
                            dist_m = smoothed

                    new_dets.append({
                        "class": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf, "distance_m": dist_m,
                        "is_danger": name in DANGER_CLASSES,
                    })

                for k in list(self._dist_history.keys()):
                    if k not in seen_keys:
                        del self._dist_history[k]

                # Assign stable ids: DeepSort (appearance re-id, survives
                # occlusion) when available, else the lightweight IoU tracker.
                if self._deepsort is not None:
                    self._assign_deepsort_ids(new_dets, frame)
                else:
                    self._assign_track_ids(new_dets, time.time())

                with self._lock:
                    self._latest_detections = new_dets
                    self._det_ms = det_ms
                self._detect_health.record_inference(det_ms)
            except Exception as e:
                log_det.error("worker error: %s", e)
                self._detect_health.set_status(EngineStatus.DEGRADED)

    def _assign_deepsort_ids(self, new_dets, frame):
        """Run DeepSort (Kalman motion + MobileNet appearance embedding) so each
        object keeps a stable id across short occlusions and position jumps.

        Mutates each det in-place adding 'id' and 'count'. Detections whose
        track is still tentative (not confirmed yet) get no id, so the caller
        skips logging them — that avoids duplicate rows during track warm-up.
        """
        ds_input = [([d["x1"], d["y1"], d["x2"] - d["x1"], d["y2"] - d["y1"]],
                     d["conf"], d["class"]) for d in new_dets]
        try:
            tracks = self._deepsort.update_tracks(ds_input, frame=frame)
        except Exception as e:
            log_det.error("DeepSort update failed: %s", e)
            return

        # Confirmed tracks updated this frame, as boxes we can match to dets.
        confirmed = []
        for tr in tracks:
            if not tr.is_confirmed() or getattr(tr, "time_since_update", 0) != 0:
                continue
            l, t, r, b = tr.to_ltrb()
            confirmed.append((tr.track_id, {"x1": l, "y1": t, "x2": r, "y2": b}))

        used = set()
        for d in new_dets:
            best_tid, best_iou = None, 0.3
            for tid, box in confirmed:
                if tid in used:
                    continue
                iou = self._iou(d, box)
                if iou >= best_iou:
                    best_iou, best_tid = iou, tid
            if best_tid is None:
                continue   # tentative track -> don't log this object yet
            used.add(best_tid)
            self._track_hits[best_tid] = self._track_hits.get(best_tid, 0) + 1
            try:
                d["id"] = int(best_tid)
            except (ValueError, TypeError):
                d["id"] = best_tid
            d["count"] = self._track_hits[best_tid]

    def _assign_track_ids(self, new_dets, now):
        """Match this frame's detections to existing tracks by class + IoU so
        each physical object keeps one stable id across frames. Unmatched
        detections get a fresh id; tracks unseen for _track_ttl seconds expire.

        Mutates each det in-place, adding 'id' (int) and 'count' (how many
        detection frames this object has appeared in so far).
        """
        # Expire stale tracks
        for tid in list(self._tracks.keys()):
            if now - self._tracks[tid]["last_seen"] > self._track_ttl:
                del self._tracks[tid]

        used = set()
        for d in new_dets:
            best_id, best_iou = None, self._track_iou_thresh
            for tid, tr in self._tracks.items():
                if tid in used or tr["class"] != d["class"]:
                    continue
                iou = self._iou(d, tr)
                if iou >= best_iou:
                    best_iou, best_id = iou, tid

            if best_id is None:
                best_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[best_id] = {
                    "class": d["class"], "x1": d["x1"], "y1": d["y1"],
                    "x2": d["x2"], "y2": d["y2"], "last_seen": now, "hits": 0,
                }

            tr = self._tracks[best_id]
            tr.update(x1=d["x1"], y1=d["y1"], x2=d["x2"], y2=d["y2"],
                      last_seen=now)
            tr["hits"] += 1
            used.add(best_id)
            d["id"] = best_id
            d["count"] = tr["hits"]

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
        bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = max(1, (ax2 - ax1) * (ay2 - ay1))
        ub = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / float(ua + ub - inter)

    def _filter_for_display(self, det_list):
        """Reduce clutter: class filter, depth cutoffs, same-class IoU dedupe,
        top-K ranking. Depth signal comes from Depth Anything V2 via the
        per-detection distance_m the worker already computed."""
        if not det_list:
            return det_list

        # Optional class whitelist.
        if self.class_filter:
            allowed = set(self.class_filter)
            det_list = [d for d in det_list if d["class"] in allowed]
            if not det_list:
                return det_list

        # ── Absolute depth cutoff ───────────────────────────────────────
        # Drop anything farther than max_distance_m (meters). Detections
        # with no depth available are kept (don't penalise them).
        if self.max_distance_m is not None:
            cutoff = float(self.max_distance_m)
            det_list = [d for d in det_list
                        if d.get("distance_m") is None
                        or d["distance_m"] <= cutoff]
            if not det_list:
                return det_list

        # ── Relative depth cutoff ───────────────────────────────────────
        # Among detections we DO have depth for, drop those that fall in
        # the far tail of THIS frame's depth distribution. Survives even
        # if depth_scale calibration is off, since it's rank-based.
        depths = [d["distance_m"] for d in det_list
                  if d.get("distance_m") is not None]
        q = float(self.rel_depth_drop_quantile or 0.0)
        if depths and q > 0.0:
            # Keep boxes with distance_m <= the (1 - q) quantile of depths;
            # e.g. q=0.35 keeps the closest 65% and drops the farthest 35%.
            far_thresh = float(np.quantile(depths, max(0.0, 1.0 - q)))
            det_list = [d for d in det_list
                        if d.get("distance_m") is None
                        or d["distance_m"] <= far_thresh]
            if not det_list:
                return det_list

        # Optional greedy NMS within each class (keep higher confidence).
        # Disabled by default (iou_dedupe_thresh >= 1.0) so two real persons
        # standing next to each other aren't collapsed into one.
        if self.iou_dedupe_thresh is not None and self.iou_dedupe_thresh < 1.0:
            by_class = {}
            for d in det_list:
                by_class.setdefault(d["class"], []).append(d)
            kept = []
            for cls_dets in by_class.values():
                cls_dets.sort(key=lambda x: x["conf"], reverse=True)
                survivors = []
                for d in cls_dets:
                    if any(self._iou(d, s) > self.iou_dedupe_thresh
                           for s in survivors):
                        continue
                    survivors.append(d)
                kept.extend(survivors)
        else:
            kept = list(det_list)

        # Rank: danger first, then nearest distance, then highest confidence.
        def _rank_key(d):
            dist = d.get("distance_m")
            dist = dist if dist is not None else 1e6
            return (0 if d["is_danger"] else 1, dist, -d["conf"])

        kept.sort(key=_rank_key)
        if self.max_drawn is not None and len(kept) > self.max_drawn:
            kept = kept[: self.max_drawn]
        return kept

    @staticmethod
    def _draw_corner_box(img, x1, y1, x2, y2, color, thickness=2, frac=0.18):
        """Draw four corner brackets instead of a full rectangle."""
        w, h = x2 - x1, y2 - y1
        L = max(8, int(min(w, h) * frac))
        # top-left
        cv2.line(img, (x1, y1), (x1 + L, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1), (x1, y1 + L), color, thickness, cv2.LINE_AA)
        # top-right
        cv2.line(img, (x2, y1), (x2 - L, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1), (x2, y1 + L), color, thickness, cv2.LINE_AA)
        # bottom-left
        cv2.line(img, (x1, y2), (x1 + L, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y2), (x1, y2 - L), color, thickness, cv2.LINE_AA)
        # bottom-right
        cv2.line(img, (x2, y2), (x2 - L, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y2), (x2, y2 - L), color, thickness, cv2.LINE_AA)

    def _draw_detections(self, frame, det_list):
        """Render cached detections onto the current frame (main thread).

        Minimalist style: corner-bracket boxes, single compact label per box,
        same-class IoU dedupe, and a top-K cap (danger + nearest prioritised).
        Tuned via self.max_drawn / self.iou_dedupe_thresh / self.class_filter.
        """
        display = frame.copy()
        h_frame = display.shape[0]
        total_raw = len(det_list)
        draw_list = self._filter_for_display(det_list)

        for d in draw_list:
            x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
            name = d["class"]
            dist_m = d["distance_m"]
            is_danger = d["is_danger"]
            color = DETECT_COLORS.get(name, DETECT_DEFAULT_COLOR)
            thickness = 2 if is_danger else 1

            self._draw_corner_box(display, x1, y1, x2, y2, color, thickness)

            # Single compact tag: "person 2.3m"   (danger gets a leading "!")
            dist_str = f"  {dist_m:.1f}m" if dist_m is not None else ""
            prefix = "! " if is_danger else ""
            label = f"{prefix}{name}{dist_str}"
            (lw, lh), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            pad_x, pad_y = 4, 3
            ty = y1 - 4 if y1 - lh - 2 * pad_y > 0 else y2 + lh + 2 * pad_y
            tx = x1
            # Subtle dark backplate (semi-opaque) for legibility without bulk.
            bg_x1, bg_y1 = tx, ty - lh - pad_y
            bg_x2, bg_y2 = tx + lw + 2 * pad_x, ty + pad_y
            sub = display[max(0, bg_y1):max(0, bg_y2),
                          max(0, bg_x1):max(0, bg_x2)]
            if sub.size:
                sub[:] = cv2.addWeighted(sub, 0.45,
                                         np.zeros_like(sub), 0.0, 0)
            cv2.putText(display, label, (tx + pad_x, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        det_fps = 1000.0 / self._det_ms if self._det_ms > 0 else 0.0
        shown = len(draw_list)
        near_str = (f"near<{self.max_distance_m:.0f}m"
                    if self.max_distance_m is not None else "all depths")
        footer = (f"Det {det_fps:4.1f} FPS  ·  "
                  f"Showing {shown}/{total_raw}  ·  {near_str}")
        cv2.putText(display, footer, (10, h_frame - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)
        return display

    def get_depth_map(self):
        """Return the latest raw depth map (or None)."""
        with self._lock:
            return self._depth_map.copy() if self._depth_map is not None else None

    def get_depth_version(self):
        """Monotonic counter of how many depth maps have been produced.

        Cheap (no copy) — lets the display skip recolorizing identical depth
        data, which only changes at depth_target_fps.
        """
        with self._lock:
            return self._depth_version

    def stop(self):
        self._depth_stop.set()
        if self._depth_thread:
            self._depth_thread.join(timeout=2)


class VideoCapture(QThread):
    """Captures video from HDMI capture card (DJI O4 via /dev/video*).

    Frames are delivered with **drop-to-latest backpressure**: the capture
    loop always drains the camera (so the displayed frame is never stale),
    but only emits ``frame_ready`` when the GUI thread has finished the
    previous frame. Without this, ``frame_ready`` is a queued cross-thread
    signal that piles up unbounded the moment per-frame GUI work exceeds the
    camera interval — growing latency + memory and tanking perceived FPS.
    The consumer calls :meth:`frame_consumed` when it is ready for the next
    frame.
    """
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, device_index=0, max_width=0, max_height=0):
        super().__init__()
        self.device_index = device_index
        self.running = False
        # Resolution cap for the per-frame GUI pipeline (0 = no cap).
        self.max_width = int(max_width)
        self.max_height = int(max_height)
        # Starts set so the very first captured frame is emitted.
        self._consumer_ready = threading.Event()
        self._consumer_ready.set()

    def frame_consumed(self):
        """Signal that the GUI thread is ready for the next frame.

        Called from the consumer slot once it has finished compositing the
        previous frame. Safe to call from any thread.
        """
        self._consumer_ready.set()

    def run(self):
        self.running = True
        if sys.platform == "win32":
            # Try MSMF first (best for HDMI capture cards), fall back to DSHOW
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_MSMF)
            if not cap.isOpened():
                cap = cv2.VideoCapture(self.device_index, cv2.CAP_DSHOW)
        elif sys.platform == "darwin":
            # macOS: AVFoundation is the only working OpenCV backend; CAP_ANY
            # sometimes picks a deprecated QTKit path that fails to open.
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_AVFOUNDATION)
        else:
            cap = cv2.VideoCapture(self.device_index, cv2.CAP_ANY)
        if not cap.isOpened():
            log_cam.error("Cannot open camera %s.", self.device_index)
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Ask the card for the capped resolution. Many HDMI grabbers ignore
        # this and keep delivering native frames, so we also downscale below
        # as a guaranteed software fallback.
        if self.max_width > 0:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.max_width)
        if self.max_height > 0:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.max_height)
        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Software resolution cap (off the GUI thread). Preserve aspect
            # ratio; only ever shrink, never upscale.
            if self.max_width > 0 and self.max_height > 0:
                fh, fw = frame.shape[:2]
                if fw > self.max_width or fh > self.max_height:
                    scale = min(self.max_width / fw, self.max_height / fh)
                    frame = cv2.resize(
                        frame, (int(fw * scale), int(fh * scale)),
                        interpolation=cv2.INTER_AREA)

            # Drop-to-latest: only hand a frame to the GUI thread when it has
            # signalled it is ready for the next one. Frames captured while
            # the GUI is busy are simply dropped (we keep draining the camera
            # buffer above), so the consumer always gets the freshest frame
            # and the event queue never backs up.
            if self._consumer_ready.is_set():
                self._consumer_ready.clear()
                # Emit a private copy: cap.read() reuses its internal buffer
                # (BUFFERSIZE=1), so without this the consumer + inference
                # workers would read pixels being overwritten by the next read.
                self.frame_ready.emit(frame.copy())
        cap.release()

    def stop(self):
        self.running = False
        # Unblock any consumer-ready wait and let run() observe running=False.
        self._consumer_ready.set()
        self.wait(2000)


class NotificationToast(QWidget):
    """Floating notification toast that auto-hides."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setMinimumWidth(300)
        self.setStyleSheet("background: transparent;")
        self.hide()

        self._label = QLabel(self)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(
            "background-color: rgba(0,0,0,200); color: #ccc; "
            "border-radius: 8px; padding: 8px 16px; font-size: 12px; font-weight: bold;"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self.hide)

    def show_message(self, text, color="#00cc00", duration=3000):
        self._label.setText(text)
        self._label.setStyleSheet(
            f"background-color: rgba(0,0,0,210); color: {color}; "
            "border-radius: 8px; padding: 8px 16px; font-size: 12px; font-weight: bold; "
            f"border: 1px solid {color}40;"
        )
        self.show()
        # raise_() spams "Wayland does not support QWindow::requestActivate()" on Wayland.
        if QApplication.platformName() != "wayland":
            self.raise_()
        self._timer.start(duration)


class VideoFeedWidget(QLabel):
    """Video display with enforced 16:9 aspect ratio and advanced HUD overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #000; border: 1px solid #1a1a1a; border-radius: 8px;")
        self.setText("NO VIDEO FEED")
        self.setFont(QFont("Segoe UI", 16, QFont.Bold))

        # HUD data
        self.hud_voltage = 0.0
        self.hud_current = 0.0
        self.hud_batt_pct = 0
        self.hud_rssi = 0
        self.hud_lq = 0
        self.hud_snr = 0
        self.hud_armed = False
        self.hud_flight_mode = "MANUAL"
        self.hud_visible = True
        self.hud_heading = 0.0
        self.hud_altitude = 0.0
        self.hud_speed = 0.0
        self.hud_pitch = 0.0
        self.hud_roll = 0.0
        self.hud_satellites = 0
        self.hud_recording = False
        self.hud_record_time = 0
        self.hud_depth_mode = None  # None, "CLEAR", "AVOID_L", "AVOID_R", "STOP"

        # Aspect ratio: 16:9 for DJI O4
        self.aspect_w = 16
        self.aspect_h = 9

        # HUD overlay cache: rebuilt at 10 Hz regardless of video framerate
        self._hud_overlay = None
        self._hud_overlay_size = (0, 0)
        self._hud_last_paint_ms = 0
        self._hud_interval_ms = 100

    def set_aspect_ratio(self, w, h):
        self.aspect_w = w
        self.aspect_h = h

    def update_hud(self, voltage, current, batt_pct, rssi, lq, snr, armed, flight_mode):
        self.hud_voltage = voltage
        self.hud_current = current
        self.hud_batt_pct = batt_pct
        self.hud_rssi = rssi
        self.hud_lq = lq
        self.hud_snr = snr
        self.hud_armed = armed
        self.hud_flight_mode = flight_mode

    def display_frame(self, frame):
        # Hot path: runs on the GUI thread at camera FPS. Feed the BGR
        # buffer straight to QImage via Format_BGR888 (Qt 5.14+) instead of
        # a full-frame cv2.cvtColor(BGR2RGB) every frame — that conversion
        # is an H*W*3 op on the main thread under the GIL, and it competes
        # with detection drawing and HUD compositing. QImage needs a
        # C-contiguous buffer; camera frames and the stacked buffer already
        # are, so the ascontiguousarray copy almost never fires.
        if _HAS_BGR888:
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Calculate aspect-ratio fit within widget
        lw, lh = self.width(), self.height()
        target_w = lw
        target_h = int(lw * self.aspect_h / self.aspect_w)
        if target_h > lh:
            target_h = lh
            target_w = int(lh * self.aspect_w / self.aspect_h)

        pixmap = QPixmap.fromImage(img).scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.FastTransformation)

        if self.hud_visible:
            self._composite_hud(pixmap)

        self.setPixmap(pixmap)

    def _composite_hud(self, pixmap):
        # HUD overlay is cached and only rebuilt every _hud_interval_ms (10 Hz)
        # so per-frame video stays smooth even though the HUD itself paints
        # 40+ QPainter primitives.
        now_ms = int(time.monotonic() * 1000)
        size_now = (pixmap.width(), pixmap.height())
        if (self._hud_overlay is None
                or self._hud_overlay_size != size_now
                or (now_ms - self._hud_last_paint_ms) >= self._hud_interval_ms):
            self._hud_overlay = self._build_hud_overlay(*size_now)
            self._hud_overlay_size = size_now
            self._hud_last_paint_ms = now_ms
        p = QPainter(pixmap)
        p.drawPixmap(0, 0, self._hud_overlay)
        p.end()

    def _build_hud_overlay(self, pw, ph):
        overlay = QPixmap(pw, ph)
        overlay.fill(Qt.transparent)
        p = QPainter(overlay)
        p.setRenderHint(QPainter.Antialiasing)

        hud_bg = QColor(0, 0, 0, 150)
        hud_font = QFont("Segoe UI", 10, QFont.Bold)
        hud_font_sm = QFont("Segoe UI", 9)
        hud_font_xs = QFont("Segoe UI", 8)
        p.setFont(hud_font)

        # ── Top-left: Battery info ──
        p.fillRect(8, 8, 210, 56, hud_bg)
        v = self.hud_voltage
        if v > 0:
            v_color = QColor(0, 255, 0) if v > 14.0 else QColor(255, 255, 0) if v > 13.2 else QColor(255, 60, 60)
        else:
            v_color = QColor(100, 100, 100)
        p.setPen(v_color)
        p.drawText(16, 28, f"{self.hud_voltage:.1f}V")
        p.setPen(QColor(200, 200, 200))
        p.drawText(80, 28, f"{self.hud_current:.1f}A")
        p.setFont(hud_font_xs)
        p.setPen(QColor(150, 150, 150))
        p.drawText(140, 28, f"{self.hud_batt_pct}%")

        # Battery bar
        pct = max(0, min(100, self.hud_batt_pct))
        pct_color = QColor(0, 255, 0) if pct > 40 else QColor(255, 255, 0) if pct > 20 else QColor(255, 60, 60)
        p.fillRect(16, 38, 190, 10, QColor(40, 40, 40))
        p.fillRect(16, 38, int(190 * pct / 100), 10, pct_color)

        # Battery segments
        p.setPen(QPen(QColor(0, 0, 0, 100), 1))
        for seg in range(1, 10):
            sx = 16 + int(190 * seg / 10)
            p.drawLine(sx, 38, sx, 48)

        # ── Top-right: Link info ──
        p.setFont(hud_font)
        p.fillRect(pw - 188, 8, 180, 56, hud_bg)
        lq_color = QColor(0, 255, 0) if self.hud_lq > 70 else QColor(255, 255, 0) if self.hud_lq > 30 else QColor(255, 60, 60)
        p.setPen(lq_color)
        p.drawText(pw - 180, 28, f"LQ {self.hud_lq}%")

        # LQ bar
        p.fillRect(pw - 180, 34, 160, 6, QColor(40, 40, 40))
        p.fillRect(pw - 180, 34, int(160 * self.hud_lq / 100), 6, lq_color)

        p.setPen(QColor(180, 180, 180))
        p.setFont(hud_font_sm)
        p.drawText(pw - 180, 55, f"RSSI -{self.hud_rssi} dBm  SNR {self.hud_snr}")

        # ── Top-center: GPS satellites ──
        if self.hud_satellites > 0:
            p.setFont(hud_font_sm)
            sat_text = f"SAT {self.hud_satellites}"
            sat_w = p.fontMetrics().horizontalAdvance(sat_text) + 20
            sat_x = (pw - sat_w) // 2
            p.fillRect(sat_x, 8, sat_w, 24, hud_bg)
            sat_color = QColor(0, 255, 0) if self.hud_satellites > 8 else QColor(255, 255, 0) if self.hud_satellites > 4 else QColor(255, 60, 60)
            p.setPen(sat_color)
            p.drawText(sat_x + 10, 25, sat_text)

        # ── Center: Artificial Horizon ──
        self._draw_artificial_horizon(p, pw, ph)

        # ── Center: Compass Tape ──
        self._draw_compass_tape(p, pw, ph)

        # ── Left side: Altitude tape ──
        self._draw_altitude_tape(p, pw, ph)

        # ── Right side: Speed tape ──
        self._draw_speed_tape(p, pw, ph)

        # ── Bottom-center: Arm state + flight mode ──
        p.setFont(hud_font)
        arm_text = "ARMED" if self.hud_armed else "DISARMED"
        arm_color = QColor(255, 40, 40) if self.hud_armed else QColor(0, 220, 0)
        mode_text = self.hud_flight_mode

        total_text = f"{arm_text}  |  {mode_text}"
        text_w = p.fontMetrics().horizontalAdvance(total_text) + 28
        box_x = (pw - text_w) // 2
        p.fillRect(box_x, ph - 40, text_w, 32, hud_bg)
        p.setPen(arm_color)
        p.drawText(box_x + 14, ph - 18, arm_text)
        p.setPen(QColor(200, 200, 200))
        arm_w = p.fontMetrics().horizontalAdvance(arm_text + "  |  ")
        p.drawText(box_x + 14 + arm_w, ph - 18, mode_text)

        # ── Recording indicator ──
        if self.hud_recording:
            p.setBrush(QBrush(QColor(255, 0, 0)))
            p.setPen(Qt.NoPen)
            p.drawEllipse(pw - 30, ph - 35, 14, 14)
            p.setFont(hud_font_sm)
            p.setPen(QColor(255, 255, 255))
            mins = self.hud_record_time // 60
            secs = self.hud_record_time % 60
            p.drawText(pw - 90, ph - 21, f"REC {mins:02d}:{secs:02d}")

        # ── Depth avoidance indicator ──
        if self.hud_depth_mode is not None:
            depth_colors = {
                "CLEAR":   QColor(0, 220, 0),
                "AVOID_L": QColor(255, 200, 0),
                "AVOID_R": QColor(255, 200, 0),
                "STOP":    QColor(255, 40, 40),
            }
            dc = depth_colors.get(self.hud_depth_mode, QColor(180, 180, 180))
            depth_label = f"DEPTH: {self.hud_depth_mode}"
            p.setFont(hud_font)
            dl_w = p.fontMetrics().horizontalAdvance(depth_label) + 20
            p.fillRect(pw - dl_w - 8, 8, dl_w, 28, hud_bg)
            p.setPen(dc)
            p.drawText(pw - dl_w + 2, 28, depth_label)

        p.end()
        return overlay

    def _draw_artificial_horizon(self, p, pw, ph):
        """Draw a compact artificial horizon indicator at screen center."""
        cx, cy = pw // 2, ph // 2
        size = min(pw, ph) // 6

        # Horizon line tilted by roll
        roll_rad = math.radians(self.hud_roll)
        pitch_offset = int(self.hud_pitch * size / 30)

        # Wing markers
        p.setPen(QPen(QColor(0, 255, 0, 180), 2))
        # Left wing
        p.drawLine(cx - size, cy, cx - size // 3, cy)
        # Right wing
        p.drawLine(cx + size // 3, cy, cx + size, cy)
        # Center dot
        p.drawLine(cx - 2, cy, cx + 2, cy)
        p.drawLine(cx, cy - 2, cx, cy + 2)

        # Pitch ladder lines (simplified)
        p.setPen(QPen(QColor(0, 255, 0, 80), 1))
        for deg in [-10, -5, 5, 10]:
            y = cy - int(deg * size / 30) + pitch_offset
            half = size // 4 if abs(deg) == 5 else size // 3
            p.drawLine(cx - half, y, cx + half, y)

    def _draw_compass_tape(self, p, pw, ph):
        """Draw a compass heading tape at the top-center."""
        tape_w = min(260, pw // 3)
        tape_h = 22
        tape_x = (pw - tape_w) // 2
        tape_y = ph - 75

        p.fillRect(tape_x, tape_y, tape_w, tape_h, QColor(0, 0, 0, 150))

        heading = self.hud_heading % 360
        p.setFont(QFont("Segoe UI", 8))
        p.setClipRect(tape_x, tape_y, tape_w, tape_h)

        cardinals = {0: "N", 45: "NE", 90: "E", 135: "SE", 180: "S", 225: "SW", 270: "W", 315: "NW"}

        for offset in range(-60, 61, 5):
            deg = (int(heading) + offset) % 360
            x = tape_x + tape_w // 2 + int(offset * tape_w / 120)
            if tape_x <= x <= tape_x + tape_w:
                if deg % 30 == 0:
                    p.setPen(QPen(QColor(200, 200, 200), 1))
                    p.drawLine(x, tape_y, x, tape_y + 8)
                    label = cardinals.get(deg, str(deg))
                    lw = p.fontMetrics().horizontalAdvance(label)
                    p.drawText(x - lw // 2, tape_y + 18, label)
                elif deg % 10 == 0:
                    p.setPen(QPen(QColor(100, 100, 100), 1))
                    p.drawLine(x, tape_y, x, tape_y + 5)

        p.setClipping(False)

        # Center triangle
        p.setPen(QPen(QColor(0, 255, 0), 2))
        cx = tape_x + tape_w // 2
        p.drawLine(cx, tape_y - 4, cx - 4, tape_y)
        p.drawLine(cx, tape_y - 4, cx + 4, tape_y)

        # Heading readout
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        hdg_text = f"{int(heading):03d}"
        hdg_w = p.fontMetrics().horizontalAdvance(hdg_text) + 12
        p.fillRect(cx - hdg_w // 2, tape_y - 20, hdg_w, 16, QColor(0, 0, 0, 180))
        p.setPen(QColor(0, 255, 0))
        p.drawText(cx - hdg_w // 2 + 6, tape_y - 8, hdg_text)

    def _draw_altitude_tape(self, p, pw, ph):
        """Draw altitude indicator on the right side."""
        tape_x = pw - 60
        tape_y = ph // 2 - 50
        tape_h = 100

        p.fillRect(tape_x, tape_y, 52, tape_h, QColor(0, 0, 0, 140))

        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.setPen(QColor(0, 255, 0))
        alt_text = f"{self.hud_altitude:.0f}m"
        p.drawText(tape_x + 6, ph // 2 + 4, alt_text)

        p.setFont(QFont("Segoe UI", 7))
        p.setPen(QColor(100, 100, 100))
        p.drawText(tape_x + 6, tape_y + 12, "ALT")

    def _draw_speed_tape(self, p, pw, ph):
        """Draw speed indicator on the left side."""
        tape_x = 8
        tape_y = ph // 2 - 50
        tape_h = 100

        p.fillRect(tape_x, tape_y, 52, tape_h, QColor(0, 0, 0, 140))

        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.setPen(QColor(0, 255, 0))
        spd_text = f"{self.hud_speed:.1f}"
        p.drawText(tape_x + 6, ph // 2 + 4, spd_text)

        p.setFont(QFont("Segoe UI", 7))
        p.setPen(QColor(100, 100, 100))
        p.drawText(tape_x + 6, tape_y + 12, "SPD m/s")


# ─── Tab: Control Panel ─────────────────────────────────────────────

class ControlPanelTab(QWidget):
    """ARM/DISARM, takeoff/land, RTH, emergency stop, flight modes, health,
    safety lock, preflight checklist, gimbal controls, camera triggers."""

    arm_requested = pyqtSignal(bool)
    flight_mode_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._armed = False
        self._rc_connected = False
        self._safety_locked = True
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(10)

        # ── Safety Lock (RosettaDrone-style) ──
        safety_group = QGroupBox("Safety")
        safety_layout = QHBoxLayout(safety_group)

        self.safety_btn = QPushButton("SAFETY LOCKED")
        self.safety_btn.setMinimumHeight(44)
        self.safety_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a0a0a; color: #ff4444;
                font-size: 14px; font-weight: bold;
                border: 2px solid #cc0000; border-radius: 6px;
            }
            QPushButton:hover { background-color: #2a0a0a; }
        """)
        self.safety_btn.clicked.connect(self._toggle_safety)
        safety_layout.addWidget(self.safety_btn)

        self.safety_indicator = QLabel("")
        self.safety_indicator.setFixedSize(18, 18)
        self.safety_indicator.setStyleSheet(
            "background-color: #cc0000; border-radius: 9px; border: 1px solid #880000;"
        )
        safety_layout.addWidget(self.safety_indicator)

        layout.addWidget(safety_group)

        # ── Command Buttons ──
        cmd_group = QGroupBox("Commands")
        cmd_layout = QGridLayout(cmd_group)
        cmd_layout.setSpacing(8)

        self.arm_btn = QPushButton("ARM")
        self.arm_btn.setMinimumHeight(50)
        self.arm_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a2a1a; color: #00dd00;
                font-size: 16px; font-weight: bold;
                border: 2px solid #00aa00; border-radius: 6px;
            }
            QPushButton:hover { background-color: #1a3a1a; }
        """)
        self.arm_btn.clicked.connect(self._toggle_arm)
        cmd_layout.addWidget(self.arm_btn, 0, 0, 1, 2)

        self.takeoff_btn = QPushButton("TAKEOFF")
        self.takeoff_btn.setMinimumHeight(40)
        self.takeoff_btn.setStyleSheet("""
            QPushButton {
                background-color: #0f1520; color: #5588ee;
                font-size: 13px; font-weight: bold;
                border: 1px solid #3366bb; border-radius: 6px;
            }
            QPushButton:hover { background-color: #152030; }
        """)
        cmd_layout.addWidget(self.takeoff_btn, 1, 0)

        self.land_btn = QPushButton("LAND")
        self.land_btn.setMinimumHeight(40)
        self.land_btn.setStyleSheet("""
            QPushButton {
                background-color: #0f1520; color: #5588ee;
                font-size: 13px; font-weight: bold;
                border: 1px solid #3366bb; border-radius: 6px;
            }
            QPushButton:hover { background-color: #152030; }
        """)
        cmd_layout.addWidget(self.land_btn, 1, 1)

        self.rth_btn = QPushButton("RTH")
        self.rth_btn.setMinimumHeight(40)
        self.rth_btn.setStyleSheet("""
            QPushButton {
                background-color: #201a0a; color: #ffcc00;
                font-size: 13px; font-weight: bold;
                border: 1px solid #aa8800; border-radius: 6px;
            }
            QPushButton:hover { background-color: #302a0a; }
        """)
        cmd_layout.addWidget(self.rth_btn, 2, 0)

        self.estop_btn = QPushButton("EMERGENCY STOP")
        self.estop_btn.setMinimumHeight(40)
        self.estop_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a0505; color: #ff3333;
                font-size: 13px; font-weight: bold;
                border: 2px solid #ff0000; border-radius: 6px;
            }
            QPushButton:hover { background-color: #3a0a0a; }
        """)
        cmd_layout.addWidget(self.estop_btn, 2, 1)

        layout.addWidget(cmd_group)

        # ── Flight Mode ──
        mode_group = QGroupBox("Flight Mode")
        mode_layout = QHBoxLayout(mode_group)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["MANUAL", "STABILIZE", "ALT_HOLD", "GUIDED", "AUTO", "LOITER", "RTH"])
        self.mode_combo.setStyleSheet("QComboBox { font-size: 13px; font-weight: bold; min-height: 30px; }")
        self.mode_combo.currentTextChanged.connect(lambda m: self.flight_mode_changed.emit(m))
        mode_layout.addWidget(self.mode_combo)

        layout.addWidget(mode_group)

        # ── Camera Controls (RosettaDrone-style) ──
        cam_group = QGroupBox("Camera")
        cam_layout = QGridLayout(cam_group)
        cam_layout.setSpacing(6)

        self.photo_btn = QPushButton("PHOTO")
        self.photo_btn.setMinimumHeight(36)
        self.photo_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a1a20; color: #aabbff;
                font-size: 12px; font-weight: bold;
                border: 1px solid #4455aa; border-radius: 6px;
            }
            QPushButton:hover { background-color: #252530; }
        """)
        cam_layout.addWidget(self.photo_btn, 0, 0)

        self.record_btn = QPushButton("REC")
        self.record_btn.setCheckable(True)
        self.record_btn.setMinimumHeight(36)
        self.record_btn.setStyleSheet("""
            QPushButton {
                background-color: #1a1a20; color: #aabbff;
                font-size: 12px; font-weight: bold;
                border: 1px solid #4455aa; border-radius: 6px;
            }
            QPushButton:checked {
                background-color: #2a0505; color: #ff3333;
                border-color: #ff0000;
            }
            QPushButton:hover { background-color: #252530; }
        """)
        cam_layout.addWidget(self.record_btn, 0, 1)

        layout.addWidget(cam_group)

        # ── Gimbal Controls ──
        gimbal_group = QGroupBox("Gimbal")
        gimbal_layout = QFormLayout(gimbal_group)

        self.gimbal_pitch = QSlider(Qt.Horizontal)
        self.gimbal_pitch.setRange(-90, 30)
        self.gimbal_pitch.setValue(0)
        self.gimbal_pitch_label = QLabel("0")
        self.gimbal_pitch_label.setStyleSheet("color: #00cc00; font-weight: bold; min-width: 30px;")
        self.gimbal_pitch.valueChanged.connect(lambda v: self.gimbal_pitch_label.setText(f"{v}"))
        pitch_row = QHBoxLayout()
        pitch_row.addWidget(self.gimbal_pitch)
        pitch_row.addWidget(self.gimbal_pitch_label)
        gimbal_layout.addRow("Pitch:", pitch_row)

        self.gimbal_yaw = QSlider(Qt.Horizontal)
        self.gimbal_yaw.setRange(-180, 180)
        self.gimbal_yaw.setValue(0)
        self.gimbal_yaw_label = QLabel("0")
        self.gimbal_yaw_label.setStyleSheet("color: #00cc00; font-weight: bold; min-width: 30px;")
        self.gimbal_yaw.valueChanged.connect(lambda v: self.gimbal_yaw_label.setText(f"{v}"))
        yaw_row = QHBoxLayout()
        yaw_row.addWidget(self.gimbal_yaw)
        yaw_row.addWidget(self.gimbal_yaw_label)
        gimbal_layout.addRow("Yaw:", yaw_row)

        gimbal_center_btn = QPushButton("Center Gimbal")
        gimbal_center_btn.clicked.connect(lambda: (self.gimbal_pitch.setValue(0), self.gimbal_yaw.setValue(0)))
        gimbal_layout.addRow(gimbal_center_btn)

        layout.addWidget(gimbal_group)

        # ── Preflight Checklist (RosettaDrone-style) ──
        preflight_group = QGroupBox("Preflight Checklist")
        preflight_layout = QVBoxLayout(preflight_group)

        self.preflight_checks = {}
        checks = [
            ("Props inspected", False),
            ("Battery secured", False),
            ("GPS lock acquired", False),
            ("Area clear", False),
            ("Failsafe configured", False),
            ("Compass calibrated", False),
        ]
        for name, default in checks:
            cb = QCheckBox(name)
            cb.setChecked(default)
            cb.setStyleSheet("QCheckBox { color: #aaa; font-size: 11px; padding: 2px; }")
            cb.stateChanged.connect(self._update_preflight_status)
            preflight_layout.addWidget(cb)
            self.preflight_checks[name] = cb

        self.preflight_status = QLabel("0/6 checks passed")
        self.preflight_status.setStyleSheet("color: #ff6644; font-weight: bold; font-size: 11px; padding: 4px;")
        preflight_layout.addWidget(self.preflight_status)

        layout.addWidget(preflight_group)

        # ── Drone Health ──
        health_group = QGroupBox("Drone Health")
        health_grid = QGridLayout(health_group)
        health_grid.setSpacing(6)

        self.health_indicators = {}
        indicators = [
            ("Battery", "#888"),
            ("Signal", "#888"),
            ("GPS", "#888"),
            ("Motors", "#888"),
            ("IMU", "#888"),
            ("Compass", "#888"),
            ("Baro", "#888"),
            ("Temp", "#888"),
        ]
        for i, (name, color) in enumerate(indicators):
            label = QLabel(f"{name}:")
            label.setStyleSheet("color: #555; font-size: 11px;")
            status = QLabel("--")
            status.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px;")
            row, col = divmod(i, 2)
            health_grid.addWidget(label, row, col * 2)
            health_grid.addWidget(status, row, col * 2 + 1)
            self.health_indicators[name] = status

        # Battery progress bar
        self.batt_bar = QProgressBar()
        self.batt_bar.setRange(0, 100)
        self.batt_bar.setValue(0)
        self.batt_bar.setFormat("%p%")
        health_grid.addWidget(QLabel("Batt:"), 4, 0)
        health_grid.addWidget(self.batt_bar, 4, 1, 1, 3)

        # Signal quality bar
        self.signal_bar = QProgressBar()
        self.signal_bar.setRange(0, 100)
        self.signal_bar.setValue(0)
        self.signal_bar.setFormat("%p%")
        health_grid.addWidget(QLabel("LQ:"), 5, 0)
        health_grid.addWidget(self.signal_bar, 5, 1, 1, 3)

        layout.addWidget(health_group)
        layout.addStretch()

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _toggle_safety(self):
        self._safety_locked = not self._safety_locked
        if self._safety_locked:
            self.safety_btn.setText("SAFETY LOCKED")
            self.safety_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1a0a0a; color: #ff4444;
                    font-size: 14px; font-weight: bold;
                    border: 2px solid #cc0000; border-radius: 6px;
                }
                QPushButton:hover { background-color: #2a0a0a; }
            """)
            self.safety_indicator.setStyleSheet(
                "background-color: #cc0000; border-radius: 9px; border: 1px solid #880000;"
            )
            self.arm_btn.setEnabled(False)
            self.takeoff_btn.setEnabled(False)
        else:
            self.safety_btn.setText("READY TO FLY")
            self.safety_btn.setStyleSheet("""
                QPushButton {
                    background-color: #0a1a0a; color: #00ff00;
                    font-size: 14px; font-weight: bold;
                    border: 2px solid #00cc00; border-radius: 6px;
                }
                QPushButton:hover { background-color: #0a2a0a; }
            """)
            self.safety_indicator.setStyleSheet(
                "background-color: #00cc00; border-radius: 9px; border: 1px solid #008800;"
            )
            self.arm_btn.setEnabled(True)
            self.takeoff_btn.setEnabled(True)

    def _update_preflight_status(self):
        passed = sum(1 for cb in self.preflight_checks.values() if cb.isChecked())
        total = len(self.preflight_checks)
        self.preflight_status.setText(f"{passed}/{total} checks passed")
        if passed == total:
            self.preflight_status.setStyleSheet("color: #00cc00; font-weight: bold; font-size: 11px; padding: 4px;")
        elif passed > total // 2:
            self.preflight_status.setStyleSheet("color: #cccc00; font-weight: bold; font-size: 11px; padding: 4px;")
        else:
            self.preflight_status.setStyleSheet("color: #ff6644; font-weight: bold; font-size: 11px; padding: 4px;")

    def _toggle_arm(self):
        if self._safety_locked:
            QMessageBox.warning(self, "Safety Lock", "Disengage the safety lock before arming.")
            return

        if self._armed:
            self.set_armed(False)
            self.arm_requested.emit(False)
        else:
            reply = QMessageBox.warning(
                self, "Confirm ARM",
                "Are you sure you want to ARM the drone?\n\n"
                "Ensure the area is clear and props are safe.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.set_armed(True)
                self.arm_requested.emit(True)

    def set_armed(self, armed):
        self._armed = armed
        if armed:
            self.arm_btn.setText("DISARM")
            self.arm_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2a0505; color: #ff3333;
                    font-size: 16px; font-weight: bold;
                    border: 2px solid #ff0000; border-radius: 6px;
                }
                QPushButton:hover { background-color: #3a0a0a; }
            """)
        else:
            self.arm_btn.setText("ARM")
            self.arm_btn.setStyleSheet("""
                QPushButton {
                    background-color: #1a2a1a; color: #00dd00;
                    font-size: 16px; font-weight: bold;
                    border: 2px solid #00aa00; border-radius: 6px;
                }
                QPushButton:hover { background-color: #1a3a1a; }
            """)

    def update_telemetry(self, telem):
        v = telem.battery_voltage
        self.batt_bar.setValue(telem.battery_percent)
        if v > 0:
            color = "#00cc00" if v > 14.0 else "#cccc00" if v > 13.2 else "#cc0000"
            self.health_indicators["Battery"].setText(f"{v:.1f}V")
            self.health_indicators["Battery"].setStyleSheet(f"color: {color}; font-weight: bold; font-size: 11px;")
            self.batt_bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {color}; border-radius: 3px; }}")

        lq = telem.link_quality
        self.signal_bar.setValue(lq)
        sig_color = "#00cc00" if lq > 70 else "#cccc00" if lq > 30 else "#cc0000"
        self.health_indicators["Signal"].setText(f"LQ {lq}%")
        self.health_indicators["Signal"].setStyleSheet(f"color: {sig_color}; font-weight: bold; font-size: 11px;")


# ─── Tab: Detection Feed ────────────────────────────────────────────

class DetectionFeedTab(QWidget):
    """Live AI detection events with filtering, stats dashboard, and export."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Bounded ring buffer: prevents the filter/stats pass from growing O(N)
        # unbounded over hours of operation. Older events drop off the left.
        self.events = deque(maxlen=5000)
        self._event_by_id = {}   # track id -> its row dict (one row per object)
        self._dirty = False
        self._build_ui()
        # Coalesce frequent detection updates into a timer so per-frame, per-
        # object add_event() calls don't flood the GUI thread (rebuilding the
        # whole table hundreds of times/sec can drop the Qt/Wayland connection).
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self._refresh_if_dirty)
        self._refresh_timer.start()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Stats Dashboard ──
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(8)

        self.stat_cards = {}
        card_defs = [
            ("Total", "0", "#00cc00"),
            ("Objects", "0", "#6688ff"),
            ("Identities", "0", "#cc88ff"),
            ("Obstacles", "0", "#ff8844"),
            ("Rate/min", "0.0", "#00aaaa"),
        ]
        for name, default, color in card_defs:
            card = self._make_stat_card(name, default, color)
            stats_layout.addWidget(card)
            self.stat_cards[name] = card

        layout.addLayout(stats_layout)

        # ── Filters ──
        filter_frame = QFrame()
        filter_frame.setStyleSheet("QFrame { background-color: #0f0f0f; border-radius: 6px; padding: 4px; }")
        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setContentsMargins(8, 6, 8, 6)

        filter_layout.addWidget(QLabel("Type:"))
        self.type_filter = QComboBox()
        self.type_filter.addItems(["All", "Object", "Identity", "Obstacle"])
        self.type_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.type_filter)

        filter_layout.addWidget(QLabel("Conf:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(50)
        self.conf_slider.setMaximumWidth(120)
        self.conf_slider.valueChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.conf_slider)
        self.conf_label = QLabel("50%")
        self.conf_label.setStyleSheet("color: #00cc00; font-weight: bold; min-width: 35px;")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_label.setText(f"{v}%"))
        filter_layout.addWidget(self.conf_label)

        filter_layout.addWidget(QLabel("Time:"))
        self.time_filter = QComboBox()
        self.time_filter.addItems(["All", "Last 1 min", "Last 5 min", "Last 15 min", "Last 1 hr"])
        self.time_filter.currentTextChanged.connect(self._apply_filters)
        filter_layout.addWidget(self.time_filter)

        # Overlay toggle
        self.overlay_toggle = QCheckBox("Show Overlay")
        self.overlay_toggle.setChecked(True)
        self.overlay_toggle.setStyleSheet("QCheckBox { color: #aaa; }")
        filter_layout.addWidget(self.overlay_toggle)

        # Sound toggle
        self.sound_toggle = QCheckBox("Alerts")
        self.sound_toggle.setChecked(False)
        self.sound_toggle.setStyleSheet("QCheckBox { color: #aaa; }")
        filter_layout.addWidget(self.sound_toggle)

        filter_layout.addStretch()

        self.event_count = QLabel("0 events")
        self.event_count.setStyleSheet("color: #00cc00; font-weight: bold;")
        filter_layout.addWidget(self.event_count)

        layout.addWidget(filter_frame)

        # ── Event table ──
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["ID", "Time", "Type", "Label", "Confidence", "Count", "Severity", "Details"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        # ── Buttons ──
        btn_layout = QHBoxLayout()

        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self._export_events)
        btn_layout.addWidget(export_btn)

        btn_layout.addStretch()

        clear_btn = QPushButton("Clear Events")
        clear_btn.clicked.connect(self._clear_events)
        btn_layout.addWidget(clear_btn)

        layout.addLayout(btn_layout)

    def _make_stat_card(self, title, value, color):
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: #0f0f0f;
                border: 1px solid #1a1a1a;
                border-radius: 8px;
                padding: 6px;
            }}
        """)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 8, 10, 8)
        card_layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setStyleSheet("color: #555; font-size: 9px; font-weight: bold; text-transform: uppercase;")
        title_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title_label)

        value_label = QLabel(value)
        value_label.setObjectName("value")
        value_label.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        value_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(value_label)

        return card

    def _update_stat_card(self, card, value):
        label = card.findChild(QLabel, "value")
        if label:
            label.setText(str(value))

    def add_event(self, event_type, label, confidence, details="", severity="Low",
                  obj_id=None, count=1):
        # One row per tracked object: if we've already logged this id, refresh
        # its row (last-seen time, count, best confidence, latest details)
        # instead of appending a duplicate. obj_id=None falls back to the old
        # append-everything behaviour.
        if obj_id is not None and obj_id in self._event_by_id:
            ev = self._event_by_id[obj_id]
            ev["time"] = datetime.now()
            ev["count"] = count
            ev["confidence"] = max(ev["confidence"], confidence)
            ev["details"] = details
            ev["severity"] = severity
        else:
            ev = {
                "id": obj_id,
                "time": datetime.now(),
                "type": event_type,
                "label": label,
                "confidence": confidence,
                "count": count,
                "details": details,
                "severity": severity,
            }
            self.events.append(ev)
            if obj_id is not None:
                self._event_by_id[obj_id] = ev
        # Coalesced refresh: mark dirty; the timer repaints a few times/sec
        # instead of rebuilding the whole table on every single detection.
        self._dirty = True

    def _refresh_if_dirty(self):
        if self._dirty:
            self._dirty = False
            self._apply_filters()
            self._update_stats()

    def _update_stats(self):
        total = len(self.events)
        objects = sum(1 for e in self.events if e["type"] == "Object")
        identities = sum(1 for e in self.events if e["type"] == "Identity")
        obstacles = sum(1 for e in self.events if e["type"] == "Obstacle")

        # Rate per minute (events in last 60s)
        now = datetime.now()
        recent = sum(1 for e in self.events if (now - e["time"]).total_seconds() < 60)

        self._update_stat_card(self.stat_cards["Total"], total)
        self._update_stat_card(self.stat_cards["Objects"], objects)
        self._update_stat_card(self.stat_cards["Identities"], identities)
        self._update_stat_card(self.stat_cards["Obstacles"], obstacles)
        self._update_stat_card(self.stat_cards["Rate/min"], f"{recent:.0f}")

    def _apply_filters(self):
        type_f = self.type_filter.currentText()
        conf_f = self.conf_slider.value()
        time_f = self.time_filter.currentText()

        now = datetime.now()
        time_deltas = {
            "All": None,
            "Last 1 min": timedelta(minutes=1),
            "Last 5 min": timedelta(minutes=5),
            "Last 15 min": timedelta(minutes=15),
            "Last 1 hr": timedelta(hours=1),
        }
        td = time_deltas.get(time_f)

        filtered = []
        for ev in self.events:
            if type_f != "All" and ev["type"] != type_f:
                continue
            if ev["confidence"] < conf_f:
                continue
            if td and (now - ev["time"]) > td:
                continue
            filtered.append(ev)

        self.table.setRowCount(len(filtered))
        for i, ev in enumerate(reversed(filtered)):
            id_str = str(ev["id"]) if ev.get("id") is not None else "-"
            self.table.setItem(i, 0, QTableWidgetItem(id_str))
            self.table.setItem(i, 1, QTableWidgetItem(ev["time"].strftime("%H:%M:%S")))

            type_item = QTableWidgetItem(ev["type"])
            type_colors = {"Object": "#6688ff", "Identity": "#cc88ff", "Obstacle": "#ff8844"}
            type_item.setForeground(QColor(type_colors.get(ev["type"], "#ccc")))
            self.table.setItem(i, 2, type_item)

            self.table.setItem(i, 3, QTableWidgetItem(ev["label"]))

            conf_item = QTableWidgetItem(f"{ev['confidence']}%")
            conf_color = QColor(0, 200, 0) if ev["confidence"] > 80 else QColor(200, 200, 0) if ev["confidence"] > 50 else QColor(200, 100, 0)
            conf_item.setForeground(conf_color)
            self.table.setItem(i, 4, conf_item)

            self.table.setItem(i, 5, QTableWidgetItem(f"x{ev.get('count', 1)}"))

            sev = ev.get("severity", "Low")
            sev_item = QTableWidgetItem(sev)
            sev_colors = {"Critical": "#ff3333", "High": "#ff8844", "Medium": "#cccc00", "Low": "#00cc00"}
            sev_item.setForeground(QColor(sev_colors.get(sev, "#888")))
            self.table.setItem(i, 6, sev_item)

            self.table.setItem(i, 7, QTableWidgetItem(ev.get("details", "")))

        self.event_count.setText(f"{len(filtered)} events")

    def _export_events(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Detection Events", "detections.csv", "CSV Files (*.csv)")
        if path:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["ID", "Time", "Type", "Label", "Confidence", "Count", "Severity", "Details"])
                for ev in self.events:
                    writer.writerow([
                        ev.get("id", ""),
                        ev["time"].strftime("%Y-%m-%d %H:%M:%S"),
                        ev["type"], ev["label"], ev["confidence"],
                        ev.get("count", 1),
                        ev.get("severity", "Low"), ev.get("details", "")
                    ])
            QMessageBox.information(self, "Export", f"Exported {len(self.events)} events to {path}")

    def _clear_events(self):
        self.events.clear()
        self._event_by_id.clear()
        self.table.setRowCount(0)
        self.event_count.setText("0 events")
        self._update_stats()


# ─── Tab: Flight Logs ───────────────────────────────────────────────

class FlightLogsTab(QWidget):
    """Searchable, filterable flight history with CSV/video export and summary stats."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.logs = []
        self._build_ui()
        self._load_sample_logs()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Summary Stats Row ──
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(8)

        self.stats_cards = {}
        for name, val, color in [
            ("Total Flights", "0", "#00cc00"),
            ("Total Time", "0:00", "#6688ff"),
            ("Total Distance", "0 km", "#ff8844"),
            ("Avg Duration", "0:00", "#cc88ff"),
        ]:
            card = QFrame()
            card.setStyleSheet(f"QFrame {{ background-color: #0f0f0f; border: 1px solid #1a1a1a; border-radius: 8px; }}")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(10, 8, 10, 8)
            cl.setSpacing(2)
            tl = QLabel(name)
            tl.setStyleSheet("color: #555; font-size: 9px; font-weight: bold;")
            tl.setAlignment(Qt.AlignCenter)
            cl.addWidget(tl)
            vl = QLabel(val)
            vl.setObjectName("value")
            vl.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
            vl.setAlignment(Qt.AlignCenter)
            cl.addWidget(vl)
            stats_layout.addWidget(card)
            self.stats_cards[name] = card

        layout.addLayout(stats_layout)

        # ── Search + filters ──
        search_frame = QFrame()
        search_frame.setStyleSheet("QFrame { background-color: #0f0f0f; border-radius: 6px; }")
        search_layout = QHBoxLayout(search_frame)
        search_layout.setContentsMargins(8, 6, 8, 6)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search flights...")
        self.search_input.textChanged.connect(self._apply_filters)
        search_layout.addWidget(self.search_input)

        search_layout.addWidget(QLabel("From:"))
        self.date_from = QDateEdit()
        self.date_from.setDate(QDate.currentDate().addMonths(-3))
        self.date_from.setCalendarPopup(True)
        self.date_from.dateChanged.connect(self._apply_filters)
        search_layout.addWidget(self.date_from)

        search_layout.addWidget(QLabel("To:"))
        self.date_to = QDateEdit()
        self.date_to.setDate(QDate.currentDate())
        self.date_to.setCalendarPopup(True)
        self.date_to.dateChanged.connect(self._apply_filters)
        search_layout.addWidget(self.date_to)

        self.mode_filter = QComboBox()
        self.mode_filter.addItems(["All Modes", "MANUAL", "GUIDED", "AUTO", "LOITER", "STABILIZE"])
        self.mode_filter.currentTextChanged.connect(self._apply_filters)
        search_layout.addWidget(self.mode_filter)

        layout.addWidget(search_frame)

        # ── Table ──
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Date", "Duration", "Distance", "Max Alt", "Avg Speed",
            "Battery Used", "Mode", "GPS", "Status"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeToContents)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)

        # ── Export buttons ──
        btn_layout = QHBoxLayout()

        self.replay_btn = QPushButton("Replay Flight")
        self.replay_btn.setEnabled(False)
        self.replay_btn.setToolTip("Select a flight to replay telemetry")
        btn_layout.addWidget(self.replay_btn)

        btn_layout.addStretch()

        self.csv_btn = QPushButton("Export CSV")
        self.csv_btn.clicked.connect(self._export_csv)
        btn_layout.addWidget(self.csv_btn)

        self.video_btn = QPushButton("Export Video")
        self.video_btn.clicked.connect(self._export_video)
        btn_layout.addWidget(self.video_btn)

        layout.addLayout(btn_layout)

    def _load_sample_logs(self):
        samples = [
            {"date": "2026-04-05 14:30", "duration": "12:45", "distance": "1.2 km",
             "max_alt": "85 m", "avg_speed": "4.2 m/s", "batt_used": "62%",
             "mode": "AUTO", "gps": "12 sats", "status": "Completed"},
            {"date": "2026-04-04 10:15", "duration": "08:22", "distance": "0.8 km",
             "max_alt": "50 m", "avg_speed": "3.1 m/s", "batt_used": "45%",
             "mode": "GUIDED", "gps": "10 sats", "status": "Completed"},
            {"date": "2026-04-03 16:00", "duration": "05:10", "distance": "0.3 km",
             "max_alt": "30 m", "avg_speed": "1.8 m/s", "batt_used": "28%",
             "mode": "MANUAL", "gps": "8 sats", "status": "Aborted"},
            {"date": "2026-04-01 09:45", "duration": "15:30", "distance": "2.1 km",
             "max_alt": "120 m", "avg_speed": "5.6 m/s", "batt_used": "78%",
             "mode": "AUTO", "gps": "14 sats", "status": "Completed"},
            {"date": "2026-03-28 11:00", "duration": "20:15", "distance": "3.5 km",
             "max_alt": "100 m", "avg_speed": "6.2 m/s", "batt_used": "85%",
             "mode": "AUTO", "gps": "11 sats", "status": "Completed"},
        ]
        self.logs = samples
        self._apply_filters()
        self._update_stats()

    def _update_stats(self):
        total = len(self.logs)
        total_dist = 0.0
        total_secs = 0
        for log in self.logs:
            # Parse distance
            d = log.get("distance", "0 km").replace(" km", "")
            try:
                total_dist += float(d)
            except ValueError:
                pass
            # Parse duration mm:ss
            parts = log.get("duration", "0:00").split(":")
            if len(parts) == 2:
                try:
                    total_secs += int(parts[0]) * 60 + int(parts[1])
                except ValueError:
                    pass

        avg_secs = total_secs // total if total > 0 else 0

        def _update_card(name, val):
            card = self.stats_cards.get(name)
            if card:
                label = card.findChild(QLabel, "value")
                if label:
                    label.setText(str(val))

        _update_card("Total Flights", str(total))
        _update_card("Total Time", f"{total_secs // 60}:{total_secs % 60:02d}")
        _update_card("Total Distance", f"{total_dist:.1f} km")
        _update_card("Avg Duration", f"{avg_secs // 60}:{avg_secs % 60:02d}")

    def add_log(self, log_entry):
        self.logs.insert(0, log_entry)
        self._apply_filters()
        self._update_stats()

    def _apply_filters(self):
        search = self.search_input.text().lower()
        date_from = self.date_from.date().toPyDate()
        date_to = self.date_to.date().toPyDate()
        mode_f = self.mode_filter.currentText()

        filtered = []
        for log in self.logs:
            log_date = datetime.strptime(log["date"], "%Y-%m-%d %H:%M").date()
            if log_date < date_from or log_date > date_to:
                continue
            if mode_f != "All Modes" and log.get("mode") != mode_f:
                continue
            if search:
                searchable = " ".join(str(v) for v in log.values()).lower()
                if search not in searchable:
                    continue
            filtered.append(log)

        self.table.setRowCount(len(filtered))
        cols = ["date", "duration", "distance", "max_alt", "avg_speed", "batt_used", "mode", "gps", "status"]
        for i, log in enumerate(filtered):
            for j, key in enumerate(cols):
                val = log.get(key, "--")
                item = QTableWidgetItem(val)

                if key == "mode":
                    mode_colors = {
                        "MANUAL": "#ff8844", "GUIDED": "#6688ff", "AUTO": "#00cc00",
                        "LOITER": "#cccc00", "STABILIZE": "#cc88ff"
                    }
                    item.setForeground(QColor(mode_colors.get(val, "#ccc")))
                elif key == "status":
                    item.setForeground(QColor("#00cc00" if val == "Completed" else "#ff4444"))

                self.table.setItem(i, j, item)

        # Enable replay if row selected
        self.table.selectionModel().selectionChanged.connect(
            lambda: self.replay_btn.setEnabled(len(self.table.selectedItems()) > 0)
        )

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Flight Logs", "flight_logs.csv", "CSV Files (*.csv)")
        if path:
            cols = ["date", "duration", "distance", "max_alt", "avg_speed", "batt_used", "mode", "gps", "status"]
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                writer.writerows(self.logs)
            QMessageBox.information(self, "Export", f"Exported {len(self.logs)} flights to {path}")

    def _export_video(self):
        QMessageBox.information(self, "Export Video", "Video export will be available when flight recording is implemented.")


# ─── Tab: Settings ──────────────────────────────────────────────────

class SettingsTab(QWidget):
    """Drone profiles, protocol config, video streaming, AI server,
    geofence, RTH, alerts, camera settings, user management."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        # ── Drone Profile ──
        profile_group = QGroupBox("Drone Profile")
        profile_layout = QFormLayout(profile_group)

        self.profile_name = QLineEdit("Walle-1")
        profile_layout.addRow("Name:", self.profile_name)

        self.frame_type = QComboBox()
        self.frame_type.addItems(["Quad X", "Quad +", "Hex X", "Hex +", "Y6", "Octo X"])
        profile_layout.addRow("Frame:", self.frame_type)

        self.battery_cells = QSpinBox()
        self.battery_cells.setRange(1, 12)
        self.battery_cells.setValue(4)
        profile_layout.addRow("Battery Cells:", self.battery_cells)

        self.battery_capacity = QSpinBox()
        self.battery_capacity.setRange(100, 50000)
        self.battery_capacity.setValue(1500)
        self.battery_capacity.setSuffix(" mAh")
        profile_layout.addRow("Capacity:", self.battery_capacity)

        self.drone_id = QLineEdit("WALLE-001")
        self.drone_id.setPlaceholderText("Unique drone identifier")
        profile_layout.addRow("Drone ID:", self.drone_id)

        layout.addWidget(profile_group)

        # ── MAVLink Connection (UDP) ──
        proto_group = QGroupBox("MAVLink Connection (UDP)")
        proto_layout = QFormLayout(proto_group)

        self.mav_mode = QComboBox()
        self.mav_mode.addItems(["Listen (drone -> GCS)", "Connect (GCS -> drone)"])
        self.mav_mode.setToolTip(
            "Listen: GCS binds the UDP port and the drone/companion pushes "
            "its MAVLink stream to it (QGroundControl default; most SBC setups).\n"
            "Connect: GCS actively sends to the Drone IP (only for vehicles "
            "that act as a UDP server, e.g. some SITL setups).")
        proto_layout.addRow("Mode:", self.mav_mode)

        self.drone_ip = QLineEdit("127.0.0.1")
        self.drone_ip.setPlaceholderText("Drone / autopilot IP (Connect mode only)")
        proto_layout.addRow("Drone IP:", self.drone_ip)

        self.udp_port = QSpinBox()
        self.udp_port.setRange(1, 65535)
        self.udp_port.setValue(14550)
        proto_layout.addRow("UDP Port:", self.udp_port)

        # Drone IP is only used in Connect mode; grey it out in Listen mode.
        self.mav_mode.currentIndexChanged.connect(
            lambda i: self.drone_ip.setEnabled(i == 1))
        self.drone_ip.setEnabled(False)

        layout.addWidget(proto_group)

        # ── Video Streaming (RosettaDrone-inspired) ──
        video_group = QGroupBox("Video Streaming")
        video_layout = QFormLayout(video_group)

        self.video_stream_ip = QLineEdit("127.0.0.1")
        self.video_stream_ip.setPlaceholderText("IP address")
        video_layout.addRow("Stream IP:", self.video_stream_ip)

        self.video_stream_port = QSpinBox()
        self.video_stream_port.setRange(1, 65535)
        self.video_stream_port.setValue(5600)
        video_layout.addRow("Stream Port:", self.video_stream_port)

        self.video_bitrate = QComboBox()
        self.video_bitrate.addItems(["1000 kbps", "2000 kbps", "3000 kbps", "5000 kbps", "8000 kbps", "10000 kbps"])
        self.video_bitrate.setCurrentText("5000 kbps")
        video_layout.addRow("Bitrate:", self.video_bitrate)

        self.video_codec = QComboBox()
        self.video_codec.addItems(["H.264", "H.265", "MJPEG"])
        video_layout.addRow("Codec:", self.video_codec)

        self.video_aspect = QComboBox()
        self.video_aspect.addItems(["16:9 (DJI O4)", "4:3", "3:2", "21:9"])
        video_layout.addRow("Aspect Ratio:", self.video_aspect)

        self.video_resolution = QComboBox()
        self.video_resolution.addItems(["1920x1080", "1280x720", "3840x2160", "2560x1440"])
        video_layout.addRow("Resolution:", self.video_resolution)

        layout.addWidget(video_group)

        # ── WebRTC / STUN Configuration ──
        webrtc_group = QGroupBox("WebRTC Streaming")
        webrtc_layout = QFormLayout(webrtc_group)

        self.webrtc_enable = QCheckBox("Enable WebRTC streaming")
        webrtc_layout.addRow(self.webrtc_enable)

        self.stun_server = QLineEdit("stun:stun.l.google.com:19302")
        webrtc_layout.addRow("STUN Server:", self.stun_server)

        self.signaling_server = QLineEdit("ws://192.168.1.220:8090")
        webrtc_layout.addRow("Signaling Server:", self.signaling_server)

        layout.addWidget(webrtc_group)

        # ── AI Detection Server ──
        ai_group = QGroupBox("AI Detection Server")
        ai_layout = QFormLayout(ai_group)

        self.ai_enable = QCheckBox("Enable AI telemetry")
        ai_layout.addRow(self.ai_enable)

        self.ai_server_ip = QLineEdit("192.168.1.100")
        self.ai_server_ip.setPlaceholderText("AI server IP")
        ai_layout.addRow("Server IP:", self.ai_server_ip)

        self.ai_server_port = QSpinBox()
        self.ai_server_port.setRange(1, 65535)
        self.ai_server_port.setValue(8080)
        ai_layout.addRow("Server Port:", self.ai_server_port)

        self.ai_model = QComboBox()
        self.ai_model.addItems(["YOLOv8", "YOLOv5", "SSD MobileNet", "Custom"])
        ai_layout.addRow("Detection Model:", self.ai_model)

        self.ai_confidence = QSpinBox()
        self.ai_confidence.setRange(1, 100)
        self.ai_confidence.setValue(60)
        self.ai_confidence.setSuffix("%")
        ai_layout.addRow("Min Confidence:", self.ai_confidence)

        layout.addWidget(ai_group)

        # ── RTH Configuration ──
        rth_group = QGroupBox("Return-To-Home")
        rth_layout = QFormLayout(rth_group)

        self.rth_altitude = QSpinBox()
        self.rth_altitude.setRange(10, 500)
        self.rth_altitude.setValue(60)
        self.rth_altitude.setSuffix(" m")
        rth_layout.addRow("RTH Altitude:", self.rth_altitude)

        self.rth_max_height = QSpinBox()
        self.rth_max_height.setRange(10, 500)
        self.rth_max_height.setValue(120)
        self.rth_max_height.setSuffix(" m")
        rth_layout.addRow("Max Height:", self.rth_max_height)

        self.smart_rth = QCheckBox("Smart RTH (obstacle avoidance)")
        rth_layout.addRow(self.smart_rth)

        self.rth_land_mode = QComboBox()
        self.rth_land_mode.addItems(["Land at home", "Land at current position", "Hover"])
        rth_layout.addRow("RTH Action:", self.rth_land_mode)

        layout.addWidget(rth_group)

        # ── Camera Settings ──
        cam_group = QGroupBox("Camera Settings")
        cam_layout = QFormLayout(cam_group)

        self.cam_iso = QComboBox()
        self.cam_iso.addItems(["Auto", "100", "200", "400", "800", "1600", "3200"])
        cam_layout.addRow("ISO:", self.cam_iso)

        self.cam_shutter = QComboBox()
        self.cam_shutter.addItems(["Auto", "1/60", "1/120", "1/250", "1/500", "1/1000", "1/2000"])
        cam_layout.addRow("Shutter:", self.cam_shutter)

        self.cam_ev = QComboBox()
        self.cam_ev.addItems(["-2.0", "-1.5", "-1.0", "-0.5", "0", "+0.5", "+1.0", "+1.5", "+2.0"])
        self.cam_ev.setCurrentText("0")
        cam_layout.addRow("EV:", self.cam_ev)

        self.cam_wb = QComboBox()
        self.cam_wb.addItems(["Auto", "Sunny", "Cloudy", "Incandescent", "Fluorescent", "Custom"])
        cam_layout.addRow("White Balance:", self.cam_wb)

        layout.addWidget(cam_group)

        # ── Map Configuration ──
        map_group = QGroupBox("Map")
        map_layout = QFormLayout(map_group)

        self.map_type = QComboBox()
        self.map_type.addItems(["Normal", "Satellite", "Hybrid", "Terrain"])
        map_layout.addRow("Map Type:", self.map_type)

        self.map_follow_drone = QCheckBox("Follow drone position")
        self.map_follow_drone.setChecked(True)
        map_layout.addRow(self.map_follow_drone)

        layout.addWidget(map_group)

        # ── Geofence ──
        geo_group = QGroupBox("Geofence")
        geo_layout = QFormLayout(geo_group)

        self.geofence_enable = QCheckBox("Enable Geofence")
        geo_layout.addRow(self.geofence_enable)

        self.max_altitude = QSpinBox()
        self.max_altitude.setRange(10, 500)
        self.max_altitude.setValue(120)
        self.max_altitude.setSuffix(" m")
        geo_layout.addRow("Max Altitude:", self.max_altitude)

        self.max_distance = QSpinBox()
        self.max_distance.setRange(50, 10000)
        self.max_distance.setValue(500)
        self.max_distance.setSuffix(" m")
        geo_layout.addRow("Max Distance:", self.max_distance)

        self.geo_action = QComboBox()
        self.geo_action.addItems(["RTH", "Land", "Loiter", "None"])
        geo_layout.addRow("Breach Action:", self.geo_action)

        layout.addWidget(geo_group)

        # ── Alert Configuration ──
        alert_group = QGroupBox("Alert Configuration")
        alert_layout = QFormLayout(alert_group)

        self.low_batt_warn = QSpinBox()
        self.low_batt_warn.setRange(0, 100)
        self.low_batt_warn.setValue(30)
        self.low_batt_warn.setSuffix("%")
        alert_layout.addRow("Low Battery Warning:", self.low_batt_warn)

        self.low_batt_crit = QSpinBox()
        self.low_batt_crit.setRange(0, 100)
        self.low_batt_crit.setValue(15)
        self.low_batt_crit.setSuffix("%")
        alert_layout.addRow("Critical Battery:", self.low_batt_crit)

        self.rssi_warn = QSpinBox()
        self.rssi_warn.setRange(0, 130)
        self.rssi_warn.setValue(90)
        self.rssi_warn.setPrefix("-")
        self.rssi_warn.setSuffix(" dBm")
        alert_layout.addRow("RSSI Warning:", self.rssi_warn)

        self.lq_warn = QSpinBox()
        self.lq_warn.setRange(0, 100)
        self.lq_warn.setValue(50)
        self.lq_warn.setSuffix("%")
        alert_layout.addRow("LQ Warning:", self.lq_warn)

        layout.addWidget(alert_group)

        # ── User Management ──
        user_group = QGroupBox("User Management")
        user_layout = QVBoxLayout(user_group)
        user_layout.setSpacing(8)

        self.current_user_label = QLabel("Current user: --")
        self.current_user_label.setStyleSheet("color: #00cc00; font-weight: bold;")
        user_layout.addWidget(self.current_user_label)

        pw_layout = QFormLayout()
        self.new_password = QLineEdit()
        self.new_password.setEchoMode(QLineEdit.Password)
        self.new_password.setPlaceholderText("New password")
        pw_layout.addRow("Change Password:", self.new_password)

        self.confirm_password = QLineEdit()
        self.confirm_password.setEchoMode(QLineEdit.Password)
        self.confirm_password.setPlaceholderText("Confirm password")
        pw_layout.addRow("Confirm:", self.confirm_password)

        change_pw_btn = QPushButton("Update Password")
        change_pw_btn.clicked.connect(self._change_password)
        pw_layout.addRow(change_pw_btn)
        user_layout.addLayout(pw_layout)

        self.admin_section = QGroupBox("Add User (Admin Only)")
        admin_layout = QFormLayout(self.admin_section)

        self.new_username = QLineEdit()
        self.new_username.setPlaceholderText("Username")
        admin_layout.addRow("Username:", self.new_username)

        self.new_user_pw = QLineEdit()
        self.new_user_pw.setEchoMode(QLineEdit.Password)
        self.new_user_pw.setPlaceholderText("Password")
        admin_layout.addRow("Password:", self.new_user_pw)

        self.new_user_role = QComboBox()
        self.new_user_role.addItems(["pilot", "observer", "admin"])
        admin_layout.addRow("Role:", self.new_user_role)

        add_user_btn = QPushButton("Create User")
        add_user_btn.clicked.connect(self._add_user)
        admin_layout.addRow(add_user_btn)

        self.user_list = QListWidget()
        self.user_list.setMaximumHeight(100)
        self.user_list.setStyleSheet("QListWidget { background-color: #111; border: 1px solid #1e1e1e; border-radius: 6px; }")
        admin_layout.addRow("Users:", self.user_list)

        del_user_btn = QPushButton("Delete Selected User")
        del_user_btn.setStyleSheet("QPushButton { color: #ff6666; }")
        del_user_btn.clicked.connect(self._delete_user)
        admin_layout.addRow(del_user_btn)

        self.admin_section.setVisible(False)
        user_layout.addWidget(self.admin_section)

        layout.addWidget(user_group)

        layout.addStretch()

        # Save button
        save_btn = QPushButton("Save All Settings")
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #002200; color: #00ff00;
                font-weight: bold; font-size: 14px;
                border: 1px solid #00aa00; border-radius: 6px;
                padding: 12px;
            }
            QPushButton:hover { background-color: #003300; }
        """)
        save_btn.clicked.connect(self._save_settings)
        layout.addWidget(save_btn)

        scroll.setWidget(container)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def set_user(self, username, role):
        self._username = username
        self._role = role
        self.current_user_label.setText(f"Current user: {username} ({role})")
        self.admin_section.setVisible(role == "admin")
        self._refresh_user_list()

    def _change_password(self):
        pw = self.new_password.text()
        confirm = self.confirm_password.text()
        if not pw:
            QMessageBox.warning(self, "Error", "Password cannot be empty.")
            return
        if pw != confirm:
            QMessageBox.warning(self, "Error", "Passwords do not match.")
            return
        users = _load_users()
        username = getattr(self, "_username", "admin")
        if username in users:
            salt, hashed = _hash_password(pw)
            users[username]["hash"] = hashed
            users[username]["salt"] = salt
            _save_users(users)
            self.new_password.clear()
            self.confirm_password.clear()
            QMessageBox.information(self, "Success", "Password updated.")

    def _add_user(self):
        username = self.new_username.text().strip()
        password = self.new_user_pw.text()
        role = self.new_user_role.currentText()
        if not username or not password:
            QMessageBox.warning(self, "Error", "Username and password required.")
            return
        users = _load_users()
        if username in users:
            QMessageBox.warning(self, "Error", f"User '{username}' already exists.")
            return
        salt, hashed = _hash_password(password)
        users[username] = {"hash": hashed, "salt": salt, "role": role}
        _save_users(users)
        self.new_username.clear()
        self.new_user_pw.clear()
        self._refresh_user_list()
        QMessageBox.information(self, "Success", f"User '{username}' created with role '{role}'.")

    def _delete_user(self):
        item = self.user_list.currentItem()
        if not item:
            return
        target = item.text().split(" ")[0]
        current = getattr(self, "_username", "admin")
        if target == current:
            QMessageBox.warning(self, "Error", "Cannot delete your own account.")
            return
        reply = QMessageBox.question(self, "Confirm", f"Delete user '{target}'?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            users = _load_users()
            if target in users:
                del users[target]
                _save_users(users)
                self._refresh_user_list()

    def _refresh_user_list(self):
        self.user_list.clear()
        users = _load_users()
        for uname, info in users.items():
            self.user_list.addItem(f"{uname} ({info.get('role', 'unknown')})")

    def _save_settings(self):
        settings = {
            "profile": {
                "name": self.profile_name.text(),
                "frame": self.frame_type.currentText(),
                "battery_cells": self.battery_cells.value(),
                "battery_capacity": self.battery_capacity.value(),
                "drone_id": self.drone_id.text(),
            },
            "mavlink": {
                "mode": self.mav_mode.currentIndex(),
                "drone_ip": self.drone_ip.text(),
                "udp_port": self.udp_port.value(),
            },
            "video_streaming": {
                "ip": self.video_stream_ip.text(),
                "port": self.video_stream_port.value(),
                "bitrate": self.video_bitrate.currentText(),
                "codec": self.video_codec.currentText(),
                "aspect_ratio": self.video_aspect.currentText(),
                "resolution": self.video_resolution.currentText(),
            },
            "webrtc": {
                "enabled": self.webrtc_enable.isChecked(),
                "stun_server": self.stun_server.text(),
                "signaling_server": self.signaling_server.text(),
            },
            "ai_detection": {
                "enabled": self.ai_enable.isChecked(),
                "server_ip": self.ai_server_ip.text(),
                "server_port": self.ai_server_port.value(),
                "model": self.ai_model.currentText(),
                "min_confidence": self.ai_confidence.value(),
            },
            "rth": {
                "altitude": self.rth_altitude.value(),
                "max_height": self.rth_max_height.value(),
                "smart_rth": self.smart_rth.isChecked(),
                "land_mode": self.rth_land_mode.currentText(),
            },
            "camera": {
                "iso": self.cam_iso.currentText(),
                "shutter": self.cam_shutter.currentText(),
                "ev": self.cam_ev.currentText(),
                "white_balance": self.cam_wb.currentText(),
            },
            "map": {
                "type": self.map_type.currentText(),
                "follow_drone": self.map_follow_drone.isChecked(),
            },
            "geofence": {
                "enabled": self.geofence_enable.isChecked(),
                "max_alt": self.max_altitude.value(),
                "max_dist": self.max_distance.value(),
                "breach_action": self.geo_action.currentText(),
            },
            "alerts": {
                "low_batt_warn": self.low_batt_warn.value(),
                "low_batt_crit": self.low_batt_crit.value(),
                "rssi_warn": self.rssi_warn.value(),
                "lq_warn": self.lq_warn.value(),
            },
        }
        settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), EDITION.settings_file)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        QMessageBox.information(self, "Settings", "All settings saved successfully.")

    def load_settings(self):
        settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), EDITION.settings_file)
        if not os.path.exists(settings_path):
            return
        try:
            with open(settings_path) as f:
                s = json.load(f)

            p = s.get("profile", {})
            self.profile_name.setText(p.get("name", "Walle-1"))
            idx = self.frame_type.findText(p.get("frame", ""))
            if idx >= 0:
                self.frame_type.setCurrentIndex(idx)
            self.battery_cells.setValue(p.get("battery_cells", 4))
            self.battery_capacity.setValue(p.get("battery_capacity", 1500))
            self.drone_id.setText(p.get("drone_id", "WALLE-001"))

            ml = s.get("mavlink", {})
            self.mav_mode.setCurrentIndex(ml.get("mode", 0))
            self.drone_ip.setText(ml.get("drone_ip", "127.0.0.1"))
            self.udp_port.setValue(ml.get("udp_port", 14550))

            vs = s.get("video_streaming", {})
            self.video_stream_ip.setText(vs.get("ip", "127.0.0.1"))
            self.video_stream_port.setValue(vs.get("port", 5600))
            self.video_bitrate.setCurrentText(vs.get("bitrate", "5000 kbps"))
            self.video_codec.setCurrentText(vs.get("codec", "H.264"))
            self.video_aspect.setCurrentText(vs.get("aspect_ratio", "16:9 (DJI O4)"))
            self.video_resolution.setCurrentText(vs.get("resolution", "1920x1080"))

            wr = s.get("webrtc", {})
            self.webrtc_enable.setChecked(wr.get("enabled", False))
            self.stun_server.setText(wr.get("stun_server", "stun:stun.l.google.com:19302"))
            self.signaling_server.setText(wr.get("signaling_server", "ws://192.168.1.220:8090"))

            ai = s.get("ai_detection", {})
            self.ai_enable.setChecked(ai.get("enabled", False))
            self.ai_server_ip.setText(ai.get("server_ip", "192.168.1.100"))
            self.ai_server_port.setValue(ai.get("server_port", 8080))
            self.ai_model.setCurrentText(ai.get("model", "YOLOv8"))
            self.ai_confidence.setValue(ai.get("min_confidence", 60))

            rth = s.get("rth", {})
            self.rth_altitude.setValue(rth.get("altitude", 60))
            self.rth_max_height.setValue(rth.get("max_height", 120))
            self.smart_rth.setChecked(rth.get("smart_rth", False))
            self.rth_land_mode.setCurrentText(rth.get("land_mode", "Land at home"))

            cam = s.get("camera", {})
            self.cam_iso.setCurrentText(cam.get("iso", "Auto"))
            self.cam_shutter.setCurrentText(cam.get("shutter", "Auto"))
            self.cam_ev.setCurrentText(cam.get("ev", "0"))
            self.cam_wb.setCurrentText(cam.get("white_balance", "Auto"))

            mp = s.get("map", {})
            self.map_type.setCurrentText(mp.get("type", "Normal"))
            self.map_follow_drone.setChecked(mp.get("follow_drone", True))

            g = s.get("geofence", {})
            self.geofence_enable.setChecked(g.get("enabled", False))
            self.max_altitude.setValue(g.get("max_alt", 120))
            self.max_distance.setValue(g.get("max_dist", 500))
            idx = self.geo_action.findText(g.get("breach_action", "RTH"))
            if idx >= 0:
                self.geo_action.setCurrentIndex(idx)

            a = s.get("alerts", {})
            self.low_batt_warn.setValue(a.get("low_batt_warn", 30))
            self.low_batt_crit.setValue(a.get("low_batt_crit", 15))
            self.rssi_warn.setValue(a.get("rssi_warn", 90))
            self.lq_warn.setValue(a.get("lq_warn", 50))

        except Exception as e:
            log_settings.error("Failed to load settings: %s", e)


# ─── User Credentials Store ──────────────────────────────────────────

USERS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), EDITION.users_file)

def _hash_password(password, salt=None):
    if salt is None:
        salt = os.urandom(16).hex()
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return salt, hashed

def _load_users():
    if not os.path.exists(USERS_FILE):
        salt, hashed = _hash_password("admin")
        default = {
            "admin": {"hash": hashed, "salt": salt, "role": "admin"},
        }
        with open(USERS_FILE, "w") as f:
            json.dump(default, f, indent=2)
        return default
    with open(USERS_FILE) as f:
        return json.load(f)

def _save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def _verify_password(username, password):
    users = _load_users()
    if username not in users:
        return False, None
    entry = users[username]
    _, hashed = _hash_password(password, entry["salt"])
    if hashed == entry["hash"]:
        return True, entry["role"]
    return False, None


# ─── Login Dialog ────────────────────────────────────────────────────

class LoginDialog(QDialog):
    """Login gate shown before the main GCS window."""

    login_success = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Walle GCS - Login")
        self.setFixedSize(440, 520)
        self.setStyleSheet(DARK_STYLESHEET + """
            QDialog { background-color: #0a0a0a; }
        """)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(40, 30, 40, 30)

        # Logo / Title
        title = QLabel("WALLE GCS")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #00cc00; font-size: 32px; font-weight: bold; letter-spacing: 6px; padding: 10px;")
        layout.addWidget(title)

        subtitle = QLabel("Ground Control Station")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #444; font-size: 12px; padding-bottom: 10px;")
        layout.addWidget(subtitle)

        version_label = QLabel(f"v{APP_VERSION}")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("color: #333; font-size: 10px; padding-bottom: 10px;")
        layout.addWidget(version_label)

        # Login form
        form_group = QGroupBox("Sign In")
        form_group.setStyleSheet("""
            QGroupBox {
                color: #666; border: 1px solid #1e1e1e; border-radius: 10px;
                margin-top: 10px; padding: 24px 20px 20px 20px; font-size: 12px;
                background-color: #0d0d0d;
            }
            QGroupBox::title { subcontrol-position: top center; padding: 0 12px; }
        """)
        form_layout = QVBoxLayout(form_group)
        form_layout.setSpacing(12)

        user_label = QLabel("Username")
        user_label.setStyleSheet("color: #666; font-size: 11px; font-weight: bold;")
        form_layout.addWidget(user_label)

        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("Enter username")
        self.username_input.setStyleSheet("""
            QLineEdit {
                background-color: #111; border: 1px solid #2a2a2a; border-radius: 8px;
                padding: 12px 14px; font-size: 13px; color: #ccc;
            }
            QLineEdit:focus { border-color: #00aa00; }
        """)
        form_layout.addWidget(self.username_input)

        pass_label = QLabel("Password")
        pass_label.setStyleSheet("color: #666; font-size: 11px; font-weight: bold;")
        form_layout.addWidget(pass_label)

        self.password_input = QLineEdit()
        self.password_input.setPlaceholderText("Enter password")
        self.password_input.setEchoMode(QLineEdit.Password)
        self.password_input.setStyleSheet("""
            QLineEdit {
                background-color: #111; border: 1px solid #2a2a2a; border-radius: 8px;
                padding: 12px 14px; font-size: 13px; color: #ccc;
            }
            QLineEdit:focus { border-color: #00aa00; }
        """)
        self.password_input.returnPressed.connect(self._attempt_login)
        form_layout.addWidget(self.password_input)

        self.error_label = QLabel("")
        self.error_label.setAlignment(Qt.AlignCenter)
        self.error_label.setStyleSheet("color: #ff3333; font-size: 11px; min-height: 20px;")
        form_layout.addWidget(self.error_label)

        self.login_btn = QPushButton("LOGIN")
        self.login_btn.setStyleSheet("""
            QPushButton {
                background-color: #002200; color: #00ff00;
                font-size: 14px; font-weight: bold;
                border: 1px solid #00aa00; border-radius: 8px;
                padding: 14px; min-height: 20px;
            }
            QPushButton:hover { background-color: #003300; }
            QPushButton:pressed { background-color: #004400; }
        """)
        self.login_btn.clicked.connect(self._attempt_login)
        form_layout.addWidget(self.login_btn)

        layout.addWidget(form_group)

        hint = QLabel("Default: admin / admin")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #2a2a2a; font-size: 10px; padding-top: 8px;")
        layout.addWidget(hint)

        layout.addStretch()

    def _attempt_login(self):
        username = self.username_input.text().strip()
        password = self.password_input.text()

        if not username or not password:
            self.error_label.setText("Please enter username and password")
            return

        ok, role = _verify_password(username, password)
        if ok:
            self.login_success.emit(username, role)
            self.accept()
        else:
            self.error_label.setText("Invalid username or password")
            self.password_input.clear()
            self.password_input.setFocus()


# ─── Sidebar Navigation ─────────────────────────────────────────────

class SidebarNav(QFrame):
    """Icon-based sidebar navigation with labels."""
    page_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(72)
        self.setStyleSheet(SIDEBAR_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(0)

        # Logo
        logo = QLabel("W")
        logo.setAlignment(Qt.AlignCenter)
        logo.setStyleSheet(
            "color: #00cc00; font-size: 22px; font-weight: bold; "
            "padding: 10px; background: transparent;"
        )
        layout.addWidget(logo)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #1a1a1a;")
        layout.addWidget(sep)

        layout.addSpacing(8)

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        nav_items = [
            ("CTRL", 0),
            ("DETECT", 1),
            ("SLAM", 2),
            ("3D RECON", 3),
            ("LOGS", 4),
            ("CONFIG", 5),
        ]

        self.nav_buttons = []
        for label, idx in nav_items:
            btn = QToolButton()
            btn.setText(label)
            btn.setCheckable(True)
            btn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
            btn.setFixedSize(72, 54)
            btn.setStyleSheet(NAV_BTN_STYLE)
            self.btn_group.addButton(btn, idx)
            layout.addWidget(btn, alignment=Qt.AlignCenter)
            self.nav_buttons.append(btn)

        self.nav_buttons[0].setChecked(True)

        layout.addStretch()

        # Bottom status indicator
        self.status_dot = QLabel("")
        self.status_dot.setFixedSize(10, 10)
        self.status_dot.setStyleSheet("background-color: #333; border-radius: 5px;")
        layout.addWidget(self.status_dot, alignment=Qt.AlignCenter)
        layout.addSpacing(8)

        self.btn_group.idClicked.connect(self.page_changed.emit)

    def set_connected(self, connected):
        color = "#00cc00" if connected else "#333"
        self.status_dot.setStyleSheet(f"background-color: {color}; border-radius: 5px;")


# ─── Main Window ─────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self, username="admin", role="admin"):
        super().__init__()
        self.current_user = username
        self.current_role = role
        self.setWindowTitle(f"Walle GCS v{APP_VERSION}{EDITION.title_tag}  - {username}")
        self.setMinimumSize(1280, 720)
        self.setStyleSheet(DARK_STYLESHEET)

        # State
        self.mav = None  # MAVLinkConnection (UDP link to the vehicle)
        self.video_capture = None
        self.armed = False
        self.flight_mode = "MANUAL"
        self.last_telemetry = Telemetry()
        # arm state + flight mode come from MAVLink HEARTBEAT
        self.recording = False
        self.record_start_time = None
        # Object detection engine
        self.detect_engine = ObjectDetectionEngine(conf_thresh=0.5,
                                                   depth_variant=EDITION.depth_variant)
        self.detect_enabled = False
        self.depth_overlay_enabled = False
        # Reused buffer for camera|depth side-by-side composite so we don't
        # allocate a fresh ~7 MB numpy array per frame.
        self._stacked_buffer = None

        # Cache for the colorized depth panel + zone decision. The depth map
        # only changes at depth_target_fps (well below display FPS), so we
        # recolorize/relabel the panel only when the engine's depth version
        # advances instead of every displayed frame.
        self._depth_panel = None          # rendered depth_color (with zones/labels)
        self._depth_panel_key = None      # (depth_version, fh, fw)
        self._depth_panel_mode = "CLEAR"
        self._depth_panel_reason = ""

        self.face_rec_engine = FaceRecognitionEngine()
        self.face_rec_enabled = False

        self.track_engine = HumanTrackerEngine()
        self.track_enabled = False

        # SLAM engine + live-tap toggle (file-mode owns its own player).
        # Depth model variant follows the active edition (base=Small,
        # max=Base) — see the EDITIONS table.
        self.slam_engine = SlamEngine()
        self.slam_live_enabled = False

        self._build_ui()
        self._read_initial_state()
        self._start_timers()
        # Kick off model loads in the background so toggles find engines
        # already warm. Two important constraints:
        #   1) Defer past the first GUI paint — heavy CUDA work during
        #      window construction crashed the Wayland compositor on
        #      Pop!_OS 24.04 + NVIDIA in earlier testing.
        #   2) Chain sequentially (detect first, then face-rec). Running
        #      both loads in parallel doubled the GPU pressure spike and
        #      reproduced the same compositor crash.
        # Opt out by setting WALLE_NO_PRELOAD=1.
        if os.environ.get("WALLE_NO_PRELOAD", "0") != "1":
            QTimer.singleShot(3000, self._preload_detect_then_facerec)

        # Start the engine watchdog. Polls every 2 s and flags any engine
        # that hasn't heartbeat for 5 s — surfaces silent worker-thread
        # deaths the UI would otherwise miss.
        self._watchdog = EngineWatchdog(get_health_registry(), poll_s=2.0)
        self._watchdog.start()

    def _preload_detect_then_facerec(self):
        """Load detect engine first; chain face-rec onto its completion signal
        so we never have two heavy model loads running at the same time."""
        def _start_facerec(_success):
            try:
                self.face_rec_engine.models_loaded.disconnect(_start_facerec)
            except TypeError:
                pass
            try:
                self.detect_engine.models_loaded.disconnect(_start_facerec)
            except TypeError:
                pass
            try:
                self.face_rec_engine.load_models()
            except Exception as e:
                log_main.error("face-rec preload failed: %s", e)
        try:
            # QueuedConnection so the slot lands on the main thread regardless
            # of which thread emits — emit fires from the load thread but we
            # want all signal handling pinned to main.
            self.detect_engine.models_loaded.connect(
                _start_facerec, Qt.QueuedConnection)
            self.detect_engine.load_models()
            # If detect somehow already loaded (e.g. a prior toggle click),
            # trigger the face-rec chain manually.
            if self.detect_engine.is_ready:
                QTimer.singleShot(0, lambda: _start_facerec(True))
        except Exception as e:
            log_main.error("detect preload failed: %s", e)
            # Detect failed; still preload face-rec so at least one is warm.
            try:
                self.face_rec_engine.load_models()
            except Exception as e2:
                log_main.error("face-rec preload failed: %s", e2)

    def _build_ui(self):
        central = QWidget()
        central.setStyleSheet("background-color: #0a0a0a;")
        self.setCentralWidget(central)
        outer_layout = QHBoxLayout(central)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # ── Sidebar Navigation ──
        self.sidebar = SidebarNav()
        self.sidebar.page_changed.connect(self._on_nav_changed)
        outer_layout.addWidget(self.sidebar)

        # ── Main content area ──
        content = QWidget()
        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(8, 6, 8, 6)
        main_layout.setSpacing(6)

        # ── Top Connection Bar ──
        conn_frame = QFrame()
        conn_frame.setStyleSheet("QFrame { background-color: #0d0d0d; border-radius: 8px; }")
        conn_bar = QHBoxLayout(conn_frame)
        conn_bar.setContentsMargins(10, 6, 10, 6)
        conn_bar.setSpacing(8)

        conn_bar.addWidget(self._make_label("Video:", "#555"))
        self.video_combo = QComboBox()
        self.video_combo.setMinimumWidth(130)
        self._populate_video_devices()
        conn_bar.addWidget(self.video_combo)

        self.video_btn = QPushButton("Start")
        self.video_btn.setCheckable(True)
        self.video_btn.clicked.connect(self._toggle_video)
        conn_bar.addWidget(self.video_btn)

        self._add_separator(conn_bar)

        # MAVLink-over-UDP link to the vehicle. The drone IP/port come
        # from Settings; this button opens/closes the link.
        conn_bar.addWidget(self._make_label("Link:", "#555"))
        self.mav_btn = QPushButton("Connect")
        self.mav_btn.setCheckable(True)
        self.mav_btn.setToolTip("Connect / disconnect the MAVLink UDP link")
        self.mav_btn.clicked.connect(self._toggle_mav)
        conn_bar.addWidget(self.mav_btn)

        self._add_separator(conn_bar)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setToolTip("Rescan video capture devices")
        self.refresh_btn.clicked.connect(self._refresh_devices)
        conn_bar.addWidget(self.refresh_btn)

        conn_bar.addStretch()

        # ── High-visibility telemetry panel (top-right) ──
        # Boxed, bright, larger type so flight-critical numbers read at a glance.
        telem_frame = QFrame()
        telem_frame.setStyleSheet(
            "QFrame { background-color: #11161f; border: 1px solid #2c3a55; "
            "border-radius: 6px; }"
        )
        telem_bar = QHBoxLayout(telem_frame)
        telem_bar.setContentsMargins(12, 5, 12, 5)
        telem_bar.setSpacing(12)

        _TELEM_BIG = "font-weight: bold; font-size: 16px;"

        self.telem_voltage = QLabel("--V")
        self.telem_voltage.setStyleSheet(f"color: #b9d4ff; {_TELEM_BIG}")
        telem_bar.addWidget(self.telem_voltage)

        self.telem_lq = QLabel("LQ --%")
        self.telem_lq.setStyleSheet(f"color: #b9d4ff; {_TELEM_BIG}")
        telem_bar.addWidget(self.telem_lq)

        self.telem_rssi = QLabel("RSSI --")
        self.telem_rssi.setStyleSheet("color: #8fa6cc; font-weight: bold; font-size: 13px;")
        telem_bar.addWidget(self.telem_rssi)

        telem_sep = QLabel("|")
        telem_sep.setStyleSheet("color: #2c3a55; font-size: 18px;")
        telem_bar.addWidget(telem_sep)

        att_label = QLabel("ATT")
        att_label.setStyleSheet("color: #6b7da6; font-size: 11px; font-weight: bold;")
        telem_bar.addWidget(att_label)

        self.telem_pitch = QLabel("P --°")
        self.telem_pitch.setStyleSheet(f"color: #00d8d8; {_TELEM_BIG}")
        telem_bar.addWidget(self.telem_pitch)

        self.telem_roll = QLabel("R --°")
        self.telem_roll.setStyleSheet(f"color: #00d8d8; {_TELEM_BIG}")
        telem_bar.addWidget(self.telem_roll)

        self.telem_yaw = QLabel("Y --°")
        self.telem_yaw.setStyleSheet(f"color: #00d8d8; {_TELEM_BIG}")
        telem_bar.addWidget(self.telem_yaw)

        self.telem_heading = QLabel("HDG --°")
        self.telem_heading.setStyleSheet(f"color: #88aaff; {_TELEM_BIG}")
        telem_bar.addWidget(self.telem_heading)

        self.telem_alt = QLabel("ALT --m")
        self.telem_alt.setStyleSheet(f"color: #88ff88; {_TELEM_BIG}")
        telem_bar.addWidget(self.telem_alt)

        self.telem_speed = QLabel("SPD --")
        self.telem_speed.setStyleSheet(f"color: #ffaa44; {_TELEM_BIG}")
        telem_bar.addWidget(self.telem_speed)

        conn_bar.addWidget(telem_frame)

        main_layout.addWidget(conn_frame)

        # ── Main Content: Video + Stacked Panels ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet("QSplitter::handle { background-color: #1a1a1a; }")

        # Video feed (left)
        video_container = QWidget()
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(4)

        self.video_widget = VideoFeedWidget()
        video_layout.addWidget(self.video_widget, stretch=1)

        # Video controls bar below video
        vid_ctrl_frame = QFrame()
        vid_ctrl_frame.setStyleSheet("QFrame { background-color: #0d0d0d; border-radius: 6px; }")
        vid_ctrl_layout = QHBoxLayout(vid_ctrl_frame)
        vid_ctrl_layout.setContentsMargins(8, 4, 8, 4)
        vid_ctrl_layout.setSpacing(6)

        self.snapshot_btn = QPushButton("Snapshot")
        self.snapshot_btn.setFixedHeight(28)
        self.snapshot_btn.clicked.connect(self._take_snapshot)
        vid_ctrl_layout.addWidget(self.snapshot_btn)

        self.rec_btn = QPushButton("REC")
        self.rec_btn.setCheckable(True)
        self.rec_btn.setFixedHeight(28)
        self.rec_btn.setStyleSheet("""
            QPushButton { color: #888; }
            QPushButton:checked { color: #ff3333; font-weight: bold; border-color: #ff0000; }
        """)
        self.rec_btn.clicked.connect(self._toggle_recording)
        vid_ctrl_layout.addWidget(self.rec_btn)

        self.detect_btn = QPushButton("Detect")
        self.detect_btn.setCheckable(True)
        self.detect_btn.setFixedHeight(28)
        self.detect_btn.setStyleSheet("""
            QPushButton { color: #888; }
            QPushButton:checked { color: #00ccff; font-weight: bold; border-color: #0088ff; }
        """)
        self.detect_btn.clicked.connect(self._toggle_detection)
        vid_ctrl_layout.addWidget(self.detect_btn)

        self.depth_btn = QPushButton("Depth")
        self.depth_btn.setCheckable(True)
        self.depth_btn.setFixedHeight(28)
        self.depth_btn.setStyleSheet("""
            QPushButton { color: #888; }
            QPushButton:checked { color: #ff8800; font-weight: bold; border-color: #ff6600; }
        """)
        self.depth_btn.clicked.connect(self._toggle_depth_overlay)
        vid_ctrl_layout.addWidget(self.depth_btn)

        self.face_btn = QPushButton("Face Rec")
        self.face_btn.setCheckable(True)
        self.face_btn.setFixedHeight(28)
        self.face_btn.setStyleSheet("""
            QPushButton { color: #888; }
            QPushButton:checked { color: #ffd000; font-weight: bold; border-color: #ffaa00; }
        """)
        self.face_btn.clicked.connect(self._toggle_face_rec)
        vid_ctrl_layout.addWidget(self.face_btn)

        self.track_btn = QPushButton("Track")
        self.track_btn.setCheckable(True)
        self.track_btn.setFixedHeight(28)
        self.track_btn.setStyleSheet("""
            QPushButton { color: #888; }
            QPushButton:checked { color: #ff66cc; font-weight: bold; border-color: #ff33aa; }
        """)
        self.track_btn.clicked.connect(self._toggle_track_human)
        vid_ctrl_layout.addWidget(self.track_btn)

        vid_ctrl_layout.addStretch()

        self.hud_toggle = QCheckBox("HUD")
        self.hud_toggle.setChecked(True)
        self.hud_toggle.setStyleSheet("QCheckBox { color: #888; font-size: 11px; }")
        self.hud_toggle.stateChanged.connect(lambda s: setattr(self.video_widget, 'hud_visible', s == Qt.Checked))
        vid_ctrl_layout.addWidget(self.hud_toggle)

        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(["16:9", "4:3", "3:2", "21:9"])
        self.aspect_combo.setFixedWidth(60)
        self.aspect_combo.currentTextChanged.connect(self._change_aspect)
        vid_ctrl_layout.addWidget(self.aspect_combo)

        video_layout.addWidget(vid_ctrl_frame)

        splitter.addWidget(video_container)

        # Right panel: stacked widget driven by sidebar
        self.stack = QStackedWidget()

        self.control_tab = ControlPanelTab()
        self.control_tab.flight_mode_changed.connect(self._on_flight_mode_changed)
        self.control_tab.arm_requested.connect(self._on_arm_requested)
        # Flight command buttons -> MAVLink commands
        self.control_tab.takeoff_btn.clicked.connect(self._on_takeoff)
        self.control_tab.land_btn.clicked.connect(self._on_land)
        self.control_tab.rth_btn.clicked.connect(self._on_rth)
        self.control_tab.estop_btn.clicked.connect(self._on_emergency_stop)
        self.stack.addWidget(self.control_tab)

        self.detection_tab = DetectionFeedTab()
        self.stack.addWidget(self.detection_tab)

        self.slam_tab = SlamTab(self.slam_engine, default_variant="base")
        self.slam_tab.live_mode_changed.connect(self._on_slam_live_changed)
        self.stack.addWidget(self.slam_tab)

        # 3D Reconstruction (LingBot-Map wrapper) — offline file in, .ply + viser out
        self.recon_tab = ReconstructionTab()
        self.stack.addWidget(self.recon_tab)

        self.logs_tab = FlightLogsTab()
        self.stack.addWidget(self.logs_tab)

        self.settings_tab = SettingsTab()
        self.settings_tab.load_settings()
        self.settings_tab.set_user(self.current_user, self.current_role)
        self.stack.addWidget(self.settings_tab)

        splitter.addWidget(self.stack)
        # Video feed is the primary surface — give it the major share and a
        # heavier stretch factor so it stays dominant on resize. The side
        # panel keeps a usable minimum width.
        self.stack.setMinimumWidth(340)
        splitter.setSizes([1180, 360])
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)

        main_layout.addWidget(splitter, stretch=1)

        outer_layout.addWidget(content, stretch=1)

        # ── Notification Toast ──
        self.toast = NotificationToast(self)
        self.toast.setGeometry(0, 0, 400, 40)

        # ── Status Bar ──
        sb = self.statusBar()
        sb.setStyleSheet("QStatusBar { background-color: #080808; border-top: 1px solid #1a1a1a; }")

        self.status_arm = QLabel("DISARMED")
        self.status_arm.setStyleSheet(
            "color: #00ff00; font-weight: bold; padding: 2px 12px; "
            "background-color: #001a00; border: 1px solid #003300; border-radius: 4px;"
        )
        sb.addWidget(self.status_arm)

        self.status_mode = QLabel("MANUAL")
        self.status_mode.setStyleSheet("color: #888; font-weight: bold; padding: 2px 8px;")
        sb.addWidget(self.status_mode)

        sb.addWidget(self._make_separator_widget())

        self.status_video = QLabel("Video: OFF")
        self.status_video.setStyleSheet("color: #555; padding: 2px 8px;")
        sb.addWidget(self.status_video)

        self.status_serial = QLabel("Link: OFF")
        self.status_serial.setStyleSheet("color: #555; padding: 2px 8px;")
        sb.addWidget(self.status_serial)

        sb.addWidget(self._make_separator_widget())

        self.status_gps = QLabel("GPS: --")
        self.status_gps.setStyleSheet("color: #555; padding: 2px 8px;")
        sb.addWidget(self.status_gps)

        self.status_recording = QLabel("")
        self.status_recording.setStyleSheet("color: #ff3333; font-weight: bold; padding: 2px 8px;")
        sb.addWidget(self.status_recording)

        sb.addWidget(self._make_separator_widget())

        self.status_user = QLabel(f"{self.current_user} ({self.current_role})")
        self.status_user.setStyleSheet("color: #00aa00; padding: 2px 8px; font-weight: bold;")
        sb.addPermanentWidget(self.status_user)

        sb.showMessage("Ready - Connect RadioMaster via USB and HDMI capture card")

    def _on_nav_changed(self, idx):
        self.stack.setCurrentIndex(idx)

    def _on_slam_live_changed(self, enabled):
        """SlamTab requests the live HDMI tap on/off."""
        self.slam_live_enabled = bool(enabled)

    def _make_label(self, text, color):
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {color}; font-size: 11px;")
        return lbl

    def _add_separator(self, layout):
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedWidth(1)
        sep.setStyleSheet("background-color: #222;")
        layout.addWidget(sep)

    def _make_separator_widget(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setFixedWidth(1)
        sep.setStyleSheet("background-color: #222;")
        return sep

    def _populate_video_devices(self):
        self.video_combo.clear()
        if sys.platform == "win32":
            # Probe MSMF first (handles HDMI capture cards), fall back to DSHOW
            for i in range(10):
                cap = cv2.VideoCapture(i, cv2.CAP_MSMF)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                if cap.isOpened():
                    self.video_combo.addItem(f"Camera {i}", i)
                    cap.release()
        elif sys.platform == "darwin":
            # macOS has no /dev/video* nodes — probe AVFoundation directly.
            for i in range(10):
                cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                if cap.isOpened():
                    self.video_combo.addItem(f"Camera {i}", i)
                    cap.release()
        else:
            for i in range(10):
                if os.path.exists(f"/dev/video{i}"):
                    self.video_combo.addItem(f"/dev/video{i}", i)

    def _refresh_devices(self):
        self._populate_video_devices()
        self.toast.show_message("Video devices refreshed", "#00cc00", 2000)

    def _toggle_video(self, checked):
        if checked:
            idx = self.video_combo.currentData()
            if idx is not None:
                _vcfg = get_config().video
                self.video_capture = VideoCapture(
                    idx,
                    max_width=_vcfg.capture_max_width,
                    max_height=_vcfg.capture_max_height)
                self.video_capture.frame_ready.connect(self._update_video)
                self.video_capture.start()
                self.video_btn.setText("Stop")
                self.status_video.setText("Video: ON")
                self.status_video.setStyleSheet("color: #00aa00; padding: 2px 8px;")
                self.toast.show_message("Video feed started", "#00cc00", 2000)
            else:
                self.video_btn.setChecked(False)
        else:
            if self.video_capture:
                self.video_capture.stop()
                self.video_capture = None
            self.video_btn.setText("Start")
            self.video_widget.setText("NO VIDEO FEED")
            self.video_widget.setPixmap(QPixmap())
            self.status_video.setText("Video: OFF")
            self.status_video.setStyleSheet("color: #555; padding: 2px 8px;")

    def _toggle_mav(self, checked):
        if checked:
            host = self.settings_tab.drone_ip.text().strip() or "127.0.0.1"
            port = self.settings_tab.udp_port.value()
            mode = "connect" if self.settings_tab.mav_mode.currentIndex() == 1 else "listen"
            self.mav = MAVLinkConnection(host, port, mode=mode)
            self.mav.telemetry_updated.connect(self._update_telemetry)
            self.mav.link_changed.connect(self._on_link_changed)
            self.mav.start()
            self.mav_btn.setText("Disconnect")
            where = f"{host}:{port}" if mode == "connect" else f":{port} (listening)"
            self.status_serial.setText(f"Link: {where} (waiting)")
            self.status_serial.setStyleSheet("color: #ffcc00; padding: 2px 8px;")
            self.toast.show_message(f"MAVLink {mode} {where}", "#00cc00", 2000)
        else:
            if self.mav:
                self.mav.stop()
                self.mav = None
            self.mav_btn.setText("Connect")
            self.status_serial.setText("Link: OFF")
            self.status_serial.setStyleSheet("color: #555; padding: 2px 8px;")
            self.sidebar.set_connected(False)
            self.toast.show_message("MAVLink link closed", "#ffcc00", 2000)

    def _on_link_changed(self, up):
        """Heartbeat-based link status from the MAVLink thread."""
        self.sidebar.set_connected(up)
        if up:
            host = self.settings_tab.drone_ip.text().strip() or "127.0.0.1"
            self.status_serial.setText(f"Link: {host} (heartbeat)")
            self.status_serial.setStyleSheet("color: #00aa00; padding: 2px 8px;")
            self.toast.show_message("Vehicle heartbeat received", "#00cc00", 2000)
        else:
            self.status_serial.setText("Link: no heartbeat")
            self.status_serial.setStyleSheet("color: #ff4444; padding: 2px 8px;")
            self.toast.show_message("Link lost (no heartbeat)", "#ff4444", 3000)

    def _on_takeoff(self):
        if self.mav:
            self.mav.takeoff(self.settings_tab.rth_altitude.value())
            self.toast.show_message("Takeoff command sent", "#5588ee", 2000)
        else:
            self.toast.show_message("Not connected", "#ff8844", 2000)

    def _on_land(self):
        if self.mav:
            self.mav.land()
            self.toast.show_message("Land command sent", "#5588ee", 2000)
        else:
            self.toast.show_message("Not connected", "#ff8844", 2000)

    def _on_rth(self):
        if self.mav:
            self.mav.return_to_launch()
            self.toast.show_message("RTL command sent", "#ffcc00", 2000)
        else:
            self.toast.show_message("Not connected", "#ff8844", 2000)

    def _on_emergency_stop(self):
        if self.mav:
            self.mav.emergency_stop()
            self.toast.show_message("EMERGENCY STOP - motors killed!", "#ff3333", 4000)
        else:
            self.toast.show_message("Not connected", "#ff8844", 2000)

    def _update_video(self, frame):
        # Feed SLAM with the clean, un-annotated frame *before* any
        # in-place processing (detection boxes, depth side-by-side
        # hstack, etc) so intrinsics + grayscale features stay valid.
        if self.slam_live_enabled and self.slam_engine.is_running:
            self.slam_engine.process_frame(frame.copy())

        if self.track_enabled and self.track_engine.is_ready:
            frame = self.track_engine.process_frame(frame)

        if self.face_rec_enabled and self.face_rec_engine.is_ready:
            frame = self.face_rec_engine.process_frame(frame)

        if self.detect_enabled and self.detect_engine.is_ready:
            alt = self.last_telemetry.altitude if hasattr(self.last_telemetry, 'altitude') else 0.0
            pitch = self.last_telemetry.pitch if hasattr(self.last_telemetry, 'pitch') else None
            frame, det_list = self.detect_engine.process_frame(frame, altitude=alt, pitch_deg=pitch)

            # Feed detection events to DetectionFeedTab
            if det_list and hasattr(self, 'detection_tab'):
                for det in det_list:
                    if det.get("id") is None:
                        continue   # tracker hasn't confirmed this object yet
                    evt_type = "Obstacle" if det["is_danger"] else "Object"
                    dist_str = f"{det['distance_m']:.2f}m" if det["distance_m"] else ""
                    severity = "High" if det["is_danger"] else "Low"
                    self.detection_tab.add_event(
                        evt_type, det["class"], round(det["conf"] * 100),
                        details=dist_str, severity=severity,
                        obj_id=det["id"], count=det.get("count", 1))

        # Depth side-by-side: original left, colorized depth right
        if self.depth_overlay_enabled and self.detect_engine.is_ready:
            fh, fw = frame.shape[:2]
            cx_lo = int(fw * (0.5 - CENTER_ZONE_FRAC / 2))
            cx_hi = int(fw * (0.5 + CENTER_ZONE_FRAC / 2))

            # The colorized + labelled depth panel only depends on the depth
            # map (which updates at depth_target_fps) and the frame size, so
            # rebuild it only when either changes. Recolorizing identical
            # depth data every displayed frame is pure wasted GUI-thread work.
            version = self.detect_engine.get_depth_version()
            key = (version, fh, fw)
            if key != self._depth_panel_key:
                depth_raw = self.detect_engine.get_depth_map()
                if depth_raw is not None:
                    depth_norm = _normalize_depth(depth_raw)
                    depth_color = _colorize_depth(depth_norm)
                    if depth_color.shape[:2] != (fh, fw):
                        depth_color = cv2.resize(depth_color, (fw, fh))

                    # Zone avoidance decision
                    mode, reason, zl, zc, zr = _depth_decision(depth_norm)

                    # Draw zones on depth panel
                    cv2.line(depth_color, (cx_lo, 0), (cx_lo, fh), (255, 255, 255), 1)
                    cv2.line(depth_color, (cx_hi, 0), (cx_hi, fh), (255, 255, 255), 1)

                    # Zone labels on depth panel
                    cv2.putText(depth_color, f"L:{zl:.0f}", (10, fh - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(depth_color, f"C:{zc:.0f}", (cx_lo + 5, fh - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(depth_color, f"R:{zr:.0f}", (cx_hi + 5, fh - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    self._depth_panel = depth_color
                    self._depth_panel_mode = mode
                    self._depth_panel_reason = reason
                    self._depth_panel_key = key

            if self._depth_panel is not None and self._depth_panel.shape[:2] == (fh, fw):
                mode = self._depth_panel_mode
                reason = self._depth_panel_reason

                # Draw zone lines on camera feed too (live frame, every frame)
                cv2.line(frame, (cx_lo, 0), (cx_lo, fh), (255, 255, 0), 1)
                cv2.line(frame, (cx_hi, 0), (cx_hi, fh), (255, 255, 0), 1)

                # Decision banner on camera feed
                mode_colors = {
                    "CLEAR":   (0, 255, 0),
                    "AVOID_L": (0, 255, 255),
                    "AVOID_R": (0, 255, 255),
                    "STOP":    (0, 0, 255),
                }
                mc = mode_colors.get(mode, (255, 255, 255))
                banner = f"DEPTH: {mode}  {reason}"
                cv2.putText(frame, banner, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, banner, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, mc, 2, cv2.LINE_AA)

                # Stack side by side: camera | depth, reusing a preallocated
                # buffer to avoid a fresh allocation+copy every frame.
                need_shape = (fh, fw * 2, 3)
                if (self._stacked_buffer is None
                        or self._stacked_buffer.shape != need_shape):
                    self._stacked_buffer = np.empty(need_shape, dtype=np.uint8)
                self._stacked_buffer[:, :fw] = frame
                self._stacked_buffer[:, fw:] = self._depth_panel
                frame = self._stacked_buffer

                self.video_widget.hud_depth_mode = mode

        self.video_widget.display_frame(frame)

        # Backpressure: tell the capture thread we're ready for the next
        # frame. Until this fires, the capture loop drops frames instead of
        # queuing them, so the GUI event loop never backs up.
        if self.video_capture is not None:
            self.video_capture.frame_consumed()

    def _update_telemetry(self, telem):
        self.last_telemetry = telem

        v = telem.battery_voltage
        if v > 0:
            v_color = "#00ff00" if v > 14.0 else "#ffff00" if v > 13.2 else "#ff3333"
            self.telem_voltage.setText(f"{v:.1f}V")
            self.telem_voltage.setStyleSheet(f"color: {v_color}; font-weight: bold; font-size: 16px;")

        lq = telem.link_quality
        lq_color = "#00ff00" if lq > 70 else "#ffff00" if lq > 30 else "#ff3333"
        self.telem_lq.setText(f"LQ {lq}%")
        self.telem_lq.setStyleSheet(f"color: {lq_color}; font-weight: bold; font-size: 16px;")

        self.telem_rssi.setText(f"RSSI -{telem.rssi_1}dBm")

        # Attitude telemetry
        self.telem_pitch.setText(f"P {telem.pitch:+.1f}°")
        self.telem_pitch.setStyleSheet("color: #00d8d8; font-weight: bold; font-size: 16px;")
        self.telem_roll.setText(f"R {telem.roll:+.1f}°")
        self.telem_roll.setStyleSheet("color: #00d8d8; font-weight: bold; font-size: 16px;")
        self.telem_yaw.setText(f"Y {telem.yaw:+.1f}°")
        self.telem_yaw.setStyleSheet("color: #00d8d8; font-weight: bold; font-size: 16px;")
        if telem.heading > 0:
            self.telem_heading.setText(f"HDG {telem.heading:.0f}°")
            self.telem_heading.setStyleSheet("color: #88aaff; font-weight: bold; font-size: 16px;")
        if telem.altitude != 0:
            self.telem_alt.setText(f"ALT {telem.altitude:.0f}m")
            self.telem_alt.setStyleSheet("color: #88ff88; font-weight: bold; font-size: 16px;")
        if telem.ground_speed > 0:
            self.telem_speed.setText(f"SPD {telem.ground_speed:.1f}m/s")
            self.telem_speed.setStyleSheet("color: #ffaa44; font-weight: bold; font-size: 16px;")

        self.video_widget.update_hud(
            telem.battery_voltage, telem.battery_current, telem.battery_percent,
            telem.rssi_1, telem.link_quality, telem.snr,
            self.armed, self.flight_mode
        )

        # Feed GPS/attitude to HUD
        self.video_widget.hud_heading = telem.heading
        self.video_widget.hud_altitude = telem.altitude
        self.video_widget.hud_speed = telem.ground_speed
        self.video_widget.hud_pitch = telem.pitch
        self.video_widget.hud_roll = telem.roll
        self.video_widget.hud_satellites = telem.satellites

        # Update GPS status bar
        if telem.satellites > 0:
            self.status_gps.setText(f"GPS: {telem.satellites} sats")
            gps_color = "#00aa00" if telem.satellites >= 6 else "#ffcc00" if telem.satellites >= 3 else "#ff3333"
            self.status_gps.setStyleSheet(f"color: {gps_color}; padding: 2px 8px;")

        # Flight mode string from the MAVLink HEARTBEAT.
        if telem.flight_mode:
            self.flight_mode = telem.flight_mode
            self.status_mode.setText(telem.flight_mode)
            self.video_widget.hud_flight_mode = telem.flight_mode

        # Arm state is authoritative from the vehicle's HEARTBEAT.
        if telem.armed != self.armed:
            self.armed = telem.armed
            self.control_tab.set_armed(telem.armed)
            self.video_widget.hud_armed = telem.armed
            if telem.armed:
                self.status_arm.setText("ARMED")
                self.status_arm.setStyleSheet(
                    "color: #ff0000; font-weight: bold; padding: 2px 12px; "
                    "background-color: #1a0000; border: 1px solid #330000; border-radius: 4px;"
                )
                self.toast.show_message("ARMED - Motors active!", "#ff3333", 3000)
            else:
                self.status_arm.setText("DISARMED")
                self.status_arm.setStyleSheet(
                    "color: #00ff00; font-weight: bold; padding: 2px 12px; "
                    "background-color: #001a00; border: 1px solid #003300; border-radius: 4px;"
                )
                self.toast.show_message("Disarmed", "#00cc00", 2000)

        self.control_tab.update_telemetry(telem)

        # Alert checks
        settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), EDITION.settings_file)
        try:
            if os.path.exists(settings_path):
                with open(settings_path) as f:
                    alerts = json.load(f).get("alerts", {})
                if telem.battery_percent > 0 and telem.battery_percent <= alerts.get("low_batt_crit", 15):
                    self.toast.show_message("CRITICAL: Battery level critical!", "#ff3333", 5000)
                elif telem.battery_percent > 0 and telem.battery_percent <= alerts.get("low_batt_warn", 30):
                    self.toast.show_message("WARNING: Low battery", "#ffcc00", 3000)
                if telem.link_quality > 0 and telem.link_quality <= alerts.get("lq_warn", 50):
                    self.toast.show_message("WARNING: Low link quality", "#ffcc00", 3000)
        except (json.JSONDecodeError, OSError):
            pass

    def _on_arm_requested(self, armed):
        # Send the arm/disarm command; the real arm state is reflected
        # back via HEARTBEAT in _update_telemetry, which updates the
        # status indicators. We only message the user here.
        if not self.mav:
            self.control_tab.set_armed(self.armed)  # revert button toggle
            self.toast.show_message("Not connected", "#ff8844", 2000)
            return
        self.mav.arm(armed)
        self.toast.show_message(
            "Arm command sent" if armed else "Disarm command sent",
            "#5588ee", 2000)

    def _on_flight_mode_changed(self, mode):
        self.flight_mode = mode
        self.status_mode.setText(mode)
        self.video_widget.hud_flight_mode = mode
        if self.mav:
            # "RTH" in the UI maps to the autopilot's RTL command.
            if mode.upper() in ("RTH", "RTL"):
                self.mav.return_to_launch()
            else:
                self.mav.set_mode(mode)

    def _take_snapshot(self):
        if self.video_widget.pixmap() and not self.video_widget.pixmap().isNull():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(os.path.expanduser("~"), f"walle_snapshot_{timestamp}.png")
            self.video_widget.pixmap().save(path)
            self.toast.show_message(f"Snapshot saved: {path}", "#00cc00", 3000)
        else:
            self.toast.show_message("No video frame to capture", "#ff8844", 2000)

    def _toggle_recording(self, checked):
        self.recording = checked
        self.video_widget.hud_recording = checked
        if checked:
            self.record_start_time = time.time()
            self.status_recording.setText("REC")
            self.toast.show_message("Recording started", "#ff3333", 2000)
        else:
            self.record_start_time = None
            self.video_widget.hud_record_time = 0
            self.status_recording.setText("")
            self.toast.show_message("Recording stopped", "#00cc00", 2000)

    def _await_engine(self, engine, on_done, interval_ms=250):
        """Poll ``engine`` until it reaches a terminal state, then fire on_done.

        Replaces the old ``models_loaded`` signal+disconnect dance, which had
        two failure modes: (1) the emit could race against connect/disconnect
        and silently drop the slot, and (2) ``disconnect()`` on the engine
        nuked the preload chain that wires face-rec onto detect's completion.
        Polling on the GUI thread is racefree and stays main-thread by
        construction, so the callback can touch widgets safely.
        """
        if engine.is_ready:
            on_done(True)
            return
        if not getattr(engine, "_loading", False):
            # Not ready and not loading anymore → load failed (or never started).
            on_done(False)
            return
        QTimer.singleShot(interval_ms,
                          lambda: self._await_engine(engine, on_done, interval_ms))

    def _toggle_detection(self, checked):
        if checked:
            if self.detect_engine.is_ready:
                self.detect_enabled = True
                self.detect_btn.setText("Detect ON")
                self.toast.show_message("Detection enabled", "#00ccff", 2000)
                return
            self.detect_btn.setText("Loading...")
            self.detect_btn.setEnabled(False)
            self.toast.show_message("Loading detection models...", "#00ccff", 5000)

            def on_loaded(success):
                self.detect_btn.setEnabled(True)
                if success:
                    self.detect_enabled = True
                    self.detect_btn.setText("Detect ON")
                    self.toast.show_message("Detection enabled - RF-DETR + Depth", "#00ccff", 3000)
                else:
                    self.detect_enabled = False
                    self.detect_btn.setChecked(False)
                    self.detect_btn.setText("Detect")
                    self.toast.show_message("Detection model load failed!", "#ff3333", 3000)

            self.detect_engine.load_models()  # no-op if already loading/loaded
            self._await_engine(self.detect_engine, on_loaded)
        else:
            self.detect_enabled = False
            self.detect_btn.setText("Detect")
            self.toast.show_message("Detection disabled", "#888888", 2000)

    def _toggle_depth_overlay(self, checked):
        if checked:
            if self.detect_engine.is_ready:
                self.depth_overlay_enabled = True
                self.depth_btn.setText("Depth ON")
                self.toast.show_message("Depth overlay enabled", "#ff8800", 2000)
                return
            self.depth_btn.setText("Loading...")
            self.depth_btn.setEnabled(False)
            self.toast.show_message("Loading depth model...", "#ff8800", 5000)

            def on_loaded(success):
                self.depth_btn.setEnabled(True)
                if success:
                    self.depth_overlay_enabled = True
                    self.depth_btn.setText("Depth ON")
                    self.toast.show_message("Depth overlay enabled", "#ff8800", 3000)
                else:
                    self.depth_overlay_enabled = False
                    self.depth_btn.setChecked(False)
                    self.depth_btn.setText("Depth")
                    self.toast.show_message("Depth model load failed!", "#ff3333", 3000)

            self.detect_engine.load_models()
            self._await_engine(self.detect_engine, on_loaded)
        else:
            self.depth_overlay_enabled = False
            self.depth_btn.setText("Depth")
            self.video_widget.hud_depth_mode = None
            self.toast.show_message("Depth overlay disabled", "#888888", 2000)

    def _toggle_face_rec(self, checked):
        if checked:
            if self.face_rec_engine.is_ready:
                self.face_rec_enabled = True
                self.face_btn.setText("Face Rec ON")
                self.toast.show_message("Face recognition enabled",
                                        "#ffd000", 2000)
                return
            self.face_btn.setText("Loading...")
            self.face_btn.setEnabled(False)
            self.toast.show_message(
                "Loading face recognition models...", "#ffd000", 6000)

            def on_loaded(success):
                self.face_btn.setEnabled(True)
                if success:
                    self.face_rec_enabled = True
                    n = self.face_rec_engine.known_count
                    self.face_btn.setText("Face Rec ON")
                    self.toast.show_message(
                        f"Face recognition enabled — {n} known faces",
                        "#ffd000", 3000)
                else:
                    self.face_rec_enabled = False
                    self.face_btn.setChecked(False)
                    self.face_btn.setText("Face Rec")
                    self.toast.show_message(
                        "Face Rec load failed — is onnxruntime installed?",
                        "#ff3333", 4000)

            self.face_rec_engine.load_models()
            self._await_engine(self.face_rec_engine, on_loaded)
        else:
            self.face_rec_enabled = False
            self.face_btn.setText("Face Rec")
            self.toast.show_message("Face recognition disabled",
                                    "#888888", 2000)

    def _toggle_track_human(self, checked):
        if checked:
            if self.track_engine.is_ready:
                self.track_enabled = True
                self.track_btn.setText("Track ON")
                self.toast.show_message("Human tracking enabled",
                                        "#ff66cc", 2000)
                return
            self.track_btn.setText("Loading...")
            self.track_btn.setEnabled(False)
            self.toast.show_message(
                "Loading human tracker (YOLOv8-pose + DeepSort)...",
                "#ff66cc", 6000)

            def on_loaded(success):
                self.track_btn.setEnabled(True)
                if success:
                    self.track_enabled = True
                    self.track_btn.setText("Track ON")
                    self.toast.show_message(
                        "Human tracking enabled — predictive arcs live",
                        "#ff66cc", 3000)
                else:
                    self.track_enabled = False
                    self.track_btn.setChecked(False)
                    self.track_btn.setText("Track")
                    self.toast.show_message(
                        "Human tracker load failed (see terminal)",
                        "#ff3333", 4000)

            self.track_engine.load_models()
            self._await_engine(self.track_engine, on_loaded)
        else:
            self.track_enabled = False
            self.track_btn.setText("Track")
            self.toast.show_message("Human tracking disabled",
                                    "#888888", 2000)

    def _change_aspect(self, text):
        ratios = {"16:9": (16, 9), "4:3": (4, 3), "3:2": (3, 2), "21:9": (21, 9)}
        w, h = ratios.get(text, (16, 9))
        self.video_widget.set_aspect_ratio(w, h)

    def _read_initial_state(self):
        # Arm state comes from the vehicle's MAVLink HEARTBEAT once
        # connected; start in the known disarmed state.
        self.control_tab.set_armed(self.armed)
        self.video_widget.hud_armed = self.armed

    def _start_timers(self):
        # Recording timer
        self.rec_timer = QTimer()
        self.rec_timer.timeout.connect(self._update_recording_time)
        self.rec_timer.start(1000)

    def _update_recording_time(self):
        if self.recording and self.record_start_time:
            elapsed = int(time.time() - self.record_start_time)
            self.video_widget.hud_record_time = elapsed

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Reposition toast at top-center
        if hasattr(self, 'toast'):
            tw = 400
            self.toast.setGeometry((self.width() - tw) // 2, 60, tw, 40)

    def closeEvent(self, event):
        self.rec_timer.stop()
        if self.video_capture:
            self.video_capture.stop()
        if self.mav:
            self.mav.stop()
            self.mav = None
        self.detect_engine.stop()
        if hasattr(self, "slam_tab"):
            try:
                self.slam_tab.cleanup()
            except Exception:
                pass
        if hasattr(self, "recon_tab"):
            try:
                self.recon_tab.shutdown()
            except Exception:
                pass
        event.accept()


def main():
    log_path = configure_logging()
    cfg = get_config()
    log_main.info("walle_gcs (%s edition) starting (log: %s)", EDITION.name, log_path)
    log_main.info("config: depth_zone=%s detection=%s",
                  cfg.depth_zone, cfg.detection)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    login = LoginDialog()
    logged_in_user = [None, None]

    def on_login(username, role):
        logged_in_user[0] = username
        logged_in_user[1] = role

    login.login_success.connect(on_login)

    if login.exec_() != QDialog.Accepted:
        sys.exit(0)

    window = MainWindow(username=logged_in_user[0], role=logged_in_user[1])
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
