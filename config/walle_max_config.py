"""Typed configuration for walle_gcs_max.

Replaces the legacy "magic numbers scattered through the file" pattern
with a single dataclass tree that:

  * Validates every field at load time — out-of-range thresholds, bad
    enum values, etc. fail fast at startup instead of silently breaking
    drone behavior mid-flight.
  * Layers JSON config (``walle_settings_max.json``) over typed defaults
    so the existing settings file keeps working, and unknown / missing
    keys don't crash — they fall back to the dataclass default.
  * Exposes a single :func:`load_config` that the GCS calls once at
    startup; engines stay free of JSON-loading code.

Add new tunables here, then read them via the returned config object
or the module-level singleton :func:`get_config`.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional


# ── Defaults map 1:1 to the historical hard-coded values ──────────────


@dataclass
class AlertConfig:
    low_batt_warn: int = 30
    low_batt_crit: int = 15
    rssi_warn: int = 90
    lq_warn: int = 50

    def validate(self) -> None:
        for name, lo, hi in (
            ("low_batt_warn", 0, 100),
            ("low_batt_crit", 0, 100),
            ("rssi_warn", 0, 130),
            ("lq_warn", 0, 100),
        ):
            v = getattr(self, name)
            if not (lo <= v <= hi):
                raise ValueError(
                    f"alerts.{name}={v} out of range [{lo}, {hi}]")
        if self.low_batt_crit >= self.low_batt_warn:
            raise ValueError(
                "alerts.low_batt_crit must be < low_batt_warn "
                f"({self.low_batt_crit} >= {self.low_batt_warn})")


@dataclass
class DetectionConfig:
    """RF-DETR Medium + Depth-Anything-V2 detection engine."""
    conf_thresh: float = 0.5
    depth_variant: str = "Base"    # Small | Base | Large
    rfdetr_compile: bool = False   # WALLE_NO_PRELOAD / RFDETR_COMPILE override
    deepsort_max_age: int = 30
    deepsort_n_init: int = 3
    # Worker inference cadence caps. The video display always runs at
    # camera FPS (drawing the most-recent cached result); these only
    # bound how often the detection / depth models actually run, which is
    # the dominant CPU cost. 0 disables the cap (run flat-out). Defaults
    # trade a little detection latency for a large CPU reduction.
    target_fps: float = 15.0        # detection (RF-DETR + DeepSort) worker
    depth_target_fps: float = 10.0  # depth (Depth-Anything) worker

    def validate(self) -> None:
        if not (0.0 <= self.conf_thresh <= 1.0):
            raise ValueError(
                f"detection.conf_thresh={self.conf_thresh} not in [0, 1]")
        if self.depth_variant not in ("Small", "Base", "Large"):
            raise ValueError(
                f"detection.depth_variant={self.depth_variant!r} must be "
                "Small / Base / Large")
        for name in ("deepsort_max_age", "deepsort_n_init"):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"detection.{name} must be positive")
        for name in ("target_fps", "depth_target_fps"):
            v = getattr(self, name)
            if not (0.0 <= v <= 240.0):
                raise ValueError(
                    f"detection.{name}={v} must be in [0, 240] (0 = uncapped)")


@dataclass
class FaceRecConfig:
    """SCRFD + ArcFace face recognition engine."""
    frame_skip: int = 3
    scales: tuple = (1.0, 2.0)
    far_face_threshold: int = 50
    enable_far_face_zoom: bool = True
    enable_multiscale: bool = True

    def validate(self) -> None:
        if self.frame_skip <= 0:
            raise ValueError("face_rec.frame_skip must be positive")
        if not self.scales:
            raise ValueError("face_rec.scales cannot be empty")
        for s in self.scales:
            if not (0.25 <= s <= 4.0):
                raise ValueError(
                    f"face_rec.scales={self.scales}: {s} out of [0.25, 4.0]")
        if self.far_face_threshold < 0:
            raise ValueError("face_rec.far_face_threshold must be >= 0")


@dataclass
class DepthZoneConfig:
    """Forward-zone closeness analysis (driving the STOP / SLOW / GO HUD).

    Closeness is the 95th-percentile of the normalized 0..255 depth map
    within the center column; higher value = obstacle is nearer.
    """
    center_zone_frac: float = 0.34
    stop_depth_rel: float = 180.0
    slow_depth_rel: float = 140.0

    def validate(self) -> None:
        if not (0.05 <= self.center_zone_frac <= 1.0):
            raise ValueError(
                f"depth_zone.center_zone_frac={self.center_zone_frac} "
                "must be in [0.05, 1.0]")
        if not (0.0 <= self.slow_depth_rel < self.stop_depth_rel <= 255.0):
            raise ValueError(
                "depth_zone.slow_depth_rel must be < stop_depth_rel, "
                "both within [0, 255]")


@dataclass
class HudConfig:
    interval_ms: int = 100         # HUD overlay rebuild cadence (10 Hz)

    def validate(self) -> None:
        if not (16 <= self.interval_ms <= 1000):
            raise ValueError(
                f"hud.interval_ms={self.interval_ms} must be in [16, 1000]")


@dataclass
class VideoConfig:
    ip: str = "127.0.0.1"
    port: int = 5600
    aspect_w: int = 16
    aspect_h: int = 9
    # Cap the resolution that flows through the per-frame GUI pipeline
    # (overlay drawing, depth colorize/stack, QPixmap scale). HDMI capture
    # cards often only deliver native (e.g. 1080p) frames; we request this
    # size and, if the card ignores it, downscale on the capture thread so
    # the GUI thread never composites more pixels than it needs to.
    # 0 on either axis disables the cap.
    capture_max_width: int = 1280
    capture_max_height: int = 720

    def validate(self) -> None:
        if not (1 <= self.port <= 65535):
            raise ValueError(f"video.port={self.port} not a valid TCP port")
        if self.aspect_w <= 0 or self.aspect_h <= 0:
            raise ValueError("video.aspect_w / aspect_h must be positive")
        if self.capture_max_width < 0 or self.capture_max_height < 0:
            raise ValueError("video.capture_max_width / height must be >= 0")


@dataclass
class WalleMaxConfig:
    """Root config — every subsystem hangs off here."""
    alerts: AlertConfig = field(default_factory=AlertConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    face_rec: FaceRecConfig = field(default_factory=FaceRecConfig)
    depth_zone: DepthZoneConfig = field(default_factory=DepthZoneConfig)
    hud: HudConfig = field(default_factory=HudConfig)
    video: VideoConfig = field(default_factory=VideoConfig)

    def validate(self) -> None:
        for f in fields(self):
            sub = getattr(self, f.name)
            if hasattr(sub, "validate"):
                sub.validate()


# ── JSON layering ─────────────────────────────────────────────────────


def _apply_dict(target: Any, src: dict) -> None:
    """Recursively overlay ``src`` keys onto dataclass ``target`` fields.

    Unknown keys are silently ignored so old / new settings files stay
    forward and backward compatible.
    """
    if not is_dataclass(target):
        return
    for f in fields(target):
        if f.name not in src:
            continue
        val = src[f.name]
        current = getattr(target, f.name)
        if is_dataclass(current):
            if isinstance(val, dict):
                _apply_dict(current, val)
            # else: skip — wrong shape, leave default
            continue
        # Tuples in JSON come back as lists — convert if the field defaults to a tuple
        if isinstance(current, tuple) and isinstance(val, list):
            val = tuple(val)
        setattr(target, f.name, val)


# Map legacy walle_settings_max.json shape (flat-ish) to our nested dataclasses.
# Only the alerts / video sections currently overlap one-to-one.
_LEGACY_MAP = {
    "alerts": "alerts",
    "video_streaming": "video",
}


def _from_legacy_json(payload: dict) -> dict:
    """Translate the historical walle_settings_max.json layout to the
    nested shape WalleMaxConfig expects."""
    out: dict = {}
    for legacy_key, new_key in _LEGACY_MAP.items():
        if legacy_key in payload and isinstance(payload[legacy_key], dict):
            out[new_key] = payload[legacy_key]
    # Native (non-legacy) sections that already use our new shape
    for nk in ("detection", "face_rec", "depth_zone", "hud"):
        if nk in payload and isinstance(payload[nk], dict):
            out[nk] = payload[nk]
    return out


_SINGLETON_LOCK = threading.Lock()
_SINGLETON: Optional[WalleMaxConfig] = None


def load_config(path: Optional[Path] = None) -> WalleMaxConfig:
    """Build a validated :class:`WalleMaxConfig`.

    ``path`` defaults to ``walle_settings_max.json`` next to this file's
    project root. Missing file → all defaults. Bad JSON → defaults plus
    a warning printed to stderr (we don't want a typo in a settings file
    to brick the drone GCS).
    """
    global _SINGLETON
    cfg = WalleMaxConfig()

    if path is None:
        # walle_gcs/ is two levels up from this module (config/walle_max_config.py)
        project_root = Path(__file__).resolve().parent.parent
        path = project_root / "walle_settings_max.json"

    if path.exists():
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            import sys
            sys.stderr.write(
                f"[config] failed to parse {path} ({e}); using defaults\n")
            payload = {}
        if payload:
            _apply_dict(cfg, _from_legacy_json(payload))

    # Env overrides — picked up before validate() so they can correct a
    # broken JSON.
    if os.environ.get("RFDETR_COMPILE", "0") == "1":
        cfg.detection.rfdetr_compile = True

    cfg.validate()
    with _SINGLETON_LOCK:
        _SINGLETON = cfg
    return cfg


def get_config() -> WalleMaxConfig:
    """Return the active config singleton, loading defaults if no
    :func:`load_config` call has happened yet."""
    global _SINGLETON
    with _SINGLETON_LOCK:
        if _SINGLETON is None:
            _SINGLETON = WalleMaxConfig()
            _SINGLETON.validate()
        return _SINGLETON
