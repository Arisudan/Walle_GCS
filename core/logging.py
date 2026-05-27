"""Structured logging for walle_gcs_max.

Replaces the historical ``print("[Tag] msg", flush=True)`` pattern across
the GCS with a real logger so we get:

  * Timestamps + level on every line (works offline post-flight).
  * A rotating log file at ``$WALLE_LOG_DIR/walle_gcs_max.log`` (default
    ``~/.walle_gcs/logs``) — kept across restarts for incident review.
  * Per-module logger names that mirror the legacy ``[Tag]`` prefixes so
    grep workflows still work.
  * Env-driven level (``WALLE_LOG_LEVEL`` = DEBUG/INFO/WARN/ERROR).

Usage::

    from core.logging import get_logger
    log = get_logger("Detection")
    log.info("Models loaded OK")
    log.warning("Falling back to CPU: %s", reason)

Call :func:`configure_logging` once at process start. ``get_logger`` is
safe to call before that — it will autoconfigure with defaults the first
time it's invoked.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path
from typing import Optional

_CONFIG_LOCK = threading.Lock()
_CONFIGURED = False

# 10 MB per file × 5 files keeps roughly a week of normal-flight logs
# without unbounded disk growth.
_MAX_BYTES = 10 * 1024 * 1024
_BACKUP_COUNT = 5

_LEVEL_NAMES = {"DEBUG", "INFO", "WARN", "WARNING", "ERROR", "CRITICAL"}


def _resolve_level(name: Optional[str]) -> int:
    if not name:
        return logging.INFO
    n = name.strip().upper()
    if n == "WARN":
        n = "WARNING"
    if n not in _LEVEL_NAMES:
        return logging.INFO
    return getattr(logging, n)


def _resolve_log_dir() -> Path:
    env = os.environ.get("WALLE_LOG_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".walle_gcs" / "logs"


def configure_logging(
    level: Optional[str] = None,
    log_dir: Optional[Path] = None,
    log_file: str = "walle_gcs_max.log",
) -> Path:
    """Install console + rotating-file handlers on the root walle logger.

    Idempotent — calling more than once is a no-op. Returns the path of
    the log file (useful for surfacing in the UI status bar).
    """
    global _CONFIGURED
    with _CONFIG_LOCK:
        if _CONFIGURED:
            return _resolve_log_dir() / log_file

        lvl = _resolve_level(level or os.environ.get("WALLE_LOG_LEVEL"))
        out_dir = Path(log_dir) if log_dir else _resolve_log_dir()
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            # Disk full / RO mount / etc. — fall back to console only and
            # surface the reason so the operator can see why file logs
            # never appeared.
            sys.stderr.write(
                f"[logging] log dir {out_dir} unwritable ({e}); "
                "console-only logging\n")
            out_dir = None

        root = logging.getLogger("walle")
        root.setLevel(lvl)
        root.propagate = False
        # Wipe any handlers from a previous failed init.
        for h in list(root.handlers):
            root.removeHandler(h)

        # Console handler — concise, human-readable.
        console = logging.StreamHandler(stream=sys.stderr)
        console.setLevel(lvl)
        console.setFormatter(logging.Formatter(
            fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        root.addHandler(console)

        # File handler — full timestamp, rotating.
        if out_dir is not None:
            path = out_dir / log_file
            try:
                fileh = logging.handlers.RotatingFileHandler(
                    path,
                    maxBytes=_MAX_BYTES,
                    backupCount=_BACKUP_COUNT,
                    encoding="utf-8",
                )
                fileh.setLevel(lvl)
                fileh.setFormatter(logging.Formatter(
                    fmt="%(asctime)s.%(msecs)03d [%(name)s] %(levelname)s "
                        "%(threadName)s: %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S",
                ))
                root.addHandler(fileh)
            except OSError as e:
                sys.stderr.write(
                    f"[logging] file handler init failed ({e}); "
                    "console-only logging\n")

        _CONFIGURED = True
        return (out_dir / log_file) if out_dir else Path("/dev/null")


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the ``walle`` namespace.

    ``name`` mirrors the legacy print tag (e.g. ``"Detection"``,
    ``"FaceRec"``, ``"Serial"``) so log lines look familiar. Autoconfigures
    on first call if :func:`configure_logging` hasn't been invoked yet.
    """
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger("walle." + name)
