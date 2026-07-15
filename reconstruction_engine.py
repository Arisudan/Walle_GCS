#!/usr/bin/env python3
"""
3D Reconstruction tab for Walle GCS — a thin wrapper around LingBot-Map.

LingBot-Map ("Geometric Context Transformer for Streaming 3D Reconstruction")
ships its own ``demo.py`` which, given an offline video or image folder, runs
streaming inference, writes a fused ``.ply`` point cloud, and serves an
interactive 3D view through a browser-based viser viewer.

LingBot-Map lives in its own virtualenv with a different (incompatible) torch
build than the GCS, so it CANNOT be imported in-process. This wrapper instead
launches ``demo.py`` as a subprocess through LingBot-Map's own venv, streams its
console output into a log pane, auto-detects when the ``.ply`` is written and
when the viser viewer comes up, and opens the viewer in the browser.

Nothing here imports torch / viser / lingbot_map — it is pure subprocess + Qt,
so it is safe to import from walle_gcs.py.
"""

import os
import re
import sys
import signal
import subprocess
import webbrowser

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QComboBox, QPushButton,
    QSpinBox, QCheckBox, QLineEdit, QPlainTextEdit, QFileDialog, QSizePolicy,
)


# ─── LingBot-Map installation discovery ─────────────────────────────────────

def find_lingbot_map_dir():
    """Locate the LingBot-Map checkout (the dir containing ``demo.py``)."""
    env_dir = os.environ.get("WALLE_LINGBOT_MAP_DIR")
    if env_dir:
        env_dir = os.path.abspath(env_dir)
        if os.path.isfile(os.path.join(env_dir, "demo.py")):
            return env_dir
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "lingbot-map"),                  # walle_gcs/ -> ./lingbot-map
        os.path.join(here, "..", "lingbot-map"),              # rosetta/ -> ../lingbot-map
        os.path.join(here, "..", "..", "lingbot-map"),        # rosetta/walle_gcs/ -> ../../lingbot-map
        os.path.join(here, "..", "lingbot-map-fresh"),
        os.path.join(here, "..", "..", "lingbot-map-fresh"),
        "/home/ndrone/Downloads/Drone Development/Drone Development/lingbot-map",
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isfile(os.path.join(c, "demo.py")):
            return c
    return os.path.abspath(candidates[0])  # best guess; UI will flag if missing


LINGBOT_DIR = find_lingbot_map_dir()


def _venv_python(lingbot_dir):
    """LingBot-Map's own interpreter — falls back to system python3."""
    candidates = [
        os.path.join(lingbot_dir, "venv", "Scripts", "python.exe"),
        os.path.join(lingbot_dir, "venv", "bin", "python3"),
        os.path.join(lingbot_dir, "venv", "bin", "python"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return "python3"


def _model_path(lingbot_dir):
    return os.path.join(lingbot_dir, "checkpoints", "lingbot-map-long.pt")


# ─── Subprocess worker ──────────────────────────────────────────────────────

class LingbotReconWorker(QThread):
    """Runs LingBot-Map ``demo.py`` in its own venv and streams its output.

    The subprocess does the whole job: frame extraction → streaming inference →
    ``.ply`` export → viser viewer (which then serves until killed). We just
    watch stdout for the two milestones and surface them as signals.
    """

    log_line      = pyqtSignal(str)
    ply_exported  = pyqtSignal(str)   # path to the written .ply
    viewer_ready  = pyqtSignal(str)   # http://localhost:<port>
    finished_run  = pyqtSignal(int)   # process exit code (-1 if never started)

    _RE_VIEWER = re.compile(r"http://localhost:\d+")

    def __init__(self, cmd, cwd, out_ply, port, parent=None):
        super().__init__(parent)
        self._cmd = cmd
        self._cwd = cwd
        self._out_ply = out_ply
        self._port = port
        self._proc = None
        self._stop = False

    def run(self):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self._proc = subprocess.Popen(
                self._cmd, cwd=self._cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                # New process group so we can signal the whole tree (viser
                # spawns a server) on stop.
                start_new_session=True,
            )
        except Exception as e:                       # noqa: BLE001
            self.log_line.emit(f"[recon] failed to launch: {e}")
            self.finished_run.emit(-1)
            return

        # Read in small blocks and treat both \n and \r (tqdm progress bars use
        # \r) as line breaks so the log pane updates live instead of stalling.
        buf = ""
        while not self._stop:
            chunk = self._proc.stdout.read(128)
            if not chunk:
                break
            buf += chunk.replace("\r", "\n")
            *lines, buf = buf.split("\n")
            for ln in lines:
                if ln.strip():
                    self._handle_line(ln)
        if buf.strip():
            self._handle_line(buf)

        code = self._proc.wait()
        self.finished_run.emit(code)

    def _handle_line(self, line):
        self.log_line.emit(line)
        if "Exported PLY" in line:
            # demo.py prints: "Exported PLY (N points, conf>X) to <path>"
            self.ply_exported.emit(self._out_ply)
        elif "3D viewer at" in line or "viser" in line.lower() and "localhost" in line:
            m = self._RE_VIEWER.search(line)
            self.viewer_ready.emit(m.group(0) if m else f"http://localhost:{self._port}")

    def stop(self):
        """Terminate the subprocess (and the viser server it spawned)."""
        self._stop = True
        proc = self._proc
        if proc and proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:                        # noqa: BLE001
                proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:                    # noqa: BLE001
                    proc.kill()


# ─── Tab: 3D Reconstruction ─────────────────────────────────────────────────

class ReconstructionTab(QWidget):
    """Pick an offline video / image folder → LingBot-Map reconstructs it →
    a ``.ply`` is written and the viser 3D viewer opens in the browser."""

    def __init__(self, lingbot_dir=None, parent=None):
        super().__init__(parent)
        self.lingbot_dir = lingbot_dir or LINGBOT_DIR
        self._worker = None
        self._input_path = None
        self._viewer_url = None
        self._build_ui()
        self._check_install()

    # ── UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Source row ──
        src = QFrame()
        src.setStyleSheet("QFrame { background-color: #0d0d0d; border-radius: 6px; }")
        sl = QHBoxLayout(src)
        sl.setContentsMargins(8, 6, 8, 6)
        sl.setSpacing(6)

        sl.addWidget(self._lbl("Source:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Video file", "Image folder"])
        self.source_combo.setFixedWidth(120)
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        sl.addWidget(self.source_combo)

        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.setFixedHeight(28)
        self.browse_btn.clicked.connect(self._on_browse)
        sl.addWidget(self.browse_btn)

        self.file_label = QLabel("(no file selected)")
        self.file_label.setStyleSheet("color: #777; font-size: 11px;")
        sl.addWidget(self.file_label, stretch=1)
        root.addWidget(src)

        # ── Options row ──
        opt = QFrame()
        opt.setStyleSheet("QFrame { background-color: #0d0d0d; border-radius: 6px; }")
        ol = QHBoxLayout(opt)
        ol.setContentsMargins(8, 6, 8, 6)
        ol.setSpacing(6)

        ol.addWidget(self._lbl("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.setFixedWidth(56)
        self.fps_spin.setToolTip("Frames per second sampled from the video")
        ol.addWidget(self.fps_spin)

        ol.addWidget(self._lbl("Keyframe interval:"))
        self.kf_spin = QSpinBox()
        self.kf_spin.setRange(1, 16)
        self.kf_spin.setValue(2)
        self.kf_spin.setFixedWidth(56)
        self.kf_spin.setToolTip("Keep every N-th frame as a keyframe (lowers KV-cache memory)")
        ol.addWidget(self.kf_spin)

        self.sky_check = QCheckBox("Mask sky")
        self.sky_check.setStyleSheet("QCheckBox { color: #ccc; font-size: 11px; }")
        self.sky_check.setToolTip("Filter sky points (outdoor scenes)")
        ol.addWidget(self.sky_check)

        ol.addWidget(self._sep())

        ol.addWidget(self._lbl("Viewer port:"))
        self.port_spin = QSpinBox()
        self.port_spin.setRange(1024, 65535)
        self.port_spin.setValue(8080)
        self.port_spin.setFixedWidth(72)
        ol.addWidget(self.port_spin)
        ol.addStretch()
        root.addWidget(opt)

        # ── Output .ply row ──
        out = QFrame()
        out.setStyleSheet("QFrame { background-color: #0d0d0d; border-radius: 6px; }")
        oll = QHBoxLayout(out)
        oll.setContentsMargins(8, 6, 8, 6)
        oll.setSpacing(6)

        oll.addWidget(self._lbl("Output .ply:"))
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("auto: <input name>.ply next to the source")
        self.out_edit.setStyleSheet(
            "QLineEdit { background-color: #111; color: #ccc; border: 1px solid #222; "
            "border-radius: 4px; padding: 3px 6px; font-size: 11px; }")
        oll.addWidget(self.out_edit, stretch=1)

        self.out_btn = QPushButton("Save As…")
        self.out_btn.setFixedHeight(28)
        self.out_btn.clicked.connect(self._on_pick_output)
        oll.addWidget(self.out_btn)
        root.addWidget(out)

        # ── Action row ──
        act = QHBoxLayout()
        act.setSpacing(6)
        self.run_btn = QPushButton("Run Reconstruction")
        self.run_btn.setMinimumHeight(34)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #102015; color: #00dd66;
                font-size: 13px; font-weight: bold;
                border: 1px solid #00aa44; border-radius: 6px; padding: 4px 14px;
            }
            QPushButton:hover { background-color: #16301f; }
            QPushButton:disabled { color: #555; border-color: #222; }
        """)
        self.run_btn.clicked.connect(self._on_run)
        act.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setMinimumHeight(34)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #200a0a; color: #ff5555;
                font-size: 13px; font-weight: bold;
                border: 1px solid #aa2222; border-radius: 6px; padding: 4px 14px;
            }
            QPushButton:hover { background-color: #2e0d0d; }
            QPushButton:disabled { color: #555; border-color: #222; }
        """)
        self.stop_btn.clicked.connect(self._on_stop)
        act.addWidget(self.stop_btn)

        self.viewer_btn = QPushButton("Open 3D Viewer")
        self.viewer_btn.setMinimumHeight(34)
        self.viewer_btn.setEnabled(False)
        self.viewer_btn.setStyleSheet("""
            QPushButton {
                background-color: #0c1a28; color: #33bbff;
                font-size: 13px; font-weight: bold;
                border: 1px solid #2277bb; border-radius: 6px; padding: 4px 14px;
            }
            QPushButton:hover { background-color: #102234; }
            QPushButton:disabled { color: #555; border-color: #222; }
        """)
        self.viewer_btn.clicked.connect(self._on_open_viewer)
        act.addWidget(self.viewer_btn)
        act.addStretch()
        root.addLayout(act)

        # ── Status ──
        self.status_label = QLabel("Idle — select an offline video or image folder, then Run.")
        self.status_label.setStyleSheet("color: #888; font-size: 11px; padding: 2px 4px;")
        root.addWidget(self.status_label)

        # ── Log pane ──
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(2000)
        self.log_view.setStyleSheet(
            "QPlainTextEdit { background-color: #080808; color: #bbb; "
            "border: 1px solid #1a1a1a; border-radius: 6px; "
            "font-family: 'Consolas','DejaVu Sans Mono',monospace; font-size: 11px; }")
        self.log_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.log_view, stretch=1)

        # ── Footer: which install we're driving ──
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #555; font-size: 10px; padding: 2px 4px;")
        self.info_label.setText(f"LingBot-Map: {self.lingbot_dir}")
        root.addWidget(self.info_label)

    def _lbl(self, text):
        l = QLabel(text)
        l.setStyleSheet("color: #777; font-size: 11px;")
        return l

    def _sep(self):
        s = QFrame()
        s.setFrameShape(QFrame.VLine)
        s.setFixedWidth(1)
        s.setStyleSheet("background-color: #222;")
        return s

    # ── Install sanity check ────────────────────────────────────────────
    def _check_install(self):
        demo = os.path.join(self.lingbot_dir, "demo.py")
        model = _model_path(self.lingbot_dir)
        problems = []
        if not os.path.isfile(demo):
            problems.append(f"demo.py not found in {self.lingbot_dir}")
        if not os.path.isdir(self.lingbot_dir):
            problems.append("LingBot-Map directory does not exist")
        if not os.path.isfile(model):
            problems.append(f"checkpoint missing: {model}")
        if problems:
            self.run_btn.setEnabled(False)
            self.status_label.setText(
                "LingBot-Map not found. Set WALLE_LINGBOT_MAP_DIR to the checkout containing demo.py. "
                + "; ".join(problems)
            )
            self.status_label.setStyleSheet("color: #ff6666; font-size: 11px; padding: 2px 4px;")

    # ── Source selection ────────────────────────────────────────────────
    def _on_source_changed(self, _idx):
        is_video = self.source_combo.currentIndex() == 0
        self.fps_spin.setEnabled(is_video)
        self._input_path = None
        self.file_label.setText("(no file selected)")

    def _on_browse(self):
        if self.source_combo.currentIndex() == 0:   # video file
            path, _ = QFileDialog.getOpenFileName(
                self, "Choose offline video", os.path.expanduser("~"),
                "Video files (*.mp4 *.mov *.avi *.mkv *.MP4 *.MOV);;All files (*)")
        else:                                       # image folder
            path = QFileDialog.getExistingDirectory(
                self, "Choose image folder", os.path.expanduser("~"))
        if not path:
            return
        self._input_path = path
        self.file_label.setText(os.path.basename(path.rstrip("/")) or path)
        self.file_label.setToolTip(path)
        # Default output .ply next to the source.
        if not self.out_edit.text().strip():
            stem = os.path.splitext(os.path.basename(path.rstrip("/")))[0]
            base = path if os.path.isdir(path) else os.path.dirname(path)
            self.out_edit.setText(os.path.join(base, f"{stem}.ply"))

    def _on_pick_output(self):
        start = self.out_edit.text().strip() or os.path.expanduser("~/reconstruction.ply")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save point cloud as", start, "PLY point cloud (*.ply)")
        if path:
            if not path.lower().endswith(".ply"):
                path += ".ply"
            self.out_edit.setText(path)

    # ── Run / stop ──────────────────────────────────────────────────────
    def _on_run(self):
        if self._worker and self._worker.isRunning():
            return
        if not self._input_path or not os.path.exists(self._input_path):
            self.status_label.setText("Select a valid offline video or image folder first.")
            return

        out_ply = self.out_edit.text().strip()
        if not out_ply:
            stem = os.path.splitext(os.path.basename(self._input_path.rstrip("/")))[0]
            base = (self._input_path if os.path.isdir(self._input_path)
                    else os.path.dirname(self._input_path))
            out_ply = os.path.join(base, f"{stem}.ply")
            self.out_edit.setText(out_ply)
        out_ply = os.path.abspath(out_ply)

        port = self.port_spin.value()
        is_video = self.source_combo.currentIndex() == 0

        cmd = [
            _venv_python(self.lingbot_dir), "demo.py",
            "--model_path", _model_path(self.lingbot_dir),
            ("--video_path" if is_video else "--image_folder"),
            os.path.abspath(self._input_path),
            "--keyframe_interval", str(self.kf_spin.value()),
            "--use_sdpa",                      # no FlashInfer dependency
            "--export_ply", out_ply,           # <-- the .ply gets written
            "--port", str(port),               # viser viewer port
        ]
        if is_video:
            cmd += ["--fps", str(self.fps_spin.value())]
        if self.sky_check.isChecked():
            cmd += ["--mask_sky"]

        self.log_view.clear()
        self._append_log("$ " + " ".join(self._quote(c) for c in cmd))
        self._append_log(f"(cwd: {self.lingbot_dir})\n")
        self._viewer_url = None
        self.viewer_btn.setEnabled(False)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Running reconstruction — extracting frames / inference…")
        self.status_label.setStyleSheet("color: #ffcc44; font-size: 11px; padding: 2px 4px;")

        self._worker = LingbotReconWorker(cmd, self.lingbot_dir, out_ply, port)
        self._worker.log_line.connect(self._append_log)
        self._worker.ply_exported.connect(self._on_ply_exported)
        self._worker.viewer_ready.connect(self._on_viewer_ready)
        self._worker.finished_run.connect(self._on_finished)
        self._worker.start()

    def _on_stop(self):
        self.status_label.setText("Stopping…")
        if self._worker:
            self._worker.stop()

    # ── Worker signal handlers ──────────────────────────────────────────
    def _append_log(self, line):
        self.log_view.appendPlainText(line)
        sb = self.log_view.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_ply_exported(self, path):
        self.status_label.setText(f"Point cloud written: {path} — launching 3D viewer…")
        self.status_label.setStyleSheet("color: #00dd66; font-size: 11px; padding: 2px 4px;")

    def _on_viewer_ready(self, url):
        self._viewer_url = url
        self.viewer_btn.setEnabled(True)
        self.status_label.setText(f"3D viewer live at {url} — opening in browser…")
        self.status_label.setStyleSheet("color: #33bbff; font-size: 11px; padding: 2px 4px;")
        try:
            webbrowser.open(url)
        except Exception:                            # noqa: BLE001
            pass

    def _on_open_viewer(self):
        if self._viewer_url:
            webbrowser.open(self._viewer_url)

    def _on_finished(self, code):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if code == 0:
            self.status_label.setText("Reconstruction finished.")
            self.status_label.setStyleSheet("color: #888; font-size: 11px; padding: 2px 4px;")
        elif code in (-15, 143, -2, 130, -9, 137):   # SIGTERM / SIGINT / SIGKILL
            self.status_label.setText("Reconstruction stopped — viewer closed.")
            self.status_label.setStyleSheet("color: #888; font-size: 11px; padding: 2px 4px;")
            self._viewer_url = None
            self.viewer_btn.setEnabled(False)
        else:
            self.status_label.setText(f"Reconstruction exited with code {code} — see log above.")
            self.status_label.setStyleSheet("color: #ff6666; font-size: 11px; padding: 2px 4px;")

    @staticmethod
    def _quote(s):
        return f'"{s}"' if " " in s else s

    # ── Lifecycle ───────────────────────────────────────────────────────
    def shutdown(self):
        """Called by MainWindow on close — kill the subprocess + viser server."""
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(6000)
