#!/usr/bin/env python3
"""
GPU Monitor — real-time NVIDIA GPU + host CPU/RAM dashboard.

Pulls metrics from NVML directly (no nvidia-smi parsing) at 2 Hz and renders
them in a PyQt5 GUI styled to match walle_gcs.py.

Run:
    source venv/bin/activate
    python gpu_monitor.py
"""

import sys
from collections import deque

import psutil
import pynvml
from PyQt5.QtCore import (
    QRect, QSize, Qt, QThread, QTimer, pyqtSignal
)
from PyQt5.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
)
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QMainWindow, QSizePolicy, QStatusBar,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
)


POLL_INTERVAL_MS = 500          # 2 Hz sampling
HIST_SECONDS = 60               # rolling chart window
HIST_LEN = HIST_SECONDS * (1000 // POLL_INTERVAL_MS)


DARK = """
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
QTableWidget {
    background-color: #0f0f0f;
    color: #cccccc;
    gridline-color: #1e1e1e;
    border: 1px solid #1e1e1e;
    border-radius: 6px;
    font-size: 12px;
}
QHeaderView::section {
    background-color: #141414;
    color: #00cc00;
    border: 0;
    padding: 6px;
    font-weight: bold;
}
QStatusBar { color: #888; }
"""


def fmt_bytes(n):
    n = float(n)
    if n < 1024:
        return f"{int(n)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024:
            return f"{n:.1f} {unit}"
    return f"{n:.1f} PB"


class StatCard(QFrame):
    """Large headline metric card (label + big value + unit)."""

    def __init__(self, label):
        super().__init__()
        self.label_text = label
        self.value_text = "--"
        self.setMinimumHeight(95)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, text):
        self.value_text = text
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)

        p.setPen(QPen(QColor("#1e1e1e"), 1))
        p.setBrush(QColor("#0f0f0f"))
        p.drawRoundedRect(rect, 8, 8)

        p.setPen(QColor("#888"))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(rect.adjusted(16, 12, -16, 0),
                   Qt.AlignLeft | Qt.AlignTop, self.label_text.upper())

        p.setPen(QColor("#00cc00"))
        p.setFont(QFont("Segoe UI", 24, QFont.Bold))
        p.drawText(rect.adjusted(16, 28, -16, -10),
                   Qt.AlignLeft | Qt.AlignVCenter, self.value_text)


class LineChart(QWidget):
    """Rolling line chart drawn with QPainter, 2 Hz update."""

    def __init__(self, title, ceiling=100.0, unit="%", color="#00cc00"):
        super().__init__()
        self.title = title
        self.ceiling = ceiling
        self.unit = unit
        self.color = QColor(color)
        self.history = deque(maxlen=HIST_LEN)
        self.setMinimumHeight(130)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def push(self, value, ceiling=None):
        if ceiling is not None and ceiling > self.ceiling:
            self.ceiling = ceiling
        self.history.append(float(value))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)

        p.setPen(QPen(QColor("#1e1e1e"), 1))
        p.setBrush(QColor("#0f0f0f"))
        p.drawRoundedRect(rect, 8, 8)

        p.setPen(QColor("#888"))
        p.setFont(QFont("Segoe UI", 9, QFont.Bold))
        p.drawText(rect.adjusted(12, 8, -12, 0),
                   Qt.AlignLeft | Qt.AlignTop, self.title.upper())

        if self.history:
            cur = self.history[-1]
            p.setPen(self.color)
            p.setFont(QFont("Segoe UI", 11, QFont.Bold))
            p.drawText(rect.adjusted(12, 8, -12, 0),
                       Qt.AlignRight | Qt.AlignTop, f"{cur:.1f}{self.unit}")

        plot = rect.adjusted(12, 32, -12, -10)
        if len(self.history) < 2 or plot.width() < 10 or plot.height() < 10:
            return

        # gridlines
        p.setPen(QPen(QColor("#181818"), 1))
        for i in range(1, 4):
            y = plot.top() + plot.height() * i / 4
            p.drawLine(plot.left(), int(y), plot.right(), int(y))

        scale = max(self.ceiling, max(self.history) * 1.05, 1e-6)
        w = plot.width()
        h = plot.height()
        pts = []
        for i, v in enumerate(self.history):
            x = plot.left() + (i / (HIST_LEN - 1)) * w
            y = plot.bottom() - (min(v, scale) / scale) * h
            pts.append((x, y))

        # fill under
        path = QPainterPath()
        path.moveTo(pts[0][0], plot.bottom())
        for x, y in pts:
            path.lineTo(x, y)
        path.lineTo(pts[-1][0], plot.bottom())
        path.closeSubpath()
        grad = QLinearGradient(0, plot.top(), 0, plot.bottom())
        c = self.color
        grad.setColorAt(0, QColor(c.red(), c.green(), c.blue(), 90))
        grad.setColorAt(1, QColor(c.red(), c.green(), c.blue(), 0))
        p.fillPath(path, QBrush(grad))

        # line
        p.setPen(QPen(self.color, 2))
        for i in range(1, len(pts)):
            p.drawLine(int(pts[i - 1][0]), int(pts[i - 1][1]),
                       int(pts[i][0]), int(pts[i][1]))


class Bar(QWidget):
    """Horizontal progress bar with label + value (for CPU/RAM)."""

    def __init__(self, label):
        super().__init__()
        self.label_text = label
        self.value_text = ""
        self.percent = 0.0
        self.setMinimumHeight(44)

    def set_value(self, percent, value_text=""):
        self.percent = max(0.0, min(100.0, percent))
        self.value_text = value_text
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = self.rect()

        top_h = rect.height() // 2
        top = QRect(rect.left(), rect.top(), rect.width(), top_h)

        p.setPen(QColor("#888"))
        p.setFont(QFont("Segoe UI", 10, QFont.Bold))
        p.drawText(top, Qt.AlignLeft | Qt.AlignVCenter, self.label_text)

        p.setPen(QColor("#cccccc"))
        p.setFont(QFont("Segoe UI", 10))
        p.drawText(top, Qt.AlignRight | Qt.AlignVCenter,
                   f"{self.value_text}   {self.percent:.0f}%")

        bar_y = rect.top() + top_h + 2
        bar_h = rect.height() - top_h - 4
        bar_rect = QRect(rect.left(), bar_y, rect.width(), bar_h)

        p.setPen(QPen(QColor("#222"), 1))
        p.setBrush(QColor("#080808"))
        p.drawRoundedRect(bar_rect, 4, 4)

        fill = bar_rect.adjusted(2, 2, -2, -2)
        fill.setWidth(int(fill.width() * self.percent / 100.0))
        if self.percent < 75:
            color = QColor("#00cc00")
        elif self.percent < 90:
            color = QColor("#ffaa00")
        else:
            color = QColor("#ff3030")
        p.setBrush(color)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(fill, 3, 3)


class GPUSampler(QThread):
    """Background NVML poller; emits a dict every interval."""

    sample = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, interval_ms=POLL_INTERVAL_MS):
        super().__init__()
        self.interval_ms = interval_ms
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            pynvml.nvmlInit()
        except Exception as e:
            self.error.emit(f"NVML init failed: {e}")
            return

        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            driver = pynvml.nvmlSystemGetDriverVersion()
            if isinstance(driver, bytes):
                driver = driver.decode()

            self.sample.emit({"init": True, "name": name, "driver": driver})

            # prime CPU percent so first reading is real
            psutil.cpu_percent(interval=None)

            while not self._stop:
                data = {"init": False}

                try:
                    u = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    data["util"] = u.gpu
                    data["mem_util"] = u.memory
                except Exception:
                    data["util"] = 0
                    data["mem_util"] = 0

                try:
                    m = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    data["mem_used"] = m.used
                    data["mem_total"] = m.total
                except Exception:
                    data["mem_used"] = 0
                    data["mem_total"] = 1

                try:
                    data["temp"] = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    data["temp"] = 0

                try:
                    data["power"] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except Exception:
                    data["power"] = 0.0
                try:
                    data["power_limit"] = (
                        pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0)
                except Exception:
                    data["power_limit"] = 0.0

                try:
                    data["fan"] = pynvml.nvmlDeviceGetFanSpeed(handle)
                except Exception:
                    data["fan"] = None

                try:
                    data["clock_core"] = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_GRAPHICS)
                except Exception:
                    data["clock_core"] = 0
                try:
                    data["clock_mem"] = pynvml.nvmlDeviceGetClockInfo(
                        handle, pynvml.NVML_CLOCK_MEM)
                except Exception:
                    data["clock_mem"] = 0

                procs = []
                try:
                    plist = []
                    try:
                        plist += list(
                            pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
                    except Exception:
                        pass
                    try:
                        plist += list(
                            pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle))
                    except Exception:
                        pass
                    seen = set()
                    for proc in plist:
                        if proc.pid in seen:
                            continue
                        seen.add(proc.pid)
                        try:
                            pname = psutil.Process(proc.pid).name()
                        except Exception:
                            pname = f"pid {proc.pid}"
                        used = proc.usedGpuMemory or 0
                        procs.append((proc.pid, pname, used))
                    procs.sort(key=lambda x: -x[2])
                except Exception:
                    pass
                data["procs"] = procs

                data["cpu"] = psutil.cpu_percent(interval=None)
                vm = psutil.virtual_memory()
                data["ram_used"] = vm.used
                data["ram_total"] = vm.total

                self.sample.emit(data)
                self.msleep(self.interval_ms)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GPU Monitor")
        self.resize(1320, 860)
        self.setStyleSheet(DARK)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # ── header ─────────────────────────────────────────────
        self.h_title = QLabel("GPU MONITOR")
        self.h_title.setStyleSheet(
            "color: #00cc00; font-size: 16px; font-weight: bold; letter-spacing: 2px;")
        self.h_name = QLabel("(detecting GPU...)")
        self.h_name.setStyleSheet("color: #cccccc; font-size: 13px; font-weight: bold;")
        self.h_driver = QLabel("")
        self.h_driver.setStyleSheet("color: #666; font-size: 11px;")
        hdr = QHBoxLayout()
        hdr.addWidget(self.h_title)
        hdr.addStretch()
        hdr.addWidget(self.h_name)
        hdr.addSpacing(20)
        hdr.addWidget(self.h_driver)
        root.addLayout(hdr)

        # ── stat cards ─────────────────────────────────────────
        self.card_util = StatCard("GPU Utilization")
        self.card_vram = StatCard("VRAM")
        self.card_temp = StatCard("Temperature")
        self.card_power = StatCard("Power")
        cards = QHBoxLayout()
        cards.setSpacing(10)
        for c in (self.card_util, self.card_vram, self.card_temp, self.card_power):
            cards.addWidget(c)
        root.addLayout(cards)

        # ── charts grid ────────────────────────────────────────
        charts = QGridLayout()
        charts.setSpacing(10)
        self.chart_util = LineChart("GPU Util", 100, "%", "#00cc00")
        self.chart_vram = LineChart("VRAM", 1.0, " GB", "#00aaff")
        self.chart_temp = LineChart("Temp", 90, " °C", "#ffaa00")
        self.chart_power = LineChart("Power", 100, " W", "#ff5050")
        self.chart_core = LineChart("Core Clock", 3000, " MHz", "#aa66ff")
        self.chart_mem = LineChart("Mem Clock", 16000, " MHz", "#66ddff")
        for i, c in enumerate((self.chart_util, self.chart_vram, self.chart_temp,
                                self.chart_power, self.chart_core, self.chart_mem)):
            charts.addWidget(c, i // 3, i % 3)
        root.addLayout(charts, 1)

        # ── bottom row: info / processes / host system ────────
        bottom = QHBoxLayout()
        bottom.setSpacing(10)

        info_group = QGroupBox("GPU Info")
        info_layout = QGridLayout(info_group)
        info_layout.setContentsMargins(14, 22, 14, 12)
        info_layout.setVerticalSpacing(8)
        info_layout.setHorizontalSpacing(20)

        def add_row(row, label, name):
            lbl = QLabel(label)
            lbl.setStyleSheet("color: #888; font-weight: normal;")
            val = QLabel("--")
            val.setStyleSheet("color: #cccccc; font-weight: bold;")
            info_layout.addWidget(lbl, row, 0)
            info_layout.addWidget(val, row, 1)
            setattr(self, name, val)

        add_row(0, "Fan speed",     "lbl_fan")
        add_row(1, "Core clock",    "lbl_core")
        add_row(2, "Mem clock",     "lbl_memclk")
        add_row(3, "Mem-bus util",  "lbl_mem_util")
        add_row(4, "Power limit",   "lbl_plimit")
        info_layout.setColumnStretch(1, 1)
        info_layout.setRowStretch(5, 1)

        proc_group = QGroupBox("Processes Using GPU")
        proc_layout = QVBoxLayout(proc_group)
        proc_layout.setContentsMargins(14, 22, 14, 12)
        self.proc_table = QTableWidget(0, 3)
        self.proc_table.setHorizontalHeaderLabels(["PID", "Process", "VRAM"])
        self.proc_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents)
        self.proc_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.proc_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeToContents)
        self.proc_table.verticalHeader().setVisible(False)
        self.proc_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.proc_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.proc_table.setShowGrid(False)
        self.proc_table.setFocusPolicy(Qt.NoFocus)
        proc_layout.addWidget(self.proc_table)

        sys_group = QGroupBox("Host System")
        sys_layout = QVBoxLayout(sys_group)
        sys_layout.setContentsMargins(14, 22, 14, 12)
        sys_layout.setSpacing(10)
        self.bar_cpu = Bar("CPU")
        self.bar_ram = Bar("RAM")
        sys_layout.addWidget(self.bar_cpu)
        sys_layout.addWidget(self.bar_ram)
        sys_layout.addStretch()

        bottom.addWidget(info_group, 1)
        bottom.addWidget(proc_group, 2)
        bottom.addWidget(sys_group, 1)
        root.addLayout(bottom, 1)

        # ── status bar ─────────────────────────────────────────
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Starting NVML sampler...")

        # ── sampler thread ─────────────────────────────────────
        self.sampler = GPUSampler(POLL_INTERVAL_MS)
        self.sampler.sample.connect(self.on_sample)
        self.sampler.error.connect(self.on_error)
        self.sampler.start()

    def closeEvent(self, e):
        self.sampler.stop()
        self.sampler.wait(1500)
        e.accept()

    def on_error(self, msg):
        self.status.showMessage(msg)
        self.h_name.setText("No GPU detected")

    def on_sample(self, d):
        if d.get("init"):
            self.h_name.setText(d.get("name", ""))
            self.h_driver.setText(f"Driver {d.get('driver', '')}")
            self.status.showMessage(
                f"Sampling at {1000 // POLL_INTERVAL_MS} Hz   |   "
                f"{HIST_SECONDS}s rolling window")
            return

        util = d.get("util", 0)
        mem_used = d.get("mem_used", 0)
        mem_total = d.get("mem_total", 1)
        used_gb = mem_used / 1e9
        total_gb = mem_total / 1e9
        temp = d.get("temp", 0)
        power = d.get("power", 0.0)
        plimit = d.get("power_limit", 0.0)
        fan = d.get("fan")
        core = d.get("clock_core", 0)
        mclk = d.get("clock_mem", 0)
        mbus = d.get("mem_util", 0)

        self.card_util.set_value(f"{util}%")
        self.card_vram.set_value(f"{used_gb:.1f} / {total_gb:.1f} GB")
        self.card_temp.set_value(f"{temp} °C")
        if plimit > 0:
            self.card_power.set_value(f"{power:.0f} / {plimit:.0f} W")
        else:
            self.card_power.set_value(f"{power:.0f} W")

        self.chart_util.push(util, 100)
        self.chart_vram.push(used_gb, total_gb)
        self.chart_temp.push(temp, 90)
        self.chart_power.push(power, plimit if plimit > 0 else 600)
        self.chart_core.push(core, 3500)
        self.chart_mem.push(mclk, 16000)

        self.lbl_fan.setText(f"{fan}%" if fan is not None else "N/A")
        self.lbl_core.setText(f"{core} MHz")
        self.lbl_memclk.setText(f"{mclk} MHz")
        self.lbl_mem_util.setText(f"{mbus}%")
        self.lbl_plimit.setText(f"{plimit:.0f} W" if plimit > 0 else "N/A")

        procs = d.get("procs", [])
        self.proc_table.setRowCount(len(procs))
        for i, (pid, name, used) in enumerate(procs):
            self.proc_table.setItem(i, 0, QTableWidgetItem(str(pid)))
            self.proc_table.setItem(i, 1, QTableWidgetItem(name))
            self.proc_table.setItem(i, 2, QTableWidgetItem(fmt_bytes(used)))

        cpu = d.get("cpu", 0)
        self.bar_cpu.set_value(cpu, f"{cpu:.0f}%")
        ru = d.get("ram_used", 0) / 1e9
        rt = d.get("ram_total", 1) / 1e9
        self.bar_ram.set_value(100 * ru / rt if rt > 0 else 0,
                                f"{ru:.1f} / {rt:.1f} GB")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
