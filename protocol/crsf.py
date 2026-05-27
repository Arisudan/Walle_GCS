"""CRSF (Crossfire) telemetry parser and serial reader.

Extracted from walle_gcs_max.py as the first step in breaking up the
~5000-line GUI file. CRSF is a self-contained protocol used by ELRS /
TBS Crossfire radios over USB serial — it knows nothing about Qt
widgets, video, AI engines, etc., so it lives in its own module and
can be unit-tested in isolation.

Exposes:

  * ``CRSF_FRAMETYPE_*`` constants — used by tests and by GUI code that
    needs to check raw frame types.
  * :class:`CRSFTelemetry` — pure-Python frame parser. No external
    dependencies beyond struct.
  * :class:`SerialReader` — QThread that auto-detects baud rate and
    emits ``telemetry_updated`` on every successfully parsed frame.
"""

from __future__ import annotations

import struct
import time

import serial
from PyQt5.QtCore import QThread, pyqtSignal

from core.logging import get_logger


log_crsf = get_logger("CRSF")
log_serial = get_logger("Serial")


# ─── CRSF Protocol Constants ────────────────────────────────────────
CRSF_SYNC_BYTE = 0xC8
CRSF_SYNC_BYTE_TX = 0xEA  # TX module address (used in telem mirror)
CRSF_FRAMETYPE_GPS = 0x02
CRSF_FRAMETYPE_VARIO = 0x07
CRSF_FRAMETYPE_BATTERY = 0x08
CRSF_FRAMETYPE_LINK_STATS = 0x14
CRSF_FRAMETYPE_RC_CHANNELS = 0x16
CRSF_FRAMETYPE_ATTITUDE = 0x1E
CRSF_FRAMETYPE_FLIGHT_MODE = 0x21


class CRSFTelemetry:
    """Parses CRSF telemetry from RadioMaster serial port."""

    def __init__(self):
        self.battery_voltage = 0.0
        self.battery_current = 0.0
        self.battery_mah = 0
        self.battery_percent = 0
        self.rssi_1 = 0
        self.rssi_2 = 0
        self.link_quality = 0
        self.snr = 0
        self.rf_mode = 0
        self.tx_power = 0
        # GPS
        self.latitude = 0.0
        self.longitude = 0.0
        self.ground_speed = 0.0
        self.heading = 0.0
        self.altitude = 0.0
        self.satellites = 0
        # Attitude
        self.pitch = 0.0
        self.roll = 0.0
        self.yaw = 0.0
        # Vario
        self.vertical_speed = 0.0
        # Flight mode
        self.flight_mode = ""
        # RC channels (16 channels, raw 11-bit values 172-1811, center 992)
        self.rc_channels = [992] * 16

    def parse_frame(self, frame_type, payload):
        if frame_type == CRSF_FRAMETYPE_GPS and len(payload) >= 15:
            self.latitude = struct.unpack(">i", payload[0:4])[0] / 1e7
            self.longitude = struct.unpack(">i", payload[4:8])[0] / 1e7
            self.ground_speed = struct.unpack(">H", payload[8:10])[0] / 36.0  # dm/s to m/s
            self.heading = struct.unpack(">H", payload[10:12])[0] / 100.0
            self.altitude = (struct.unpack(">H", payload[12:14])[0] - 1000)  # meters, offset by 1000
            self.satellites = payload[14]
        elif frame_type == CRSF_FRAMETYPE_BATTERY and len(payload) >= 8:
            self.battery_voltage = struct.unpack(">H", payload[0:2])[0] / 10.0
            self.battery_current = struct.unpack(">H", payload[2:4])[0] / 10.0
            self.battery_mah = (payload[4] << 16 | payload[5] << 8 | payload[6])
            self.battery_percent = payload[7]
        elif frame_type == CRSF_FRAMETYPE_LINK_STATS and len(payload) >= 10:
            self.rssi_1 = payload[0]
            self.rssi_2 = payload[1]
            self.link_quality = payload[2]
            self.snr = struct.unpack("b", bytes([payload[3]]))[0]
            self.rf_mode = payload[5]
            self.tx_power = payload[6]
        elif frame_type == CRSF_FRAMETYPE_ATTITUDE and len(payload) >= 6:
            self.pitch = struct.unpack(">h", payload[0:2])[0] / 10000.0 * (180.0 / 3.14159)
            self.roll = struct.unpack(">h", payload[2:4])[0] / 10000.0 * (180.0 / 3.14159)
            self.yaw = struct.unpack(">h", payload[4:6])[0] / 10000.0 * (180.0 / 3.14159)
        elif frame_type == CRSF_FRAMETYPE_VARIO and len(payload) >= 2:
            self.vertical_speed = struct.unpack(">h", payload[0:2])[0] / 10.0
        elif frame_type == CRSF_FRAMETYPE_RC_CHANNELS and len(payload) >= 22:
            # 16 channels packed as 11-bit values in 22 bytes
            bits = int.from_bytes(payload[:22], byteorder='little')
            for i in range(16):
                self.rc_channels[i] = (bits >> (i * 11)) & 0x7FF
        elif frame_type == CRSF_FRAMETYPE_FLIGHT_MODE and len(payload) >= 1:
            self.flight_mode = bytes(payload).split(b'\x00')[0].decode('ascii', errors='ignore')
        else:
            if not hasattr(self, '_logged_types'):
                self._logged_types = set()
            if frame_type not in self._logged_types:
                self._logged_types.add(frame_type)
                log_crsf.debug("Unknown frame type=0x%02X len=%d payload=%s",
                               frame_type, len(payload), bytes(payload).hex())


class SerialReader(QThread):
    """Reads CRSF telemetry from RadioMaster serial port in background."""
    telemetry_updated = pyqtSignal(object)

    def __init__(self, port_name, baud=420000):
        super().__init__()
        self.port_name = port_name
        self.baud = baud
        self.running = False
        self.telemetry = CRSFTelemetry()

    def _find_sync(self, buf):
        """Find first CRSF sync byte (0xC8 or 0xEA)."""
        for i, b in enumerate(buf):
            if b == CRSF_SYNC_BYTE or b == CRSF_SYNC_BYTE_TX:
                return i
        return -1

    def _count_crsf_frames(self, data):
        """Count valid CRSF frames in a data buffer."""
        buf = bytearray(data)
        count = 0
        while len(buf) >= 4:
            idx = self._find_sync(buf)
            if idx < 0:
                break
            buf = buf[idx:]
            if len(buf) < 3:
                break
            frame_len = buf[1]
            if frame_len < 2 or frame_len > 62:
                buf = buf[1:]
                continue
            total_len = frame_len + 2
            if len(buf) < total_len:
                break
            count += 1
            buf = buf[total_len:]
        return count

    def run(self):
        self.running = True
        # RadioMaster TX USB serial: EdgeTX uses 400000 or 921600 depending on
        # version/config. ELRS passthrough uses 420000. Try all common rates.
        baud_rates = [400000, 420000, 921600, 115200, 57600]
        if self.baud not in baud_rates:
            baud_rates.insert(0, self.baud)
        elif baud_rates[0] != self.baud:
            baud_rates.remove(self.baud)
            baud_rates.insert(0, self.baud)

        ser = None
        best_baud = None
        best_frames = 0

        log_serial.info("Probing %s …", self.port_name)
        for baud in baud_rates:
            try:
                test_ser = serial.Serial(self.port_name, baud, timeout=1.5)
                # Flush stale data then read fresh
                test_ser.reset_input_buffer()
                time.sleep(0.1)
                test_data = test_ser.read(1024)
                n_bytes = len(test_data)
                n_frames = self._count_crsf_frames(test_data) if test_data else 0

                if n_frames > 0:
                    log_serial.info("%d baud: %d bytes, %d CRSF frame(s) ✓",
                                    baud, n_bytes, n_frames)
                    if n_frames > best_frames:
                        best_baud = baud
                        best_frames = n_frames
                elif test_data:
                    has_sync = any(b in (CRSF_SYNC_BYTE, CRSF_SYNC_BYTE_TX) for b in test_data)
                    log_serial.info("%d baud: %d bytes, sync=%s, 0 frames",
                                    baud, n_bytes, "yes" if has_sync else "no")
                    log_serial.debug("  raw: %s", test_data[:48].hex())
                else:
                    log_serial.info("%d baud: no data", baud)

                test_ser.close()

                # If we got good frames, no need to try more
                if best_frames >= 3:
                    break
            except Exception as e:
                log_serial.warning("%d baud failed: %s", baud, e)

        # Use best baud found, or fall back to user-specified
        chosen_baud = best_baud or self.baud
        if best_baud:
            log_serial.info("Using %d baud (%d CRSF frames detected)",
                            chosen_baud, best_frames)
        else:
            log_serial.warning("No CRSF detected — opening at %d baud anyway",
                               chosen_baud)

        try:
            ser = serial.Serial(self.port_name, chosen_baud, timeout=0.1)
            ser.reset_input_buffer()
        except Exception as e:
            log_serial.error("Failed to open port: %s", e)
            return

        try:
            buf = bytearray()
            frame_count = 0
            no_data_count = 0
            while self.running:
                data = ser.read(256)
                if not data:
                    no_data_count += 1
                    if no_data_count == 50:  # ~5 seconds of silence
                        log_serial.warning("No data from %s for ~5s", self.port_name)
                    continue
                no_data_count = 0
                buf.extend(data)

                while len(buf) >= 4:
                    idx = self._find_sync(buf)
                    if idx < 0:
                        if frame_count == 0 and len(buf) > 0:
                            log_serial.warning("No sync in %d bytes. Raw: %s",
                                               len(buf), buf[:48].hex())
                        buf.clear()
                        break
                    if idx > 0:
                        buf = buf[idx:]
                    if len(buf) < 3:
                        break
                    frame_len = buf[1]
                    if frame_len < 2 or frame_len > 62:
                        buf = buf[1:]
                        continue
                    total_len = frame_len + 2
                    if len(buf) < total_len:
                        break

                    frame_type = buf[2]
                    payload = buf[3:total_len - 1]
                    self.telemetry.parse_frame(frame_type, bytes(payload))
                    self.telemetry_updated.emit(self.telemetry)
                    frame_count += 1
                    if frame_count <= 20:
                        log_serial.debug("Frame #%d: type=0x%02X, len=%d",
                                         frame_count, frame_type, frame_len)
                    if frame_count == 21:
                        log_serial.info("Got %d frames, suppressing logs …",
                                        frame_count)
                    buf = buf[total_len:]

            ser.close()
        except Exception as e:
            log_serial.error("Error: %s", e)
            if ser and ser.is_open:
                ser.close()

    def stop(self):
        self.running = False
        self.wait(2000)
