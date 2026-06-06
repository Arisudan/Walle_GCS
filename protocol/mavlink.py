"""MAVLink-over-UDP link for the GCS.

Replaces the old CRSF-over-serial telemetry path and the
joystick -> ESP32-PPM control path. Telemetry now arrives as standard
MAVLink messages over a UDP link (QGroundControl-style), and commands
(arm/disarm, takeoff, land, RTL, mode change) go back out over the same
link as ``COMMAND_LONG`` / mode-set messages.

Exposes:

  * :class:`Telemetry` — plain data holder. It deliberately keeps the
    same field names the GUI already consumed from the old CRSF object
    (``battery_voltage``, ``link_quality``, ``pitch``, …) so the display
    code didn't need a field-by-field rename, plus an ``armed`` flag.
  * :class:`MAVLinkConnection` — a QThread that owns a pymavlink
    connection, parses incoming messages into a :class:`Telemetry`
    object, emits ``telemetry_updated`` on every update, and offers
    command helpers used by the on-screen buttons.
"""

from __future__ import annotations

import math
import time

from PyQt5.QtCore import QThread, pyqtSignal
from pymavlink import mavutil

from core.logging import get_logger


log_mav = get_logger("MAVLink")


class Telemetry:
    """Snapshot of vehicle state, populated from MAVLink messages.

    Field names match what the GUI already renders so the rest of the
    app is agnostic to whether the source is CRSF or MAVLink.
    """

    def __init__(self):
        # Power
        self.battery_voltage = 0.0
        self.battery_current = 0.0
        self.battery_mah = 0
        self.battery_percent = 0
        # Link (no real RSSI/LQ over UDP — link_quality is 100 while
        # heartbeats are arriving, 0 once they stop)
        self.rssi_1 = 0
        self.rssi_2 = 0
        self.link_quality = 0
        self.snr = 0
        self.rf_mode = 0
        self.tx_power = 0
        # Position
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
        self.vertical_speed = 0.0
        # Mode / arm
        self.flight_mode = ""
        self.armed = False


class MAVLinkConnection(QThread):
    """Owns a pymavlink UDP connection on its own thread.

    Parameters
    ----------
    host, port:
        Drone endpoint / local bind port.
    mode:
        ``"listen"`` (default) opens ``udpin:0.0.0.0:port`` — the GCS
        binds the port and waits for the vehicle/companion to push its
        MAVLink stream to it (the QGroundControl default, and what most
        companion SBCs do with ``--out udp:<GCS_IP>:port``). Outbound
        commands are sent to whichever address the stream arrives from.

        ``"connect"`` opens ``udpout:host:port`` — the GCS actively sends
        to ``host:port`` and receives replies on the same socket. Use this
        only when the vehicle side is a UDP server (e.g. some SITL setups)
        waiting for the GCS to initiate.
    """

    telemetry_updated = pyqtSignal(object)
    # Emitted True when the first vehicle heartbeat is seen, False on
    # link loss (no heartbeat for LINK_TIMEOUT seconds).
    link_changed = pyqtSignal(bool)

    LINK_TIMEOUT = 3.0  # seconds without a heartbeat before link is "lost"

    def __init__(self, host, port, mode="listen", source_system=255):
        super().__init__()
        self.host = host
        self.port = int(port)
        self.mode = mode
        self.source_system = source_system
        self.running = False
        self.master = None
        self.telemetry = Telemetry()
        # Defaults until the first heartbeat tells us the real ids.
        self.target_system = 1
        self.target_component = 1
        self._last_heartbeat = 0.0
        self._linked = False

    # ─── thread main loop ───────────────────────────────────────────
    def run(self):
        if self.mode == "connect":
            conn_str = f"udpout:{self.host}:{self.port}"
        else:
            # Listen mode: bind the port and let the vehicle push to us.
            conn_str = f"udpin:0.0.0.0:{self.port}"
        try:
            self.master = mavutil.mavlink_connection(
                conn_str, source_system=self.source_system)
        except Exception as e:
            log_mav.error("could not open %s: %s", conn_str, e)
            return

        log_mav.info("MAVLink link opened: %s", conn_str)
        self.running = True
        last_tx = 0.0

        while self.running:
            now = time.time()

            # Periodic GCS heartbeat. In connect (udpout) mode this also
            # tells the vehicle where to send its telemetry stream, so we
            # send from the start. In listen (udpin) mode we have no peer
            # address until the first packet arrives, so hold off until
            # then to avoid "no address" send errors.
            can_tx = self.mode == "connect" or self._last_heartbeat > 0
            if can_tx and now - last_tx > 1.0:
                self._send_gcs_heartbeat()
                last_tx = now

            try:
                msg = self.master.recv_match(blocking=True, timeout=0.5)
            except Exception as e:
                log_mav.error("recv error: %s", e)
                continue

            if msg is None:
                # No traffic — check for link loss.
                if (self._linked and self._last_heartbeat
                        and now - self._last_heartbeat > self.LINK_TIMEOUT):
                    self._linked = False
                    self.telemetry.link_quality = 0
                    self.link_changed.emit(False)
                    self.telemetry_updated.emit(self.telemetry)
                continue

            self._handle(msg, now)

        try:
            self.master.close()
        except Exception:
            pass
        log_mav.info("MAVLink link closed")

    def stop(self):
        self.running = False
        self.wait(2000)

    # ─── message handling ───────────────────────────────────────────
    def _handle(self, msg, now):
        mtype = msg.get_type()
        tel = self.telemetry
        emit = True

        if mtype == "HEARTBEAT":
            # Ignore our own GCS heartbeat echoed back on the socket.
            if msg.type == mavutil.mavlink.MAV_TYPE_GCS:
                return
            self.target_system = msg.get_srcSystem()
            self.target_component = msg.get_srcComponent()
            tel.armed = bool(
                msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            # mavutil decodes custom_mode -> mode string for us.
            tel.flight_mode = self.master.flightmode or ""
            tel.link_quality = 100
            self._last_heartbeat = now
            if not self._linked:
                self._linked = True
                self.link_changed.emit(True)

        elif mtype == "ATTITUDE":
            tel.roll = math.degrees(msg.roll)
            tel.pitch = math.degrees(msg.pitch)
            tel.yaw = math.degrees(msg.yaw)

        elif mtype == "GLOBAL_POSITION_INT":
            tel.latitude = msg.lat / 1e7
            tel.longitude = msg.lon / 1e7
            tel.altitude = msg.relative_alt / 1000.0  # mm -> m (AGL)
            if msg.hdg != 65535:
                tel.heading = msg.hdg / 100.0
            tel.vertical_speed = -msg.vz / 100.0  # cm/s down -> m/s up

        elif mtype == "VFR_HUD":
            tel.ground_speed = msg.groundspeed
            tel.heading = msg.heading
            tel.vertical_speed = msg.climb

        elif mtype == "SYS_STATUS":
            if msg.voltage_battery not in (0, 65535):
                tel.battery_voltage = msg.voltage_battery / 1000.0
            if msg.current_battery != -1:
                tel.battery_current = msg.current_battery / 100.0
            if msg.battery_remaining != -1:
                tel.battery_percent = msg.battery_remaining

        elif mtype == "BATTERY_STATUS":
            if msg.current_battery != -1:
                tel.battery_current = msg.current_battery / 100.0
            if msg.battery_remaining != -1:
                tel.battery_percent = msg.battery_remaining
            if getattr(msg, "current_consumed", -1) != -1:
                tel.battery_mah = msg.current_consumed

        elif mtype == "GPS_RAW_INT":
            tel.satellites = msg.satellites_visible

        elif mtype == "RADIO_STATUS":
            # Only present when a telemetry radio sits in the path.
            tel.rssi_1 = msg.rssi
            tel.rssi_2 = msg.remrssi
            tel.snr = getattr(msg, "noise", 0)

        else:
            emit = False

        if emit:
            self.telemetry_updated.emit(tel)

    # ─── outbound commands ──────────────────────────────────────────
    def _ready(self):
        if not (self.running and self.master is not None):
            log_mav.warning("command ignored: link not connected")
            return False
        return True

    def _send_gcs_heartbeat(self):
        try:
            self.master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        except Exception as e:
            log_mav.error("heartbeat send failed: %s", e)

    def _command_long(self, command, *params):
        params = list(params) + [0.0] * (7 - len(params))
        try:
            self.master.mav.command_long_send(
                self.target_system, self.target_component,
                command, 0, *params[:7])
        except Exception as e:
            log_mav.error("command %s failed: %s", command, e)

    def arm(self, arm=True):
        if not self._ready():
            return
        log_mav.info("%s", "ARM" if arm else "DISARM")
        self._command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            1 if arm else 0)

    def emergency_stop(self):
        """Force-disarm immediately (kill motors), bypassing checks."""
        if not self._ready():
            return
        log_mav.warning("EMERGENCY STOP - force disarm")
        # param2 = 21196 is the ArduPilot/PX4 "force" magic number.
        self._command_long(
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 21196)

    def takeoff(self, altitude=10.0):
        if not self._ready():
            return
        log_mav.info("TAKEOFF to %.1f m", altitude)
        self._command_long(
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, altitude)

    def land(self):
        if not self._ready():
            return
        log_mav.info("LAND")
        self._command_long(mavutil.mavlink.MAV_CMD_NAV_LAND)

    def return_to_launch(self):
        if not self._ready():
            return
        log_mav.info("RTL")
        self._command_long(mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH)

    def set_mode(self, mode_name):
        if not self._ready():
            return
        mode_name = str(mode_name).upper()
        mapping = self.master.mode_mapping()
        if not mapping:
            log_mav.warning("mode map unavailable (no heartbeat yet)")
            return
        if mode_name not in mapping:
            log_mav.warning("unknown flight mode %r for this autopilot", mode_name)
            return
        log_mav.info("set mode %s", mode_name)
        self.master.set_mode(mapping[mode_name])
