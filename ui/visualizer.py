"""
Visualizer — clean, enterprise-grade overlay.
NO action labels. Focus: identity + trail + 1-second prediction arc + ghost.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from config.settings import TrackerConfig


class Visualizer:
    def __init__(self, config: TrackerConfig):
        self.cfg = config
        self._colors: Dict[int, Tuple[int, int, int]] = {}
        self._smooth_vel: Dict[int, np.ndarray] = {}   # EMA velocity direction per track

    # ------------------------------------------------------------------
    def _color(self, track_id: int) -> Tuple[int, int, int]:
        if track_id not in self._colors:
            np.random.seed(int(track_id) % 9999)
            r, g, b = np.random.randint(60, 230, 3).tolist()
            self._colors[track_id] = (b, g, r)   # OpenCV BGR
        return self._colors[track_id]

    # ------------------------------------------------------------------
    # Active person: solid box + ID badge
    # ------------------------------------------------------------------
    def draw_person(self, frame: np.ndarray,
                    bbox: Tuple, track_id: int,
                    is_ghost: bool = False) -> None:
        x1, y1, x2, y2 = map(int, bbox)
        color = self._color(track_id)

        if is_ghost:
            # Dashed / dimmed box for ghost
            self._draw_dashed_rect(frame, x1, y1, x2, y2, color, gap=10)
            label = f"#{track_id}  [occluded]"
            label_bg = (40, 40, 40)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"#{track_id}"
            label_bg = color

        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.rectangle(frame,
                      (x1, y1 - lh - 8), (x1 + lw + 8, y1),
                      label_bg, -1)
        cv2.putText(frame, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Past trajectory trail
    # ------------------------------------------------------------------
    def draw_trail(self, frame: np.ndarray,
                   history: np.ndarray, track_id: int,
                   is_ghost: bool = False) -> None:
        if not self.cfg.SHOW_TRAIL or len(history) < 2:
            return
        color = self._color(track_id)
        pts = history.astype(np.int32)
        n = len(pts)
        for i in range(n - 1):
            fade = 0.20 + 0.80 * (i / max(n - 1, 1))
            c = tuple(int(v * fade) for v in color)
            thickness = 1 if is_ghost else 2
            cv2.line(frame, tuple(pts[i]), tuple(pts[i + 1]),
                     c, thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # 1-second prediction arc
    # ------------------------------------------------------------------
    def draw_prediction(self, frame: np.ndarray,
                        trajectory: Optional[np.ndarray],
                        track_id: int,
                        is_ghost: bool = False) -> None:
        if not self.cfg.SHOW_PREDICTION:
            return
        if trajectory is None or len(trajectory) < 2:
            return

        color = self._color(track_id)
        pts = trajectory.reshape(-1, 2).astype(np.int32)
        n = len(pts)

        overlay = frame.copy()
        for i in range(n - 1):
            # Fade from bright → dim toward end
            intensity = 1.0 - 0.70 * (i / max(n - 1, 1))
            c = tuple(int(v * intensity) for v in color)
            # Thinner dashes if ghost
            cv2.line(overlay, tuple(pts[i]), tuple(pts[i + 1]),
                     c, 1 if is_ghost else 2, cv2.LINE_AA)

        alpha = 0.55 if not is_ghost else 0.30
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Endpoint marker — where the person will be in ~1 s
        end = tuple(pts[-1])
        cv2.circle(frame, end, 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, end, 9, (255, 255, 255), 1, cv2.LINE_AA)

        # Small "1s" label near endpoint
        cv2.putText(frame, "1s", (end[0] + 10, end[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    color, 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Ghost circle when bbox is unknown (fully occluded)
    # ------------------------------------------------------------------
    def draw_ghost_dot(self, frame: np.ndarray,
                       position: np.ndarray, track_id: int) -> None:
        if not self.cfg.SHOW_GHOST:
            return
        color = self._color(track_id)
        cx, cy = int(position[0]), int(position[1])
        # Pulsing semi-transparent circle
        overlay = frame.copy()
        cv2.circle(overlay, (cx, cy), 24, color, -1)
        cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
        cv2.circle(frame, (cx, cy), 24, color, 2, cv2.LINE_AA)
        # "?" mark
        cv2.putText(frame, "?", (cx - 6, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        label = f"#{track_id}"
        cv2.putText(frame, label, (cx + 28, cy + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    color, 2, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Velocity arrow  — stable, jitter-free
    # ------------------------------------------------------------------
    def draw_velocity(self, frame: np.ndarray,
                      anchor: np.ndarray,
                      velocity: Optional[np.ndarray],
                      track_id: int = -1) -> None:
        if not self.cfg.SHOW_VELOCITY or velocity is None:
            return

        speed = np.linalg.norm(velocity)

        # Dead-band: ignore sub-pixel noise (person essentially still)
        if speed < 0.6:
            # Erase any stored smoothed direction so it doesn't linger
            self._smooth_vel.pop(track_id, None)
            return

        # ── EMA direction smoothing ────────────────────────────────────
        # Blend the new unit-direction with the previous one.
        # This prevents the arrowhead from flipping due to single-frame
        # bbox jitter while still tracking real direction changes.
        unit = velocity / speed
        if track_id in self._smooth_vel:
            prev = self._smooth_vel[track_id]
            unit = 0.75 * prev + 0.25 * unit          # strong smoothing
            norm = np.linalg.norm(unit)
            if norm > 1e-6:
                unit = unit / norm
        self._smooth_vel[track_id] = unit

        # ── Arrow length: proportional to speed, hard-capped ──────────
        arrow_len = min(speed * 4.0, 80)               # px, max 80
        v_draw = unit * arrow_len

        start = (int(anchor[0]), int(anchor[1]))
        end   = (int(anchor[0] + v_draw[0]),
                 int(anchor[1] + v_draw[1]))

        col = (0, 200, 60) if speed < 2.0 else \
              (0, 190, 255) if speed < 6.0 else (0, 60, 230)

        cv2.arrowedLine(frame, start, end, col, 2, cv2.LINE_AA, tipLength=0.25)

    def remove_track(self, track_id: int) -> None:
        """Call when a track is dropped so we don't leak smoothing state."""
        self._smooth_vel.pop(track_id, None)
        self._colors.pop(track_id, None)

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------
    def draw_hud(self, frame: np.ndarray,
                 frame_no: int, fps: float,
                 n_active: int, n_ghost: int) -> None:
        lines = [
            f"Frame  : {frame_no}",
            f"FPS    : {fps:.1f}",
            f"Active : {n_active}",
            f"Ghost  : {n_ghost}",
        ]
        h = 22 * len(lines) + 14
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (220, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (16, 28 + 22 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (220, 220, 220), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _draw_dashed_rect(frame, x1, y1, x2, y2, color, gap=8):
        """Draw a dashed rectangle (for ghost state)."""
        pts = [
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ]
        for (ax, ay), (bx, by) in pts:
            dx, dy = bx - ax, by - ay
            length = max(int((dx**2 + dy**2) ** 0.5), 1)
            segments = max(length // (gap * 2), 1)
            for s in range(segments):
                t0 = s / segments
                t1 = (s + 0.5) / segments
                p0 = (int(ax + dx * t0), int(ay + dy * t0))
                p1 = (int(ax + dx * t1), int(ay + dy * t1))
                cv2.line(frame, p0, p1, color, 1, cv2.LINE_AA)
