"""
TrajectoryPredictor
===================
Per-track Kalman filters + occlusion ghost tracking.
predict_ahead() accepts a MotionHint to physically shape the arc.
"""

from __future__ import annotations

from collections import deque
from enum import Enum, auto
from typing import Dict, Optional

import numpy as np

from config.settings import TrackerConfig
from core.kalman_tracker import KalmanTracker


class TrackState(Enum):
    ACTIVE = auto()
    GHOST  = auto()
    DEAD   = auto()


class TrackInfo:
    __slots__ = ("kf", "state", "ghost_frames",
                 "history", "velocities", "last_bbox")

    def __init__(self, pos: np.ndarray, history_len: int, vel_smooth: int):
        self.kf           = KalmanTracker()
        self.kf.initialise(pos)
        self.state        = TrackState.ACTIVE
        self.ghost_frames = 0
        self.history      : deque = deque(maxlen=history_len)
        self.velocities   : deque = deque(maxlen=vel_smooth)
        self.last_bbox    : Optional[np.ndarray] = None
        self.history.append(pos.copy())


class TrajectoryPredictor:
    def __init__(self, config: TrackerConfig):
        self.cfg    = config
        self._tracks: Dict[int, TrackInfo] = {}
        self._prediction_steps = 30

    def set_fps(self, fps: float) -> None:
        self._prediction_steps = max(
            1, int(round(self.cfg.PREDICTION_SECONDS * fps)))

    # ------------------------------------------------------------------
    def update(self, track_id: int,
               position: np.ndarray,
               bbox: Optional[np.ndarray] = None) -> None:
        pos = np.array(position, dtype=float).flatten()[:2]
        if not np.isfinite(pos).all():
            return

        if track_id not in self._tracks:
            self._tracks[track_id] = TrackInfo(
                pos, self.cfg.HISTORY_LEN, self.cfg.VELOCITY_SMOOTH)
        else:
            info = self._tracks[track_id]
            if len(info.history) >= 1:
                info.velocities.append(pos - info.history[-1])
            info.kf.update(pos)
            info.history.append(pos.copy())
            info.state        = TrackState.ACTIVE
            info.ghost_frames = 0
            if bbox is not None:
                info.last_bbox = bbox

    # ------------------------------------------------------------------
    def ghost_step(self, track_id: int) -> Optional[np.ndarray]:
        if track_id not in self._tracks:
            return None
        info = self._tracks[track_id]
        info.ghost_frames += 1
        if info.ghost_frames > self.cfg.GHOST_MAX_FRAMES:
            info.state = TrackState.DEAD
            return None
        info.state = TrackState.GHOST
        ghost_pos  = info.kf.step_predict()
        info.history.append(ghost_pos.copy())
        return ghost_pos

    # ------------------------------------------------------------------
    def predict_ahead(self, track_id: int,
                      hint=None) -> Optional[np.ndarray]:
        """Returns (steps, 2) future positions, shaped by hint."""
        info = self._tracks.get(track_id)
        if info is None or len(info.history) < 3:
            return None
        return info.kf.predict_ahead(self._prediction_steps, hint=hint)

    # ------------------------------------------------------------------
    def get_velocity(self, track_id: int) -> Optional[np.ndarray]:
        info = self._tracks.get(track_id)
        if info is None:
            return None
        kf_vel = info.kf.velocity
        if np.isfinite(kf_vel).all() and len(info.history) >= 3:
            return kf_vel
        if len(info.velocities):
            return np.mean(list(info.velocities), axis=0)
        return None

    def get_speed(self, track_id: int) -> float:
        v = self.get_velocity(track_id)
        return float(np.linalg.norm(v)) if v is not None else 0.0

    def get_history(self, track_id: int) -> np.ndarray:
        info = self._tracks.get(track_id)
        if info and len(info.history):
            return np.array(list(info.history))
        return np.empty((0, 2))

    def get_state(self, track_id: int) -> Optional[TrackState]:
        return self._tracks[track_id].state if track_id in self._tracks else None

    def get_ghost_position(self, track_id: int) -> Optional[np.ndarray]:
        info = self._tracks.get(track_id)
        if info and info.state == TrackState.GHOST and len(info.history):
            return info.history[-1]
        return None

    def get_last_bbox(self, track_id: int) -> Optional[np.ndarray]:
        info = self._tracks.get(track_id)
        return info.last_bbox if info else None

    def all_track_ids(self):
        return list(self._tracks.keys())

    def is_ghost(self, track_id: int) -> bool:
        info = self._tracks.get(track_id)
        return info is not None and info.state == TrackState.GHOST

    def remove_dead(self) -> None:
        dead = [t for t, i in self._tracks.items() if i.state == TrackState.DEAD]
        for tid in dead:
            del self._tracks[tid]

    def remove_inactive(self, active_ids) -> None:
        remove = set(self._tracks.keys()) - set(active_ids)
        for tid in remove:
            del self._tracks[tid]
