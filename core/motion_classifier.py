"""
MotionClassifier
================
Silently classifies motion state from pose keypoints + speed.
Returns a MotionHint that the TrajectoryPredictor uses to shape
the Kalman prediction — NO labels are ever shown on screen.

Motion states (internal only):
  STATIONARY  – barely moving; damp the prediction arc to near-zero
  WALKING     – moderate speed; project forward along current heading
  RUNNING     – high speed; project further, trust velocity more
  FALLING     – nose below hips; add downward gravity component
  CROUCHING   – compressed body; slow/stop prediction
  SITTING     – knee close to hip; zero out prediction

COCO keypoint indices used:
  0  nose
  5  left_shoulder   6  right_shoulder
  11 left_hip        12 right_hip
  13 left_knee       14 right_knee
  15 left_ankle      16 right_ankle
   9 left_wrist      10 right_wrist
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


# ── Keypoint index aliases ─────────────────────────────────────────────
NOSE          = 0
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP,      R_HIP      = 11, 12
L_KNEE,     R_KNEE     = 13, 14
L_ANKLE,    R_ANKLE    = 15, 16
L_WRIST,    R_WRIST    =  9, 10


@dataclass
class MotionHint:
    """
    Returned each frame per track. Shapes the Kalman prediction.
    All fields are used internally — nothing is ever printed/drawn.
    """
    # Multiplier on the prediction arc length: 0=no arc, 1=normal, >1=longer
    arc_scale: float = 1.0

    # Extra per-step velocity to inject (e.g. gravity for falling)
    # Shape (2,) in pixel/frame units
    gravity: np.ndarray = None

    # Damp acceleration term (0=ignore accel, 1=full accel)
    accel_damp: float = 1.0

    def __post_init__(self):
        if self.gravity is None:
            self.gravity = np.zeros(2)


class MotionClassifier:
    """
    Per-track pose + speed analyser.
    Call update() every frame, get a MotionHint back.
    """

    # Speed thresholds in px/frame (at ~30 fps)
    _WALK_THRESH = 1.8
    _RUN_THRESH  = 5.0

    def __init__(self, fps: float = 30.0, smooth: int = 8):
        self.fps    = fps
        self._hints: Dict[int, deque] = {}          # EMA of arc_scale per track
        self._smooth = smooth

    # ------------------------------------------------------------------
    def update(self,
               track_id: int,
               speed_px_frame: float,
               keypoints: Optional[np.ndarray]      # (17, 3) or None
               ) -> MotionHint:

        hint = self._classify(speed_px_frame, keypoints)

        # Smooth arc_scale with a short EMA so it doesn't jump frame-to-frame
        if track_id not in self._hints:
            self._hints[track_id] = deque(maxlen=self._smooth)
        self._hints[track_id].append(hint.arc_scale)
        hint.arc_scale = float(np.mean(self._hints[track_id]))

        return hint

    # ------------------------------------------------------------------
    def _classify(self,
                  speed: float,
                  kps: Optional[np.ndarray]) -> MotionHint:

        # ── No pose available → fall back to pure speed ────────────────
        if kps is None or len(kps) < 17:
            return self._speed_only(speed)

        def vis(i):    return kps[i, 2] > 0.28
        def pt(i):     return kps[i, :2]
        def mid(a, b): return (pt(a) + pt(b)) / 2

        # ── FALLING: nose below midpoint of hips ───────────────────────
        if vis(NOSE) and vis(L_HIP) and vis(R_HIP):
            hip_y  = mid(L_HIP, R_HIP)[1]
            nose_y = pt(NOSE)[1]
            if nose_y > hip_y + 15:
                # Strong downward gravity component, extend arc
                g = np.array([0.0, 2.5])      # px/frame² downward
                return MotionHint(arc_scale=1.4, gravity=g, accel_damp=0.3)

        # ── SITTING: knee Y ≈ hip Y (small vertical gap) ───────────────
        if vis(L_KNEE) and vis(R_KNEE) and vis(L_HIP) and vis(R_HIP):
            knee_y = mid(L_KNEE, R_KNEE)[1]
            hip_y  = mid(L_HIP,  R_HIP )[1]
            if abs(knee_y - hip_y) < 35:
                return MotionHint(arc_scale=0.05, accel_damp=0.0)

        # ── CROUCHING: body height (nose→ankle) is short ───────────────
        if vis(NOSE) and vis(L_ANKLE) and vis(R_ANKLE):
            ankle_y = mid(L_ANKLE, R_ANKLE)[1]
            h_px    = abs(ankle_y - pt(NOSE)[1])
            if h_px < 110:
                return MotionHint(arc_scale=0.30, accel_damp=0.2)

        # ── Default: speed-based ───────────────────────────────────────
        return self._speed_only(speed)

    # ------------------------------------------------------------------
    @staticmethod
    def _speed_only(speed: float) -> MotionHint:
        if speed < 0.5:
            # Stationary — tiny arc just shows "staying here"
            return MotionHint(arc_scale=0.08, accel_damp=0.0)
        elif speed < MotionClassifier._WALK_THRESH:
            return MotionHint(arc_scale=0.65, accel_damp=0.5)
        elif speed < MotionClassifier._RUN_THRESH:
            return MotionHint(arc_scale=1.0,  accel_damp=0.8)
        else:
            # Running — full arc, trust the acceleration
            return MotionHint(arc_scale=1.3,  accel_damp=1.0)

    # ------------------------------------------------------------------
    def remove_inactive(self, active_ids) -> None:
        dead = set(self._hints.keys()) - set(active_ids)
        for tid in dead:
            del self._hints[tid]
