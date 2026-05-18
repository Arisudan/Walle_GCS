"""
KalmanTracker — 6-state constant-acceleration model.
State:   [x, y, vx, vy, ax, ay]
Measure: [x, y]

predict_ahead() now accepts an optional MotionHint that:
  • scales the arc length (arc_scale)
  • injects a per-step gravity offset (gravity)
  • damps the acceleration term (accel_damp)
This shapes the prediction physically without any visible labels.
"""

import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanTracker:
    def __init__(self, process_noise: float = 0.08,
                 measure_noise: float = 8.0):
        self.kf = KalmanFilter(dim_x=6, dim_z=2)
        dt = 1.0

        self.kf.F = np.array([
            [1, 0, dt, 0,  0.5*dt**2, 0        ],
            [0, 1, 0,  dt, 0,         0.5*dt**2],
            [0, 0, 1,  0,  dt,        0        ],
            [0, 0, 0,  1,  0,         dt       ],
            [0, 0, 0,  0,  1,         0        ],
            [0, 0, 0,  0,  0,         1        ],
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ])
        self.kf.R *= measure_noise
        self.kf.P *= 1000.0
        self.kf.Q *= process_noise
        self._initialised = False

    # ------------------------------------------------------------------
    def initialise(self, pos: np.ndarray) -> None:
        self.kf.x[:2] = np.array(pos, dtype=float).reshape(2, 1)
        self._initialised = True

    def update(self, pos: np.ndarray) -> np.ndarray:
        self.kf.predict()
        self.kf.update(np.array(pos, dtype=float).reshape(2, 1))
        return self.kf.x[:2].flatten()

    def step_predict(self) -> np.ndarray:
        """One ghost step — advances live state."""
        self.kf.predict()
        return self.kf.x[:2].flatten()

    # ------------------------------------------------------------------
    def predict_ahead(self, steps: int, hint=None) -> np.ndarray:
        """
        Read-ahead N steps. hint is an optional MotionHint.
        Returns (steps, 2).
        """
        x_bak = self.kf.x.copy()
        P_bak = self.kf.P.copy()

        # Extract live velocity + acceleration
        vx, vy = float(self.kf.x[2]), float(self.kf.x[3])
        ax, ay = float(self.kf.x[4]), float(self.kf.x[5])

        arc_scale  = 1.0
        gravity    = np.zeros(2)
        accel_damp = 1.0
        if hint is not None:
            arc_scale  = float(hint.arc_scale)
            gravity    = np.array(hint.gravity, dtype=float)
            accel_damp = float(hint.accel_damp)

        # Build a modified state to predict from
        x_mod = self.kf.x.copy()
        x_mod[2] = vx * arc_scale          # scale velocity component
        x_mod[3] = vy * arc_scale
        x_mod[4] = ax * accel_damp         # optionally damp acceleration
        x_mod[5] = ay * accel_damp

        # Temporarily swap state
        self.kf.x = x_mod

        preds = []
        for _ in range(steps):
            self.kf.predict()
            p = self.kf.x[:2].flatten().copy()
            # Inject gravity (cumulative per step)
            p += gravity * (_ + 1)
            preds.append(p)

        # Restore live state — prediction never corrupts it
        self.kf.x = x_bak
        self.kf.P = P_bak
        return np.array(preds)   # (steps, 2)

    # ------------------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.kf.x[:2].flatten()

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.x[2:4].flatten()
