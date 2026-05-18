"""
TrackerConfig — occlusion-robust tracking + physics-aware 1-second prediction.
"""

from dataclasses import dataclass


@dataclass
class TrackerConfig:
    # ── Source ────────────────────────────────────────────────────────
    SOURCE: str = "0"
    OUTPUT_PATH: str = "./output.mp4"
    MAX_SECONDS: int = 0
    SKIP_FRAMES: int = 0

    # ── Detection ─────────────────────────────────────────────────────
    # Use pose model — keypoints silently shape the prediction arc
    MODEL_PATH: str = "yolov8n-pose.pt"
    CONF_THRESHOLD: float = 0.35
    IOU_THRESHOLD: float = 0.50

    # ── DeepSort ──────────────────────────────────────────────────────
    MAX_AGE: int = 90
    N_INIT: int = 3
    MAX_IOU_DISTANCE: float = 0.70
    NN_BUDGET: int = 100

    # ── Kalman / Prediction ───────────────────────────────────────────
    PREDICTION_SECONDS: float = 1.0    # arc covers this many seconds ahead
    HISTORY_LEN: int = 45
    VELOCITY_SMOOTH: int = 12

    # ── Occlusion ─────────────────────────────────────────────────────
    GHOST_MAX_FRAMES: int = 60         # ~2 s at 30 fps

    # ── Visualisation ─────────────────────────────────────────────────
    SHOW_TRAIL: bool = True
    SHOW_GHOST: bool = True
    SHOW_PREDICTION: bool = True
    SHOW_VELOCITY: bool = True
    TRAIL_ALPHA: float = 0.50

    # ── Display window ────────────────────────────────────────────────
    DISPLAY_WIDTH:  int = 1440
    DISPLAY_HEIGHT: int = 860

    # ── Hardware / Output ─────────────────────────────────────────────
    DEVICE: str = "auto"
    DISPLAY_LIVE: bool = True
    SAVE_VIDEO: bool = True
    CODEC: str = "mp4v"
