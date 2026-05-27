#!/usr/bin/env python3
"""
Human Tracker Engine for Walle GCS.

Thin async wrapper around the existing ``human_tracker_v3/`` package
(YOLOv8-pose detection + DeepSort tracking + Kalman trajectory prediction +
ghost-frame occlusion handling + motion-aware physics arcs).

Designed to plug into the GCS video loop alongside the detection, depth, and
face-rec engines — inference runs in its own worker thread, ``process_frame``
returns in O(1) so video FPS is never gated by tracker speed.
"""

import os
import sys
import time
import threading

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from core.logging import get_logger

log = get_logger("HumanTrack")

# Make the existing tracker package importable. Its sub-modules use absolute
# imports rooted at ``config``, ``core``, ``ui`` — they only resolve when the
# tracker root is on sys.path.
# human_tracker_engine.py lives in walle_gcs/; the human_tracker_v3 package
# stays in the rosetta project root (shared with run_human_tracker_video.py).
_HT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "human_tracker_v3")
if _HT_DIR not in sys.path:
    sys.path.insert(0, _HT_DIR)


class HumanTrackerEngine(QObject):
    """Predictive human tracker — async, non-blocking.

    Emits ``models_loaded(bool)`` once YOLOv8-pose + DeepSort + predictor are
    ready. Call ``process_frame(bgr)`` from the GUI thread to get back an
    annotated frame; inference runs entirely on the worker thread.
    """

    models_loaded = pyqtSignal(bool)

    def __init__(self,
                 model_path: str = "yolov8n-pose.pt",
                 conf_threshold: float = 0.35,
                 predict_seconds: float = 1.0,
                 video_fps: float = 30.0):
        super().__init__()
        self._loaded = False
        self._loading = False
        self._lock = threading.Lock()

        # Settings
        self._model_path = model_path
        self._conf = conf_threshold
        self._predict_seconds = predict_seconds
        self._video_fps = video_fps

        # Heavy state (populated by load_models)
        self._model = None
        self._deepsort = None
        self._predictor = None
        self._motion_cls = None
        self._viz = None
        self._cfg = None
        self._device = None

        # Async inference worker
        self._infer_input_frame = None
        self._infer_pending = threading.Event()
        self._infer_stop = threading.Event()
        self._infer_thread = None

        # Latest per-track render snapshot built by the worker
        self._latest_render = []
        self._last_proc_ms = 0.0

        self._stats = {
            'active': 0,
            'ghost': 0,
            'peak_tracks': 0,
        }

    @property
    def is_ready(self):
        return self._loaded

    # ── Loading ───────────────────────────────────────────────────
    def load_models(self):
        if self._loaded or self._loading:
            return
        self._loading = True

        def _run():
            try:
                import torch
                from ultralytics import YOLO
                from deep_sort_realtime.deepsort_tracker import DeepSort
                from config.settings import TrackerConfig
                from core.motion_classifier import MotionClassifier
                from core.trajectory_predictor import TrajectoryPredictor
                from ui.visualizer import Visualizer

                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                log.info("Device: %s", self._device)

                # Force pose variant — keypoints feed the motion classifier
                mp = self._model_path
                if "pose" not in mp:
                    mp = mp.replace(".pt", "-pose.pt")
                log.info("Loading %s ...", mp)
                self._model = YOLO(mp)
                self._model.to(self._device)

                self._cfg = TrackerConfig(
                    SOURCE="0",
                    MODEL_PATH=mp,
                    CONF_THRESHOLD=self._conf,
                    PREDICTION_SECONDS=self._predict_seconds,
                    DEVICE=self._device,
                    SAVE_VIDEO=False,
                    DISPLAY_LIVE=False,
                )

                log.info("Initializing DeepSort + predictor ...")
                self._deepsort = DeepSort(
                    max_age=self._cfg.MAX_AGE,
                    n_init=self._cfg.N_INIT,
                    max_iou_distance=self._cfg.MAX_IOU_DISTANCE,
                    nn_budget=self._cfg.NN_BUDGET,
                    embedder="mobilenet",
                    half=(self._device == "cuda"),
                )

                self._predictor = TrajectoryPredictor(self._cfg)
                self._predictor.set_fps(self._video_fps)
                self._motion_cls = MotionClassifier(fps=self._video_fps)
                self._viz = Visualizer(self._cfg)

                # Start async worker
                self._infer_thread = threading.Thread(
                    target=self._inference_worker, daemon=True)
                self._infer_thread.start()

                self._loaded = True
                self._loading = False
                log.info("Ready")
                self.models_loaded.emit(True)
            except Exception as e:
                log.exception("Load failed: %s", e)
                self._loading = False
                self.models_loaded.emit(False)

        threading.Thread(target=_run, daemon=True).start()

    # ── Per-frame (NON-BLOCKING) ──────────────────────────────────
    def process_frame(self, frame):
        if not self._loaded or frame is None:
            return frame

        with self._lock:
            self._infer_input_frame = frame
        self._infer_pending.set()

        return self._draw(frame)

    def _inference_worker(self):
        """Background loop: YOLO pose → DeepSort → predictor → snapshot."""
        while not self._infer_stop.is_set():
            self._infer_pending.wait(timeout=0.5)
            self._infer_pending.clear()

            with self._lock:
                frame = self._infer_input_frame
                self._infer_input_frame = None
            if frame is None:
                continue

            t0 = time.perf_counter()
            try:
                results = self._model(
                    frame,
                    conf=self._cfg.CONF_THRESHOLD,
                    iou=self._cfg.IOU_THRESHOLD,
                    classes=[0],
                    verbose=False,
                )[0]

                dets = []
                for box in results.boxes:
                    if int(box.cls[0]) != 0:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    dets.append(([x1, y1, x2 - x1, y2 - y1],
                                 float(box.conf[0]), 0))

                tracks = self._deepsort.update_tracks(dets, frame=frame)

                confirmed_ids = set()
                id_to_bbox = {}
                id_to_kps = {}

                for tr in tracks:
                    if not tr.is_confirmed():
                        continue
                    tid = tr.track_id
                    bbox = tr.to_ltrb()
                    confirmed_ids.add(tid)
                    id_to_bbox[tid] = bbox
                    id_to_kps[tid] = self._get_keypoints_for_bbox(results, bbox)

                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2
                    cy = y1 + 0.30 * (y2 - y1)
                    self._predictor.update(tid, np.array([cx, cy]), bbox)

                # Ghost-step occluded
                known_ids = set(self._predictor.all_track_ids())
                for tid in known_ids - confirmed_ids:
                    self._predictor.ghost_step(tid)
                self._predictor.remove_dead()

                # Build render snapshot
                all_ids = set(self._predictor.all_track_ids())
                snapshot = []
                n_ghost = 0
                for tid in all_ids:
                    is_ghost = self._predictor.is_ghost(tid)
                    if is_ghost:
                        n_ghost += 1
                    history = self._predictor.get_history(tid)
                    velocity = self._predictor.get_velocity(tid)
                    speed = self._predictor.get_speed(tid)
                    kps = id_to_kps.get(tid)
                    hint = self._motion_cls.update(tid, speed, kps)
                    traj = self._predictor.predict_ahead(tid, hint=hint)
                    snapshot.append({
                        'tid': tid,
                        'is_ghost': is_ghost,
                        'history': history,
                        'velocity': velocity,
                        'traj': traj,
                        'bbox': id_to_bbox.get(tid),
                        'ghost_pos': (self._predictor.get_ghost_position(tid)
                                      if is_ghost else None),
                        'last_bbox': (self._predictor.get_last_bbox(tid)
                                      if is_ghost else None),
                    })

                self._motion_cls.remove_inactive(all_ids)

                with self._lock:
                    self._latest_render = snapshot
                    self._stats['active'] = len(confirmed_ids)
                    self._stats['ghost'] = n_ghost
                    if len(all_ids) > self._stats['peak_tracks']:
                        self._stats['peak_tracks'] = len(all_ids)

                self._last_proc_ms = (time.perf_counter() - t0) * 1000
            except Exception as e:
                log.error("worker error: %s", e)

    def _get_keypoints_for_bbox(self, results, bbox):
        if not hasattr(results, "keypoints") or results.keypoints is None:
            return None
        kp_data = results.keypoints.data
        if kp_data is None or len(kp_data) == 0:
            return None
        x1, y1, x2, y2 = bbox
        cx_bbox = (x1 + x2) / 2
        cy_bbox = (y1 + y2) / 2
        best_idx, best_dist = -1, float("inf")
        for i, kp in enumerate(kp_data):
            kp_np = kp.cpu().numpy()
            vis = kp_np[:, 2] > 0.25
            if vis.sum() == 0:
                continue
            cx = kp_np[vis, 0].mean()
            cy = kp_np[vis, 1].mean()
            d = abs(cx - cx_bbox) + abs(cy - cy_bbox)
            if d < best_dist:
                best_dist, best_idx = d, i
        return kp_data[best_idx].cpu().numpy() if best_idx >= 0 else None

    def _draw(self, frame):
        with self._lock:
            snap = list(self._latest_render)

        for e in snap:
            tid = e['tid']
            is_ghost = e['is_ghost']

            self._viz.draw_trail(frame, e['history'], tid, is_ghost)
            self._viz.draw_prediction(frame, e['traj'], tid, is_ghost)

            if is_ghost:
                gp = e['ghost_pos']
                lb = e['last_bbox']
                if gp is not None:
                    if lb is not None:
                        x1, y1, x2, y2 = lb
                        bw, bh = x2 - x1, y2 - y1
                        cx, cy = gp
                        shifted = (cx - bw / 2, cy - 0.30 * bh,
                                   cx + bw / 2, cy + 0.70 * bh)
                        self._viz.draw_person(frame, shifted, tid, is_ghost=True)
                    else:
                        self._viz.draw_ghost_dot(frame, gp, tid)
            else:
                bbox = e['bbox']
                if bbox is not None:
                    self._viz.draw_person(frame, bbox, tid)
                    x1, y1, x2, y2 = bbox
                    anchor = np.array([(x1 + x2) / 2, y1 + 0.30 * (y2 - y1)])
                    self._viz.draw_velocity(frame, anchor, e['velocity'], tid)

        return frame

    # ── External API ──────────────────────────────────────────────
    def get_stats(self):
        with self._lock:
            s = dict(self._stats)
        s['proc_ms'] = self._last_proc_ms
        s['proc_fps'] = (1000.0 / self._last_proc_ms) if self._last_proc_ms > 0 else 0.0
        return s

    def stop(self):
        self._infer_stop.set()
        self._infer_pending.set()
