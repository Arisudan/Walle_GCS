"""SLAM engine for Walle GCS.

Wraps the GPU 2D SLAM pipeline (Depth Anything V2 + ORB keyframe VO +
GPU log-odds occupancy grid) into a Qt-friendly engine that:

  * loads heavy models in a background thread
  * accepts BGR frames via process_frame() from the main thread
    (drops if a worker is busy, so the GUI never stalls)
  * emits update_ready(fpv_bgr, map_bgr, stats_dict) on each
    finished frame, deliverable to QLabel widgets

The OccupancyGrid2D / DepthAnythingV2Wrap / KeyframeVO classes are
vendored from gpu_2d_slam.py + MASt3R-SLAM/mast3r_slam/occupancy_2d.py
(class only — no lietorch / CUDA-backend dependency).

Frame convention: cam = +X right, +Y down, +Z forward. World = first
cam frame. Ground plane = world XZ. Floor auto-detected.
"""

from __future__ import annotations

import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import torch
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)


# ─── Constants ──────────────────────────────────────────────────────

DEPTH_MODELS_SLAM = {
    "indoor-small":  "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
    "indoor-base":   "depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf",
    "outdoor-small": "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf",
    "outdoor-base":  "depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf",
}


# ─── Depth Anything V2 wrapper ──────────────────────────────────────

class DepthAnythingV2Wrap:
    """DA V2 metric depth, GPU preprocessing path. fp16 inference."""

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, model_id, device="cuda", dtype=torch.float16,
                 input_size=322):
        from transformers import AutoModelForDepthEstimation
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(model_id)
            .to(device, dtype=dtype)
            .eval()
        )
        self.device = device
        self.dtype = dtype
        self.input_size = max(14, (input_size // 14) * 14)
        self.mean = torch.tensor(self.IMAGENET_MEAN, device=device,
                                 dtype=dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(self.IMAGENET_STD, device=device,
                                dtype=dtype).view(1, 3, 1, 1)
        self.depth_scale = 1.0

    @torch.inference_mode()
    def __call__(self, img_rgb_u8):
        H, W = img_rgb_u8.shape[:2]
        x = torch.from_numpy(img_rgb_u8).to(self.device, non_blocking=True)
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.dtype) / 255.0
        x = torch.nn.functional.interpolate(
            x, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False,
        )
        x = (x - self.mean) / self.std
        out = self.model(pixel_values=x)
        depth = out.predicted_depth
        depth = torch.nn.functional.interpolate(
            depth[:, None].float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        if self.depth_scale != 1.0:
            depth = depth * self.depth_scale
        return depth


def make_intrinsics(W, H, hfov_deg):
    fx = W / (2.0 * np.tan(np.radians(hfov_deg) / 2.0))
    fy = fx
    return np.array([[fx, 0, W / 2.0], [0, fy, H / 2.0], [0, 0, 1]],
                    dtype=np.float32)


def unproject_depth(depth_t, K, stride=4, dmin=0.3, dmax=12.0, mask=None):
    H, W = depth_t.shape
    device = depth_t.device
    ds = depth_t[::stride, ::stride]
    Hs, Ws = ds.shape
    us = torch.arange(0, W, stride, device=device, dtype=torch.float32)[:Ws]
    vs = torch.arange(0, H, stride, device=device, dtype=torch.float32)[:Hs]
    vv, uu = torch.meshgrid(vs, us, indexing="ij")
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    valid = (ds > dmin) & (ds < dmax)
    if mask is not None:
        m_sub = mask[::stride, ::stride][:Hs, :Ws]
        valid = valid & m_sub
    x = (uu - cx) / fx * ds
    y = (vv - cy) / fy * ds
    z = ds
    pts = torch.stack([x, y, z], dim=-1)[valid]
    return pts


def transform_points(pts, T):
    R = T[:3, :3]
    t = T[:3, 3]
    return pts @ R.T + t


# ─── Edge / corner masks for occupancy gating ────────────────────────

def _corners_fast(gray_u8, threshold=20, dilate_px=2):
    fast = cv2.FastFeatureDetector_create(
        threshold=int(threshold), nonmaxSuppression=True)
    kps = fast.detect(gray_u8, None)
    mask = np.zeros_like(gray_u8, dtype=np.uint8)
    if not kps:
        return mask.astype(bool)
    H, W = mask.shape
    pts = np.array([(int(kp.pt[0]), int(kp.pt[1])) for kp in kps],
                   dtype=np.int32)
    pts = pts[(pts[:, 0] >= 0) & (pts[:, 0] < W)
              & (pts[:, 1] >= 0) & (pts[:, 1] < H)]
    mask[pts[:, 1], pts[:, 0]] = 255
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kern, iterations=1)
    return mask > 0


def _corners_harris(gray_u8, block_size=2, ksize=3, k=0.04,
                    thresh_rel=0.01, dilate_px=2):
    h = cv2.cornerHarris(gray_u8.astype(np.float32),
                         int(block_size), int(ksize), float(k))
    if h.size == 0 or float(h.max()) <= 0.0:
        return np.zeros_like(gray_u8, dtype=bool)
    mask = (h > float(thresh_rel) * float(h.max())).astype(np.uint8) * 255
    if dilate_px > 0:
        kdim = 2 * int(dilate_px) + 1
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kdim, kdim))
        mask = cv2.dilate(mask, kern, iterations=1)
    return mask > 0


def compute_corner_mask(gray_u8, dilate=1, fast_thresh=20):
    e1 = _corners_fast(gray_u8, threshold=fast_thresh)
    e2 = _corners_harris(gray_u8)
    e = e1 | e2
    if dilate > 0:
        e_u8 = e.astype(np.uint8) * 255
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        e_u8 = cv2.dilate(e_u8, kern, iterations=int(dilate))
        e = e_u8 > 0
    return e


def overlay_edges(bgr, edge_mask, color=(0, 255, 255)):
    if edge_mask is None:
        return bgr
    out = bgr.copy()
    out[edge_mask] = (
        0.3 * out[edge_mask] + 0.7 * np.array(color, dtype=np.float32)
    ).astype(np.uint8)
    return out


# ─── Keyframe ORB VO ─────────────────────────────────────────────────

class KeyframeVO:
    """Keyframe-anchored ORB+PnP visual odometry. Vendored from
    gpu_2d_slam.py; see that file for the design notes."""

    def __init__(self, K, orb, lowe_ratio=0.75, min_matches=25,
                 max_matches=600, reproj_err_px=2.0,
                 max_trans_m=0.6, max_rot_deg=30.0,
                 kf_trans_m=0.35, kf_rot_deg=25.0,
                 static_pix_thresh=0.6,
                 klt_min_tracks=40, klt_win=31, klt_levels=4,
                 parallax_thresh_px=1.5):
        self.K = K
        self.orb = orb
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.lowe_ratio = lowe_ratio
        self.min_matches = min_matches
        self.max_matches = max_matches
        self.reproj_err_px = reproj_err_px
        self.max_trans_m = max_trans_m
        self.max_rot_deg = max_rot_deg
        self.kf_trans_m = kf_trans_m
        self.kf_rot_deg = kf_rot_deg
        self.static_pix_thresh = static_pix_thresh

        # KLT (pyramidal Lucas-Kanade) optical-flow params for
        # frame-to-frame tracking of the keyframe's keypoints.
        self.klt_min_tracks = int(klt_min_tracks)
        self._klt_lk_params = dict(
            winSize=(int(klt_win), int(klt_win)),
            maxLevel=int(klt_levels),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      20, 0.03),
            minEigThreshold=1e-4,
        )

        self.kf_kps = None
        self.kf_des = None
        self.kf_depth_np = None
        self.kf_gray = None
        self.kf_T_world = np.eye(4, dtype=np.float32)
        self.kf_id = 0
        self.T_world_curr = np.eye(4, dtype=np.float32)
        self.prev_T_world_curr = np.eye(4, dtype=np.float32)

        # Parallax threshold (median pixel residual after de-rotation,
        # in pixels) below which we treat the inter-frame motion as
        # pure rotation and lock translation.
        self.parallax_thresh_px = float(parallax_thresh_px)
        # Last accepted body-frame translation delta (t_{prev→curr}),
        # used as the PnP translation seed only when parallax is high.
        self.last_t_delta = np.zeros(3, dtype=np.float32)

        # KLT state carried frame-to-frame.
        self.prev_gray = None
        self.tracked_uv = None      # (M, 1, 2) float32 — current 2D positions
        self.tracked_pts3d = None   # (M, 3) float32 — 3D in keyframe cam frame

    def _set_keyframe(self, kps, des, depth_np, T_world_curr, gray=None):
        self.kf_kps = kps
        self.kf_des = des
        self.kf_depth_np = depth_np
        self.kf_T_world = T_world_curr.copy()
        self.kf_id += 1
        # Drop the carried body-frame velocity: the new keyframe is a
        # fresh reference and seeding PnP with a stale t-delta hurts.
        self.last_t_delta = np.zeros(3, dtype=np.float32)

        # Pre-unproject keyframe keypoints to 3D in camera frame and
        # seed the KLT tracker. From here on those points move
        # frame-to-frame via optical flow until the next keyframe.
        self.kf_gray = gray
        self.tracked_uv = None
        self.tracked_pts3d = None
        if gray is not None and kps is not None and depth_np is not None \
                and len(kps) > 0:
            Hd, Wd = depth_np.shape
            K = self.K
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            uv = np.array([kp.pt for kp in kps], dtype=np.float32)
            u, v = uv[:, 0], uv[:, 1]
            ui = np.clip(np.round(u).astype(np.int32), 0, Wd - 1)
            vi = np.clip(np.round(v).astype(np.int32), 0, Hd - 1)
            d = depth_np[vi, ui].astype(np.float32)
            ok = np.isfinite(d) & (d > 0.3) & (d < 50.0)
            if int(ok.sum()) >= 4:
                u, v, d = u[ok], v[ok], d[ok]
                x = (u - cx) / fx * d
                y = (v - cy) / fy * d
                self.tracked_pts3d = np.stack([x, y, d], axis=1).astype(np.float32)
                self.tracked_uv = np.stack([u, v], axis=1) \
                    .reshape(-1, 1, 2).astype(np.float32)
            self.prev_gray = gray

    def _ratio_match(self, kf_des, curr_des):
        if kf_des is None or curr_des is None:
            return []
        try:
            knn = self.matcher.knnMatch(kf_des, curr_des, k=2)
        except cv2.error:
            return []
        out = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n_ = pair
            if m.distance < self.lowe_ratio * n_.distance:
                out.append(m)
        out.sort(key=lambda m: m.distance)
        return out[: self.max_matches]

    def _klt_track(self, gray):
        """KLT-track keyframe points from prev_gray to gray.

        Mutates self.tracked_uv / self.tracked_pts3d to keep only the
        surviving tracks. Returns (pts3d_kf_cam, pts2d_current,
        prev_pts2d_current) for survivors, or (None, None, None) if
        we don't have enough state to track."""
        if (self.prev_gray is None or self.tracked_uv is None
                or len(self.tracked_uv) < 4):
            return None, None, None
        prev_uv = self.tracked_uv.reshape(-1, 2).copy()
        next_uv, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.tracked_uv, None,
            **self._klt_lk_params)
        if next_uv is None or status is None:
            self.tracked_uv = None
            self.tracked_pts3d = None
            return None, None, None
        status = status.flatten().astype(bool)
        uv2 = next_uv.reshape(-1, 2)
        Hh, Ww = gray.shape[:2]
        inb = ((uv2[:, 0] >= 0) & (uv2[:, 0] < Ww)
               & (uv2[:, 1] >= 0) & (uv2[:, 1] < Hh))
        keep = status & inb
        if int(keep.sum()) < 4:
            self.tracked_uv = None
            self.tracked_pts3d = None
            return None, None, None
        self.tracked_uv = next_uv[keep]
        self.tracked_pts3d = self.tracked_pts3d[keep]
        pts2d = self.tracked_uv.reshape(-1, 2).astype(np.float32)
        pts3d = self.tracked_pts3d.astype(np.float32)
        prev_pts2d = prev_uv[keep].astype(np.float32)
        return pts3d, pts2d, prev_pts2d

    def _interframe_motion(self, prev_pts2d, curr_pts2d):
        """Estimate inter-frame rotation R (prev -> curr) and parallax
        from two-view correspondences using a 5-point essential matrix.

        Parallax is the median pixel residual after de-rotating prev
        points by R: ~0 px under pure rotation, grows with translation.
        Returns (R_3x3_float32, parallax_px, n_inliers) or
        (None, +inf, 0) if estimation fails."""
        n = len(prev_pts2d)
        if n < 8:
            return None, float("inf"), 0
        K = self.K
        try:
            E, mask = cv2.findEssentialMat(
                prev_pts2d, curr_pts2d, K,
                method=cv2.RANSAC, prob=0.999, threshold=1.0)
        except cv2.error:
            return None, float("inf"), 0
        if E is None or mask is None or int(mask.sum()) < 8:
            return None, float("inf"), 0
        try:
            n_pose, R, _t_dir, _mask_pose = cv2.recoverPose(
                E, prev_pts2d, curr_pts2d, K, mask=mask.copy())
        except cv2.error:
            return None, float("inf"), 0
        if n_pose < 8:
            return None, float("inf"), 0
        # De-rotate prev points by R and measure pixel residual to curr.
        K_inv = np.linalg.inv(K).astype(np.float64)
        prev_h = np.hstack([prev_pts2d.astype(np.float64),
                            np.ones((n, 1))])                  # (N,3)
        rays = (K_inv @ prev_h.T).T                            # (N,3)
        rays_rot = (R @ rays.T).T                              # (N,3)
        z = rays_rot[:, 2:3]
        z = np.where(np.abs(z) < 1e-6, np.sign(z) * 1e-6 + 1e-12, z)
        proj = (K @ (rays_rot / z).T).T[:, :2]
        res = np.linalg.norm(proj - curr_pts2d.astype(np.float64), axis=1)
        # Use only inliers from the essential-matrix RANSAC for parallax.
        m = mask.ravel().astype(bool)
        parallax = float(np.median(res[m])) if int(m.sum()) > 0 \
            else float(np.median(res))
        return R.astype(np.float32), parallax, int(n_pose)

    def process(self, gray, depth_np):
        curr_kps, curr_des = self.orb.detectAndCompute(gray, None)
        info = {
            "status": "ok", "n_matches": 0, "n_inliers": 0,
            "kf_id": self.kf_id, "delta_trans_m": 0.0,
            "delta_rot_deg": 0.0, "static": False,
            "tracker": "klt",
        }

        if self.kf_kps is None:
            self._set_keyframe(curr_kps, curr_des, depth_np,
                               np.eye(4, dtype=np.float32), gray=gray)
            info["status"] = "init"
            return self.T_world_curr, info

        # --- Primary: KLT-track keyframe points from prev frame -------
        pts3d_klt, pts2d_klt, prev_pts2d_klt = self._klt_track(gray)
        used_klt = pts3d_klt is not None and len(pts3d_klt) >= self.klt_min_tracks

        # Inter-frame motion estimate (R_meas: prev_cam -> curr_cam,
        # parallax: median de-rotation residual in px). Populated only
        # when KLT gave us enough correspondences.
        R_meas = None
        parallax_px = float("inf")
        n_pose_in = 0
        info["parallax_px"] = parallax_px
        info["rot_only"] = False

        pts3d = pts2d = None
        if used_klt:
            info["tracker"] = "klt"
            info["n_matches"] = int(len(pts3d_klt))
            pts3d = pts3d_klt
            pts2d = pts2d_klt
            R_meas, parallax_px, n_pose_in = self._interframe_motion(
                prev_pts2d_klt, pts2d_klt)
            info["parallax_px"] = parallax_px
            # Static check: median frame-to-frame pixel motion.
            med_pix = float(np.median(np.linalg.norm(
                pts2d_klt - prev_pts2d_klt, axis=1)))
            if med_pix < self.static_pix_thresh:
                info["status"] = "static"
                info["static"] = True
                info["n_inliers"] = int(len(pts3d))
                self.prev_gray = gray
                return self.T_world_curr, info
        else:
            # --- Fallback: ORB ratio match against keyframe -----------
            info["tracker"] = "orb"
            matches = self._ratio_match(self.kf_des, curr_des)
            info["n_matches"] = len(matches)
            if len(matches) < self.min_matches:
                info["status"] = "too_few_matches"
                if len(matches) < self.min_matches // 2 and curr_des is not None:
                    self._set_keyframe(curr_kps, curr_des, depth_np,
                                       self.T_world_curr, gray=gray)
                    info["status"] = "kf_relocked"
                self.prev_gray = gray
                return self.T_world_curr, info

            K = self.K
            Hd, Wd = self.kf_depth_np.shape
            o_pts3d, o_pts2d, o_kf_uv, o_cu_uv = [], [], [], []
            for m in matches:
                pu, pv = self.kf_kps[m.queryIdx].pt
                cu, cv_ = curr_kps[m.trainIdx].pt
                ui, vi = int(round(pu)), int(round(pv))
                if not (0 <= ui < Wd and 0 <= vi < Hd):
                    continue
                d = float(self.kf_depth_np[vi, ui])
                if not (0.3 < d < 50.0):
                    continue
                x = (pu - K[0, 2]) / K[0, 0] * d
                y = (pv - K[1, 2]) / K[1, 1] * d
                o_pts3d.append([x, y, d])
                o_pts2d.append([cu, cv_])
                o_kf_uv.append([pu, pv])
                o_cu_uv.append([cu, cv_])
            if len(o_pts3d) < 8:
                info["status"] = "too_few_3d"
                self.prev_gray = gray
                return self.T_world_curr, info
            pts3d = np.asarray(o_pts3d, dtype=np.float32)
            pts2d = np.asarray(o_pts2d, dtype=np.float32)
            o_kf_uv = np.asarray(o_kf_uv, dtype=np.float32)
            o_cu_uv = np.asarray(o_cu_uv, dtype=np.float32)

            med_pix = float(np.median(np.linalg.norm(
                o_cu_uv - o_kf_uv, axis=1)))
            if med_pix < self.static_pix_thresh:
                info["status"] = "static"
                info["static"] = True
                info["n_inliers"] = len(pts3d)
                self.prev_T_world_curr = self.T_world_curr.copy()
                self.T_world_curr = self.kf_T_world.copy()
                self.prev_gray = gray
                return self.T_world_curr, info

        K = self.K
        T_world_kf = self.kf_T_world
        T_kf_world = np.linalg.inv(T_world_kf)

        # Decide whether this frame is rotation-dominated. If parallax
        # is low and the essential-matrix recovery had enough inliers,
        # depth-based PnP translation is ill-conditioned: estimate
        # rotation from the inter-frame R and lock translation.
        rot_only = (R_meas is not None
                    and parallax_px < self.parallax_thresh_px
                    and n_pose_in >= 30)
        info["rot_only"] = bool(rot_only)

        # Build PnP seed pose (in keyframe frame). We always seed
        # rotation from the *measured* inter-frame R when available —
        # that's robust through fast yaw where constant-velocity is
        # wrong. Translation is seeded from the prior body-frame delta
        # only when parallax shows real translation; otherwise zero.
        if R_meas is not None:
            T_prev_curr = np.eye(4, dtype=np.float32)
            T_prev_curr[:3, :3] = R_meas
            if not rot_only:
                T_prev_curr[:3, 3] = self.last_t_delta
            T_pred_world = self.T_world_curr @ T_prev_curr
        else:
            T_pred_world = self.T_world_curr

        T_kf_curr_pred = T_kf_world @ T_pred_world
        T_curr_kf_pred = np.linalg.inv(T_kf_curr_pred)
        rvec_init, _ = cv2.Rodrigues(T_curr_kf_pred[:3, :3])
        rvec_init = rvec_init.astype(np.float64)
        tvec_init = T_curr_kf_pred[:3, 3].reshape(3, 1).astype(np.float64)

        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d, pts2d, K, None,
            iterationsCount=200,
            reprojectionError=self.reproj_err_px,
            flags=cv2.SOLVEPNP_ITERATIVE,
            useExtrinsicGuess=True,
            rvec=rvec_init.copy(), tvec=tvec_init.copy(),
        )
        if not ok or inliers is None or len(inliers) < 8:
            # PnP failed but if we have a confident inter-frame R from
            # KLT essentials, still advance the rotation (rotation-only
            # fallback) so trajectory carries through fast spins.
            if R_meas is not None and n_pose_in >= 20:
                T_prev_curr_rot = np.eye(4, dtype=np.float32)
                T_prev_curr_rot[:3, :3] = R_meas
                self.prev_T_world_curr = self.T_world_curr.copy()
                self.T_world_curr = (self.T_world_curr @ T_prev_curr_rot).astype(np.float32)
                self.last_t_delta = np.zeros(3, dtype=np.float32)
                info["status"] = "rot_only_pnp_fail"
                info["rot_only"] = True
                self.prev_gray = gray
                return self.T_world_curr, info
            info["status"] = "pnp_fail"
            self.prev_gray = gray
            return self.T_world_curr, info

        inl = inliers.flatten()
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                pts3d[inl], pts2d[inl], K, None, rvec, tvec,
            )
        except cv2.error:
            pass

        R, _ = cv2.Rodrigues(rvec)
        t_ = tvec.flatten()
        T_curr_kf = np.eye(4, dtype=np.float64)
        T_curr_kf[:3, :3] = R
        T_curr_kf[:3, 3] = t_
        T_kf_curr = np.linalg.inv(T_curr_kf).astype(np.float32)
        T_world_curr_new = (T_world_kf @ T_kf_curr).astype(np.float32)

        # Rotation-only mode: keep the PnP rotation but lock translation
        # to what it was — depth-based t is unreliable under pure
        # rotation and would drift the trajectory.
        if rot_only:
            T_world_curr_new[:3, 3] = self.T_world_curr[:3, 3]

        dT = np.linalg.inv(self.T_world_curr) @ T_world_curr_new
        delta_trans = float(np.linalg.norm(dT[:3, 3]))
        cos_a = float(np.clip((np.trace(dT[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
        delta_rot_deg = float(np.degrees(np.arccos(cos_a)))
        info["delta_trans_m"] = delta_trans
        info["delta_rot_deg"] = delta_rot_deg
        info["n_inliers"] = int(len(inl))

        if delta_trans > self.max_trans_m or delta_rot_deg > self.max_rot_deg:
            info["status"] = "rejected_jump"
            self.prev_gray = gray
            return self.T_world_curr, info

        self.prev_T_world_curr = self.T_world_curr.copy()
        self.T_world_curr = T_world_curr_new
        # Track body-frame translation delta only when it was a true
        # translation update (not a locked rotation-only frame).
        if rot_only:
            self.last_t_delta = np.zeros(3, dtype=np.float32)
        else:
            self.last_t_delta = dT[:3, 3].astype(np.float32)

        # When KLT was used, prune the carried tracks to PnP inliers
        # so future-frame correspondences stay clean.
        if used_klt and self.tracked_uv is not None \
                and len(self.tracked_uv) == len(pts3d):
            self.tracked_uv = self.tracked_uv[inl]
            self.tracked_pts3d = self.tracked_pts3d[inl]

        # Update prev_gray for the next KLT step.
        self.prev_gray = gray

        kf_trans = float(np.linalg.norm(T_kf_curr[:3, 3]))
        kf_cos = float(np.clip((np.trace(T_kf_curr[:3, :3]) - 1.0) / 2.0,
                               -1.0, 1.0))
        kf_rot_deg = float(np.degrees(np.arccos(kf_cos)))
        # Spawn a new keyframe when motion is large OR tracker support
        # is weakening (low inliers / KLT track count dropping).
        klt_starved = (used_klt and self.tracked_uv is not None
                       and len(self.tracked_uv) < self.klt_min_tracks)
        if (kf_trans > self.kf_trans_m
                or kf_rot_deg > self.kf_rot_deg
                or info["n_inliers"] < max(40, self.min_matches * 2)
                or klt_starved):
            self._set_keyframe(curr_kps, curr_des, depth_np,
                               T_world_curr_new, gray=gray)
            info["status"] = "kf_spawned"

        return self.T_world_curr, info


# ─── Frame-to-frame visual odometry (true VO, no keyframe anchor) ────

class FrameToFrameVO:
    """Pure frame-to-frame RGB-D visual odometry.

    Each frame:
      1. KLT-track corners from previous frame to current.
      2. Unproject prev-frame 2D corners via the *previous* frame's
         depth -> 3D in the previous camera frame.
      3. Solve PnP: prev-cam 3D <-> current 2D -> T_prev_curr.
      4. Compose: T_world_curr = T_world_prev @ T_prev_curr.

    No keyframe anchor; pose is always chained off the immediately
    preceding frame. Handles fast yaw / 360 spins because the
    inter-frame baseline is always 1 frame and the keyframe never
    goes stale.

    Drift accumulates linearly with frames -- compensate with loop
    closure or windowed BA on top. For a short trajectory through a
    spin, this is the right default."""

    def __init__(self, K,
                 max_corners=800, quality=0.01, min_distance=8,
                 klt_win=31, klt_levels=4, refresh_below=300,
                 reproj_err_px=2.0, parallax_thresh_px=1.5,
                 min_depth_m=0.3, max_depth_m=50.0,
                 max_trans_m=0.6, max_rot_deg=30.0):
        self.K = K
        self.max_corners = int(max_corners)
        self.quality = float(quality)
        self.min_distance = int(min_distance)
        self.refresh_below = int(refresh_below)
        self.reproj_err_px = float(reproj_err_px)
        self.parallax_thresh_px = float(parallax_thresh_px)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.max_trans_m = float(max_trans_m)
        self.max_rot_deg = float(max_rot_deg)

        self._klt_lk_params = dict(
            winSize=(int(klt_win), int(klt_win)),
            maxLevel=int(klt_levels),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      20, 0.03),
            minEigThreshold=1e-4,
        )

        self.T_world_curr = np.eye(4, dtype=np.float32)
        self.prev_gray = None
        self.prev_depth_np = None
        self.prev_pts = None      # (N, 1, 2) float32

        # Compatibility shims with KeyframeVO so the existing main
        # loop and Trajectory/LoopDetector code paths don't blow up.
        # F2F doesn't spawn keyframes, so kf_id stays 0.
        self.kf_id = 0
        self.kf_kps = None
        self.kf_des = None
        self.kf_depth_np = None
        self.kf_T_world = np.eye(4, dtype=np.float32)

    def _refresh_corners(self, gray, mask=None):
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners,
            qualityLevel=self.quality, minDistance=self.min_distance,
            mask=mask, blockSize=7)

    def _interframe_R_parallax(self, prev_pts2d, curr_pts2d):
        """Inter-frame rotation R (prev_cam -> curr_cam) and parallax
        in pixels (median de-rotation residual). Returns
        (R_3x3_f32, parallax_px, n_inliers) or (None, +inf, 0)."""
        n = len(prev_pts2d)
        if n < 8:
            return None, float("inf"), 0
        K = self.K
        try:
            E, mask = cv2.findEssentialMat(
                prev_pts2d, curr_pts2d, K,
                method=cv2.RANSAC, prob=0.999, threshold=1.0)
        except cv2.error:
            return None, float("inf"), 0
        if E is None or mask is None or int(mask.sum()) < 8:
            return None, float("inf"), 0
        try:
            n_pose, R, _t_dir, _ = cv2.recoverPose(
                E, prev_pts2d, curr_pts2d, K, mask=mask.copy())
        except cv2.error:
            return None, float("inf"), 0
        if n_pose < 8:
            return None, float("inf"), 0
        K_inv = np.linalg.inv(K).astype(np.float64)
        prev_h = np.hstack([prev_pts2d.astype(np.float64),
                            np.ones((n, 1))])
        rays = (K_inv @ prev_h.T).T
        rays_rot = (R @ rays.T).T
        z = rays_rot[:, 2:3]
        z = np.where(np.abs(z) < 1e-6, np.sign(z) * 1e-6 + 1e-12, z)
        proj = (K @ (rays_rot / z).T).T[:, :2]
        res = np.linalg.norm(proj - curr_pts2d.astype(np.float64), axis=1)
        m = mask.ravel().astype(bool)
        parallax = float(np.median(res[m])) if int(m.sum()) > 0 \
            else float(np.median(res))
        return R.astype(np.float32), parallax, int(n_pose)

    def _seed_first_frame(self, gray, depth_np):
        self.prev_gray = gray.copy()
        self.prev_depth_np = depth_np
        self.prev_pts = self._refresh_corners(gray)

    def process(self, gray, depth_np):
        info = {
            "status": "ok", "n_matches": 0, "n_inliers": 0,
            "kf_id": 0, "delta_trans_m": 0.0, "delta_rot_deg": 0.0,
            "static": False, "tracker": "f2f",
            "parallax_px": float("inf"), "rot_only": False,
        }

        if (self.prev_gray is None or self.prev_pts is None
                or self.prev_depth_np is None
                or len(self.prev_pts) < 8):
            self._seed_first_frame(gray, depth_np)
            info["status"] = "init"
            return self.T_world_curr, info

        # KLT prev -> curr
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            **self._klt_lk_params)
        if next_pts is None or status is None:
            self._seed_first_frame(gray, depth_np)
            info["status"] = "klt_fail"
            return self.T_world_curr, info

        status = status.flatten().astype(bool)
        next_uv = next_pts.reshape(-1, 2)
        prev_uv = self.prev_pts.reshape(-1, 2)
        Hh, Ww = gray.shape[:2]
        inb = ((next_uv[:, 0] >= 0) & (next_uv[:, 0] < Ww)
               & (next_uv[:, 1] >= 0) & (next_uv[:, 1] < Hh))
        keep = status & inb
        if int(keep.sum()) < 8:
            self._seed_first_frame(gray, depth_np)
            info["status"] = "klt_too_few"
            return self.T_world_curr, info
        prev_uv = prev_uv[keep]
        next_uv = next_uv[keep]
        info["n_matches"] = int(len(prev_uv))

        # Unproject prev 2D via prev depth -> 3D in prev cam frame
        K = self.K
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        Hd, Wd = self.prev_depth_np.shape
        u, v = prev_uv[:, 0], prev_uv[:, 1]
        ui = np.clip(np.round(u).astype(np.int32), 0, Wd - 1)
        vi = np.clip(np.round(v).astype(np.int32), 0, Hd - 1)
        d = self.prev_depth_np[vi, ui].astype(np.float32)
        ok = (np.isfinite(d) & (d > self.min_depth_m)
              & (d < self.max_depth_m))
        if int(ok.sum()) < 8:
            self._advance_prev(gray, depth_np, next_uv)
            info["status"] = "no_depth"
            return self.T_world_curr, info
        u = u[ok]; v = v[ok]; d = d[ok]
        prev_uv_ok = prev_uv[ok]
        curr_uv_ok = next_uv[ok]
        x = (u - cx) / fx * d
        y = (v - cy) / fy * d
        pts3d_prev = np.stack([x, y, d], axis=1).astype(np.float32)
        pts2d_curr = curr_uv_ok.astype(np.float32)

        # Inter-frame R + parallax (rotation gate)
        R_meas, parallax_px, n_pose_in = self._interframe_R_parallax(
            prev_uv_ok.astype(np.float32), pts2d_curr)
        info["parallax_px"] = parallax_px
        rot_only = (R_meas is not None
                    and parallax_px < self.parallax_thresh_px
                    and n_pose_in >= 30)
        info["rot_only"] = bool(rot_only)

        # PnP seed: T_curr_prev (PnP solves *curr* expressed in prev
        # camera frame, i.e. transforms prev-cam points to curr-cam).
        if R_meas is not None:
            T_prev_curr_seed = np.eye(4, dtype=np.float64)
            T_prev_curr_seed[:3, :3] = R_meas.astype(np.float64)
        else:
            T_prev_curr_seed = np.eye(4, dtype=np.float64)
        T_curr_prev_seed = np.linalg.inv(T_prev_curr_seed)
        rvec_init, _ = cv2.Rodrigues(T_curr_prev_seed[:3, :3])
        rvec_init = rvec_init.astype(np.float64)
        tvec_init = T_curr_prev_seed[:3, 3].reshape(3, 1).astype(np.float64)

        ok_pnp, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d_prev, pts2d_curr, K, None,
            iterationsCount=200,
            reprojectionError=self.reproj_err_px,
            flags=cv2.SOLVEPNP_ITERATIVE,
            useExtrinsicGuess=True,
            rvec=rvec_init.copy(), tvec=tvec_init.copy(),
        )

        if not ok_pnp or inliers is None or len(inliers) < 8:
            # Rotation-only fallback when essential matrix had enough
            # support — keep trajectory alive through hard frames.
            if R_meas is not None and n_pose_in >= 20:
                T_prev_curr = np.eye(4, dtype=np.float32)
                T_prev_curr[:3, :3] = R_meas
                self.T_world_curr = (self.T_world_curr
                                     @ T_prev_curr).astype(np.float32)
                info["status"] = "rot_only_pnp_fail"
                info["rot_only"] = True
                self._advance_prev(gray, depth_np, curr_uv_ok)
                return self.T_world_curr, info
            info["status"] = "pnp_fail"
            self._advance_prev(gray, depth_np, next_uv)
            return self.T_world_curr, info

        inl = inliers.flatten()
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                pts3d_prev[inl], pts2d_curr[inl], K, None, rvec, tvec)
        except cv2.error:
            pass

        Rm, _ = cv2.Rodrigues(rvec)
        t_ = tvec.flatten()
        T_curr_prev = np.eye(4, dtype=np.float64)
        T_curr_prev[:3, :3] = Rm
        T_curr_prev[:3, 3] = t_
        T_prev_curr = np.linalg.inv(T_curr_prev).astype(np.float32)

        # Lock translation under pure rotation.
        if rot_only:
            T_prev_curr[:3, 3] = 0.0

        # Per-frame motion sanity check.
        delta_trans = float(np.linalg.norm(T_prev_curr[:3, 3]))
        cos_a = float(np.clip((np.trace(T_prev_curr[:3, :3]) - 1.0) / 2.0,
                              -1.0, 1.0))
        delta_rot_deg = float(np.degrees(np.arccos(cos_a)))
        info["delta_trans_m"] = delta_trans
        info["delta_rot_deg"] = delta_rot_deg
        info["n_inliers"] = int(len(inl))

        if delta_trans > self.max_trans_m or delta_rot_deg > self.max_rot_deg:
            info["status"] = "rejected_jump"
            self._advance_prev(gray, depth_np, curr_uv_ok)
            return self.T_world_curr, info

        self.T_world_curr = (self.T_world_curr @ T_prev_curr).astype(np.float32)

        # Carry inlier tracks forward; refresh if depleted.
        survivors = pts2d_curr[inl]
        self._advance_prev(gray, depth_np, survivors)
        return self.T_world_curr, info

    def _advance_prev(self, gray, depth_np, current_uv):
        """Make `gray`/`depth_np`/`current_uv` the new previous frame.
        Tops up corner count if we've fallen below `refresh_below`."""
        pts = current_uv.reshape(-1, 1, 2).astype(np.float32) \
            if current_uv is not None and len(current_uv) > 0 else None
        if pts is None or len(pts) < self.refresh_below:
            Hh, Ww = gray.shape[:2]
            mask = np.full((Hh, Ww), 255, dtype=np.uint8)
            if pts is not None:
                for (xx, yy) in pts.reshape(-1, 2):
                    cv2.circle(mask, (int(round(xx)), int(round(yy))),
                               self.min_distance, 0, -1)
            new_corners = self._refresh_corners(gray, mask=mask)
            if new_corners is not None and len(new_corners) > 0:
                if pts is not None:
                    pts = np.vstack([pts, new_corners])
                else:
                    pts = new_corners
        self.prev_pts = pts
        self.prev_gray = gray.copy()
        self.prev_depth_np = depth_np


# ─── GPU log-odds 2D occupancy grid ──────────────────────────────────

class OccupancyGrid2D:
    """GPU log-odds 2D occupancy grid. Vendored from
    MASt3R-SLAM/mast3r_slam/occupancy_2d.py (the OccupancyGrid2D class
    only — the standalone worker uses lietorch which we don't depend on
    here)."""

    def __init__(
        self,
        size_m=40.0,
        cell_m=0.10,
        device="cuda",
        obstacle_low=0.10,
        obstacle_high=2.0,
        lo_hit=0.85,
        lo_max=5.0,
        lo_min=-5.0,
        floor_init_points=8000,
        floor_refresh_frames=120,
    ):
        self.size_m = size_m
        self.cell_m = cell_m
        self.n = int(round(size_m / cell_m))
        self.origin = -size_m / 2.0
        self.device = device
        self.obstacle_low = obstacle_low
        self.obstacle_high = obstacle_high
        self.lo_hit = lo_hit
        self.lo_max = lo_max
        self.lo_min = lo_min
        self.floor_init_points = floor_init_points
        self.floor_refresh_frames = floor_refresh_frames

        self.log_odds = torch.zeros(self.n, self.n, device=device,
                                    dtype=torch.float32)
        self.floor_y = None
        self._floor_buf = []
        self._n_updates = 0

    @torch.inference_mode()
    def update_world_points(self, X_world, cam_xz=None,
                            ray_samples=14, miss_weight=-0.35,
                            max_ray_m=12.0):
        if X_world.shape[0] < 64:
            return

        y = X_world[:, 1]
        self._n_updates += 1
        if self.floor_y is None:
            self._floor_buf.append(y.detach())
            total = sum(t.numel() for t in self._floor_buf)
            if total >= self.floor_init_points:
                ys = torch.cat(self._floor_buf)
                if ys.numel() > 1_000_000:
                    idx = torch.randint(0, ys.numel(), (1_000_000,),
                                        device=ys.device)
                    ys = ys[idx]
                self.floor_y = float(torch.quantile(ys, 0.95).item())
                self._floor_buf = []
            else:
                return
        elif self._n_updates % max(1, self.floor_refresh_frames) == 0:
            ys = y.detach()
            if ys.numel() > 200_000:
                idx = torch.randint(0, ys.numel(), (200_000,), device=ys.device)
                ys = ys[idx]
            new_floor = float(torch.quantile(ys, 0.95).item())
            self.floor_y = 0.9 * self.floor_y + 0.1 * new_floor

        rel_h = self.floor_y - y
        mask = (rel_h > self.obstacle_low) & (rel_h < self.obstacle_high)
        if not mask.any():
            return

        xs = X_world[mask, 0]
        zs = X_world[mask, 2]

        if cam_xz is not None:
            cx_w, cz_w = float(cam_xz[0]), float(cam_xz[1])
            cam_t = torch.tensor([cx_w, cz_w], device=self.device,
                                 dtype=xs.dtype)
            d = torch.sqrt((xs - cx_w) ** 2 + (zs - cz_w) ** 2)
            in_range = d < max_ray_m
            xs = xs[in_range]
            zs = zs[in_range]
            if xs.numel() == 0:
                return

        cx = ((xs - self.origin) / self.cell_m).long()
        cz = ((zs - self.origin) / self.cell_m).long()
        ok = (cx >= 0) & (cx < self.n) & (cz >= 0) & (cz < self.n)
        cx_h, cz_h = cx[ok], cz[ok]
        if cx_h.numel() > 0:
            flat = cz_h * self.n + cx_h
            hits = torch.full(
                (flat.numel(),), self.lo_hit,
                dtype=self.log_odds.dtype, device=self.device,
            )
            self.log_odds.view(-1).scatter_add_(0, flat, hits)

        if cam_xz is not None and cx_h.numel() > 0 and ray_samples > 0:
            pts_xz = torch.stack([xs[ok], zs[ok]], dim=-1)
            cam_pt = cam_t.to(pts_xz.dtype)
            t = torch.linspace(0.05, 0.88, ray_samples,
                               device=self.device, dtype=pts_xz.dtype)
            diff = pts_xz - cam_pt
            samples = cam_pt + t[None, :, None] * diff[:, None, :]
            samples = samples.view(-1, 2)
            mcx = ((samples[:, 0] - self.origin) / self.cell_m).long()
            mcz = ((samples[:, 1] - self.origin) / self.cell_m).long()
            mok = (mcx >= 0) & (mcx < self.n) & (mcz >= 0) & (mcz < self.n)
            mcx, mcz = mcx[mok], mcz[mok]
            if mcx.numel() > 0:
                mflat = mcz * self.n + mcx
                miss = torch.full(
                    (mflat.numel(),), miss_weight,
                    dtype=self.log_odds.dtype, device=self.device,
                )
                self.log_odds.view(-1).scatter_add_(0, mflat, miss)

        self.log_odds.clamp_(self.lo_min, self.lo_max)

    @torch.inference_mode()
    def render(self, robot_xz=None, trail_xz=None, target_px=720,
               view_radius_m=None, free_thresh=0.40,
               weak_thresh=0.52, strong_thresh=0.72,
               draw_grid=True, grid_step_m=5.0):
        prob = torch.sigmoid(self.log_odds)
        bg_tup = (50, 30, 30)
        bg = torch.tensor(bg_tup, dtype=torch.float32, device=self.device)
        free_col = torch.tensor([220, 200, 180], dtype=torch.float32,
                                device=self.device)
        weak_col = torch.tensor([30, 80, 160], dtype=torch.float32,
                                device=self.device)
        obs_col = torch.tensor([50, 200, 255], dtype=torch.float32,
                               device=self.device)
        img = bg.expand(self.n, self.n, 3).clone()

        free = prob < free_thresh
        weak = (prob >= weak_thresh) & (prob < strong_thresh)
        strong = prob >= strong_thresh
        img[free] = free_col
        if weak.any():
            a = ((prob - weak_thresh) /
                 max(1e-6, strong_thresh - weak_thresh)).clamp_(0.0, 1.0)
            a = a.unsqueeze(-1)
            mixed = (1 - a) * bg + a * weak_col
            img[weak] = mixed[weak]
        img[strong] = obs_col

        img = img.clamp_(0, 255).to(torch.uint8)
        img_np = img.cpu().numpy()

        if view_radius_m is not None and robot_xz is not None:
            rx, rz = float(robot_xz[0]), float(robot_xz[1])
            x_lo, x_hi = rx - view_radius_m, rx + view_radius_m
            z_lo, z_hi = rz - view_radius_m, rz + view_radius_m
        else:
            x_lo = self.origin
            x_hi = self.origin + self.size_m
            z_lo = self.origin
            z_hi = self.origin + self.size_m

        cx_lo = int(np.floor((x_lo - self.origin) / self.cell_m))
        cx_hi = int(np.ceil((x_hi - self.origin) / self.cell_m))
        cz_lo = int(np.floor((z_lo - self.origin) / self.cell_m))
        cz_hi = int(np.ceil((z_hi - self.origin) / self.cell_m))
        out_w = max(1, cx_hi - cx_lo)
        out_h = max(1, cz_hi - cz_lo)
        out = np.full((out_h, out_w, 3), bg_tup, dtype=np.uint8)

        sx0 = max(cx_lo, 0)
        sx1 = min(cx_hi, self.n)
        sz0 = max(cz_lo, 0)
        sz1 = min(cz_hi, self.n)
        if sx1 > sx0 and sz1 > sz0:
            out[(sz0 - cz_lo):(sz1 - cz_lo),
                (sx0 - cx_lo):(sx1 - cx_lo)] = img_np[sz0:sz1, sx0:sx1]

        out = np.flipud(out).copy()

        scale = max(1, target_px // max(out_h, out_w))
        if scale > 1:
            out = cv2.resize(out, (out_w * scale, out_h * scale),
                             interpolation=cv2.INTER_NEAREST)
        H_px, W_px = out.shape[:2]

        def to_pix(xz_arr):
            xz = np.asarray(xz_arr, dtype=np.float32).reshape(-1, 2)
            cx = ((xz[:, 0] - x_lo) / self.cell_m * scale).astype(np.int32)
            cz = H_px - 1 - ((xz[:, 1] - z_lo) / self.cell_m * scale).astype(np.int32)
            return np.stack([cx, cz], axis=1)

        if draw_grid and grid_step_m > 0:
            extent = max(x_hi - x_lo, z_hi - z_lo)
            ideal_step = extent / 10.0
            for cand in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500, 1000):
                if cand >= ideal_step:
                    grid_step_m = max(grid_step_m, float(cand))
                    break
            x_first = np.ceil(x_lo / grid_step_m) * grid_step_m
            for x_g in np.arange(x_first, x_hi + 1e-6, grid_step_m):
                px = int((x_g - x_lo) / self.cell_m * scale)
                if 0 <= px < W_px:
                    cv2.line(out, (px, 0), (px, H_px), (60, 60, 60), 1)
                    cv2.putText(out, f"{x_g:.0f}", (px + 2, H_px - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (180, 180, 180), 1, cv2.LINE_AA)
            z_first = np.ceil(z_lo / grid_step_m) * grid_step_m
            for z_g in np.arange(z_first, z_hi + 1e-6, grid_step_m):
                py = H_px - 1 - int((z_g - z_lo) / self.cell_m * scale)
                if 0 <= py < H_px:
                    cv2.line(out, (0, py), (W_px, py), (60, 60, 60), 1)
                    cv2.putText(out, f"{z_g:.0f}", (4, py - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                (180, 180, 180), 1, cv2.LINE_AA)

        if trail_xz is not None and len(trail_xz) >= 2:
            pts = to_pix(trail_xz)
            cv2.polylines(out, [pts], False, (80, 220, 80), 2, cv2.LINE_AA)

        if robot_xz is not None:
            pix = to_pix([robot_xz])[0]
            cv2.circle(out, (int(pix[0]), int(pix[1])), 9, (0, 220, 255), -1)
            cv2.circle(out, (int(pix[0]), int(pix[1])), 9, (0, 0, 0), 1)

        return out


# ─── Background worker thread ────────────────────────────────────────

class _SlamWorker(QThread):
    """Owns the depth model, VO, and grid. Pulls frames from a 1-slot
    inbox and emits results."""

    update_ready = pyqtSignal(np.ndarray, np.ndarray, dict)
    models_loaded = pyqtSignal(bool)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._inbox = None  # latest queued bgr frame (drops old)
        self._wake = threading.Event()
        self._reset_requested = False

        self.scene = "indoor"
        self.variant = "small"
        self.fov_deg = 82.0
        self.frame_long_edge = 640
        self.depth_every = 1
        self.map_size_m = 40.0
        self.cell_m = 0.10
        self.view_radius_m = 15.0
        self.stride = 4
        self.min_depth = 0.3
        self.max_depth = 12.0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.depth_model = None
        self.grid = None
        self.orb = None
        self.vo = None
        self.K = None
        self.trail = []
        self.distance_m = 0.0
        self.last_t = np.zeros(3, dtype=np.float32)
        self.prev_depth_np = None
        self._n = 0
        self._t0 = time.time()

    def configure(self, scene="indoor", variant="small", fov_deg=82.0,
                  frame_long_edge=640, depth_every=1, map_size_m=40.0,
                  cell_m=0.10, view_radius_m=15.0):
        self.scene = scene
        self.variant = variant
        self.fov_deg = float(fov_deg)
        self.frame_long_edge = int(frame_long_edge)
        self.depth_every = max(1, int(depth_every))
        self.map_size_m = float(map_size_m)
        self.cell_m = float(cell_m)
        self.view_radius_m = float(view_radius_m) if view_radius_m else None

    def submit(self, bgr):
        """Push the latest frame; drop the prior queued one if not yet
        consumed (so the worker always works on the freshest frame)."""
        with self._lock:
            self._inbox = bgr
        self._wake.set()

    def request_reset(self):
        self._reset_requested = True
        self._wake.set()

    def stop(self):
        self._stop.set()
        self._wake.set()
        self.wait(3000)

    def _load_models(self):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            key = f"{self.scene}-{self.variant}"
            model_id = DEPTH_MODELS_SLAM.get(
                key, DEPTH_MODELS_SLAM["indoor-small"])
            print(f"[SLAM] Loading depth model {model_id} on {self.device}")
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.depth_model = DepthAnythingV2Wrap(
                model_id, device=self.device, dtype=dtype, input_size=322)
            self.grid = OccupancyGrid2D(
                size_m=self.map_size_m, cell_m=self.cell_m,
                device=self.device,
                obstacle_high=2.5 if self.scene == "indoor" else 8.0,
            )
            self.orb = cv2.ORB_create(nfeatures=1500, scaleFactor=1.2, nlevels=8)
            self.vo = None  # built lazily once we see the first frame
            self.K = None
            self.trail = []
            self.distance_m = 0.0
            self.last_t = np.zeros(3, dtype=np.float32)
            self.prev_depth_np = None
            self._n = 0
            self._t0 = time.time()
            self.models_loaded.emit(True)
        except Exception as e:
            print(f"[SLAM] model load failed: {e}")
            self.error.emit(str(e))
            self.models_loaded.emit(False)

    def _reset_state(self):
        if self.grid is not None:
            self.grid.log_odds.zero_()
            self.grid.floor_y = None
            self.grid._floor_buf = []
            self.grid._n_updates = 0
        self.vo = None
        self.K = None
        self.trail = []
        self.distance_m = 0.0
        self.last_t = np.zeros(3, dtype=np.float32)
        self.prev_depth_np = None
        self._n = 0
        self._t0 = time.time()
        self._reset_requested = False

    def _resize_frame(self, bgr):
        H, W = bgr.shape[:2]
        long_edge = max(H, W)
        if self.frame_long_edge >= long_edge:
            return bgr
        s = self.frame_long_edge / long_edge
        return cv2.resize(bgr, (int(round(W * s)), int(round(H * s))),
                          interpolation=cv2.INTER_AREA)

    def run(self):
        self._load_models()
        if self.depth_model is None:
            return

        while not self._stop.is_set():
            self._wake.wait(timeout=0.25)
            self._wake.clear()
            if self._stop.is_set():
                break
            if self._reset_requested:
                self._reset_state()

            with self._lock:
                bgr = self._inbox
                self._inbox = None
            if bgr is None:
                continue

            try:
                self._process_one(bgr)
            except Exception as e:
                print(f"[SLAM] frame processing error: {e}")
                self.error.emit(str(e))

    def _process_one(self, frame_bgr):
        frame_bgr = self._resize_frame(frame_bgr)
        H, W = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.K is None:
            self.K = make_intrinsics(W, H, self.fov_deg)
            self.vo = KeyframeVO(self.K, self.orb)

        run_depth = (self._n % self.depth_every == 0) or self.prev_depth_np is None
        if run_depth:
            depth_t = self.depth_model(rgb)
            depth_np = depth_t.detach().cpu().numpy()
        else:
            depth_t = None
            depth_np = self.prev_depth_np

        T_world_cam, vo_info = self.vo.process(gray, depth_np)
        self.prev_depth_np = depth_np

        cur_t = T_world_cam[:3, 3]
        self.distance_m += float(np.linalg.norm(cur_t - self.last_t))
        self.last_t = cur_t.copy()
        self.trail.append((float(cur_t[0]), float(cur_t[2])))

        edge_mask = compute_corner_mask(gray, dilate=1, fast_thresh=20)
        edge_mask_t = (torch.from_numpy(edge_mask).to(self.device)
                       if depth_t is not None else None)

        if depth_t is not None:
            T_t = torch.from_numpy(T_world_cam).to(self.device)
            cam_xz = (float(cur_t[0]), float(cur_t[2]))
            pts_edge = unproject_depth(
                depth_t, self.K, stride=self.stride,
                dmin=self.min_depth, dmax=self.max_depth,
                mask=edge_mask_t,
            )
            if pts_edge.shape[0] > 0:
                pts_w = transform_points(pts_edge, T_t)
                self.grid.update_world_points(
                    pts_w, cam_xz=cam_xz, ray_samples=14,
                    miss_weight=-0.35, max_ray_m=12.0,
                )

        self._n += 1
        elapsed = time.time() - self._t0
        fps = self._n / elapsed if elapsed > 0 else 0.0
        yaw_deg = float(np.degrees(np.arctan2(
            T_world_cam[0, 2], T_world_cam[2, 2])))
        obs_cells = int((torch.sigmoid(self.grid.log_odds) > 0.6).sum().item())

        fpv_disp = overlay_edges(frame_bgr, edge_mask)
        map_img = self.grid.render(
            robot_xz=(float(cur_t[0]), float(cur_t[2])),
            trail_xz=self.trail, target_px=720,
            view_radius_m=self.view_radius_m,
        )

        stats = {
            "Frame": self._n,
            "FPS": f"{fps:.1f}",
            "Pose x": f"{cur_t[0]:+.2f} m",
            "Pose y": f"{cur_t[1]:+.2f} m",
            "Pose z": f"{cur_t[2]:+.2f} m",
            "Yaw": f"{yaw_deg:+.1f}°",
            "Distance": f"{self.distance_m:.2f} m",
            "Floor y": (f"{self.grid.floor_y:+.2f} m"
                        if self.grid.floor_y is not None else "auto"),
            "VO status": vo_info["status"],
            "Matches": vo_info["n_matches"],
            "Inliers": vo_info["n_inliers"],
            "KF id": vo_info["kf_id"],
            "Cells known": obs_cells,
            "Trail pts": len(self.trail),
        }
        self.update_ready.emit(fpv_disp, map_img, stats)


# ─── Public Qt-friendly engine ───────────────────────────────────────

class SlamEngine(QObject):
    """Public façade for the SLAM pipeline.

    Lifecycle:
      eng = SlamEngine()
      eng.update_ready.connect(slot)          # slot(fpv, map, stats)
      eng.models_loaded.connect(on_ready)
      eng.start(scene='indoor', variant='small', fov_deg=82)
      eng.process_frame(bgr)                  # call per video frame
      eng.reset_map()
      eng.stop()
    """

    update_ready = pyqtSignal(np.ndarray, np.ndarray, dict)
    models_loaded = pyqtSignal(bool)
    error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: _SlamWorker | None = None
        self._running = False
        self._device_known = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    @property
    def is_running(self):
        return self._running

    @property
    def device(self):
        return self._device_known

    def start(self, scene="indoor", variant="small", fov_deg=82.0,
              frame_long_edge=640, depth_every=1, map_size_m=40.0,
              cell_m=0.10, view_radius_m=15.0):
        if self._running:
            return
        self._worker = _SlamWorker()
        self._worker.configure(
            scene=scene, variant=variant, fov_deg=fov_deg,
            frame_long_edge=frame_long_edge, depth_every=depth_every,
            map_size_m=map_size_m, cell_m=cell_m,
            view_radius_m=view_radius_m,
        )
        self._worker.update_ready.connect(self.update_ready.emit)
        self._worker.models_loaded.connect(self.models_loaded.emit)
        self._worker.error.connect(self.error.emit)
        self._worker.start()
        self._running = True

    def process_frame(self, frame_bgr):
        if self._running and self._worker is not None:
            self._worker.submit(frame_bgr)

    def reset_map(self):
        if self._running and self._worker is not None:
            self._worker.request_reset()

    def stop(self):
        if not self._running:
            return
        try:
            if self._worker is not None:
                self._worker.stop()
        finally:
            self._worker = None
            self._running = False


# ─── Offline MP4 playback thread (file mode) ─────────────────────────

class VideoFilePlayer(QThread):
    """Reads an MP4 with cv2.VideoCapture and emits each frame at the
    file's native FPS. Pairs with SlamEngine in 'file' mode."""

    frame_ready = pyqtSignal(np.ndarray)
    finished_playback = pyqtSignal()
    opened = pyqtSignal(float, int, int)  # fps, width, height

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()
        self.wait(2000)

    def run(self):
        if not os.path.exists(self.path):
            print(f"[SLAM] video file not found: {self.path}")
            self.finished_playback.emit()
            return
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            print(f"[SLAM] could not open video: {self.path}")
            self.finished_playback.emit()
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.opened.emit(float(fps), W, H)

        period = 1.0 / max(1.0, fps)
        next_t = time.time()
        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            self.frame_ready.emit(frame)
            next_t += period
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(min(sleep, 0.1))
            else:
                next_t = time.time()
        cap.release()
        self.finished_playback.emit()


class VideoDevicePlayer(QThread):
    """Opens a /dev/videoN device directly (independent of MainWindow's
    HDMI tap) and emits frames as they arrive. Used by the SLAM tab's
    'Camera device' source so SLAM can run on a separate camera port —
    e.g. an IR cam on /dev/video1 while HDMI is on /dev/video0."""

    frame_ready = pyqtSignal(np.ndarray)
    opened = pyqtSignal(float, int, int)  # fps, width, height
    error = pyqtSignal(str)
    finished_playback = pyqtSignal()

    def __init__(self, device_index, parent=None):
        super().__init__(parent)
        self.device_index = int(device_index)
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()
        self.wait(2000)

    def run(self):
        cap = cv2.VideoCapture(self.device_index, cv2.CAP_ANY)
        if not cap.isOpened():
            msg = f"could not open /dev/video{self.device_index}"
            print(f"[SLAM] {msg}")
            self.error.emit(msg)
            self.finished_playback.emit()
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.opened.emit(float(fps), W, H)

        # Live cameras dictate their own pacing — just read as fast as
        # the device produces frames; don't sleep-throttle.
        while not self._stop.is_set():
            ok, frame = cap.read()
            if ok:
                self.frame_ready.emit(frame)
            else:
                time.sleep(0.01)
        cap.release()
        self.finished_playback.emit()


# ─── Qt tab widget ───────────────────────────────────────────────────

def _bgr_to_pixmap(arr_bgr):
    """np.ndarray (H, W, 3) BGR -> QPixmap. Copies through QImage so
    the pixmap owns its buffer."""
    h, w = arr_bgr.shape[:2]
    rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)
    qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


class SlamTab(QWidget):
    """GPU 2D SLAM tab: real-time depth-anything + ORB keyframe VO +
    log-odds occupancy map. Two sources: live HDMI tap (forwarded from
    the parent MainWindow) or offline DJI MP4 replay.

    Emits `live_mode_changed(bool)`; MainWindow listens and gates
    whether to call slam_engine.process_frame() in its video callback.
    """

    live_mode_changed = pyqtSignal(bool)

    def __init__(self, slam_engine, default_variant="small", parent=None):
        super().__init__(parent)
        self.engine = slam_engine
        self.player = None
        self._latest_map = None
        self._chosen_file = None
        self._default_variant = default_variant
        self._build_ui()

        self.engine.update_ready.connect(self._on_update)
        self.engine.models_loaded.connect(self._on_models_loaded)
        self.engine.error.connect(self._on_engine_error)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Top control bar ──
        ctrl = QFrame()
        ctrl.setStyleSheet("QFrame { background-color: #0d0d0d; border-radius: 6px; }")
        cl = QHBoxLayout(ctrl)
        cl.setContentsMargins(8, 6, 8, 6)
        cl.setSpacing(6)

        cl.addWidget(self._lbl("Source:"))
        self.mode_combo = QComboBox()
        # 0 = MainWindow HDMI tap, 1 = direct /dev/videoN, 2 = file
        self.mode_combo.addItems(["MainWindow tap", "Camera device",
                                  "Offline file"])
        self.mode_combo.setFixedWidth(130)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        cl.addWidget(self.mode_combo)

        # Device picker (used when mode = "Camera device")
        self.device_combo = QComboBox()
        self.device_combo.setFixedWidth(120)
        self.device_combo.setEnabled(False)
        self._populate_devices()
        cl.addWidget(self.device_combo)

        self.refresh_dev_btn = QPushButton("⟳")
        self.refresh_dev_btn.setToolTip("Refresh /dev/video* list")
        self.refresh_dev_btn.setFixedSize(28, 28)
        self.refresh_dev_btn.setEnabled(False)
        self.refresh_dev_btn.clicked.connect(self._populate_devices)
        cl.addWidget(self.refresh_dev_btn)

        # File picker (used when mode = "Offline file")
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setFixedHeight(28)
        self.browse_btn.clicked.connect(self._on_browse)
        self.browse_btn.setEnabled(False)
        cl.addWidget(self.browse_btn)

        self.file_label = QLabel("(no file)")
        self.file_label.setStyleSheet("color: #777; font-size: 11px;")
        self.file_label.setMinimumWidth(140)
        cl.addWidget(self.file_label, stretch=1)

        cl.addWidget(self._sep())

        cl.addWidget(self._lbl("Scene:"))
        self.scene_combo = QComboBox()
        self.scene_combo.addItems(["indoor", "outdoor"])
        self.scene_combo.setFixedWidth(90)
        cl.addWidget(self.scene_combo)

        cl.addWidget(self._lbl("Model:"))
        self.variant_combo = QComboBox()
        self.variant_combo.addItems(["small", "base"])
        idx = self.variant_combo.findText(self._default_variant)
        if idx >= 0:
            self.variant_combo.setCurrentIndex(idx)
        self.variant_combo.setFixedWidth(80)
        cl.addWidget(self.variant_combo)

        cl.addWidget(self._lbl("FOV°:"))
        self.fov_spin = QDoubleSpinBox()
        self.fov_spin.setRange(20.0, 180.0)
        self.fov_spin.setValue(82.0)
        self.fov_spin.setDecimals(1)
        self.fov_spin.setFixedWidth(70)
        cl.addWidget(self.fov_spin)

        cl.addWidget(self._sep())

        self.start_btn = QPushButton("Start")
        self.start_btn.setCheckable(True)
        self.start_btn.setFixedHeight(28)
        self.start_btn.setStyleSheet(
            "QPushButton:checked { color: #00ccff; font-weight: bold; "
            "border-color: #0088ff; }"
        )
        self.start_btn.clicked.connect(self._on_start_toggle)
        cl.addWidget(self.start_btn)

        self.reset_btn = QPushButton("Reset Map")
        self.reset_btn.setFixedHeight(28)
        self.reset_btn.clicked.connect(self._on_reset)
        cl.addWidget(self.reset_btn)

        self.snap_btn = QPushButton("Snapshot")
        self.snap_btn.setFixedHeight(28)
        self.snap_btn.clicked.connect(self._on_snapshot)
        cl.addWidget(self.snap_btn)

        root.addWidget(ctrl)

        # ── Body: FPV | Map | Stats ──
        body = QHBoxLayout()
        body.setSpacing(6)

        self.fpv_label = QLabel("FPV (start SLAM to begin)")
        self.fpv_label.setAlignment(Qt.AlignCenter)
        self.fpv_label.setStyleSheet(
            "QLabel { background-color: #000; color: #555; "
            "border: 1px solid #1a1a1a; border-radius: 6px; }")
        self.fpv_label.setMinimumSize(320, 240)
        self.fpv_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body.addWidget(self.fpv_label, stretch=4)

        self.map_label = QLabel("Top-down map")
        self.map_label.setAlignment(Qt.AlignCenter)
        self.map_label.setStyleSheet(
            "QLabel { background-color: #181010; color: #555; "
            "border: 1px solid #1a1a1a; border-radius: 6px; }")
        self.map_label.setMinimumSize(320, 320)
        self.map_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        body.addWidget(self.map_label, stretch=4)

        stats_frame = QFrame()
        stats_frame.setStyleSheet(
            "QFrame { background-color: #0d0d0d; border-radius: 6px; }")
        stats_frame.setMinimumWidth(220)
        stats_frame.setMaximumWidth(280)
        sl = QVBoxLayout(stats_frame)
        sl.setContentsMargins(10, 10, 10, 10)
        sl.setSpacing(2)

        title = QLabel("POSE / TRACKING")
        title.setStyleSheet(
            "color: #00cc00; font-weight: bold; font-size: 11px;")
        sl.addWidget(title)
        sl.addSpacing(4)

        self.stats_label = QLabel("(idle)")
        self.stats_label.setStyleSheet(
            "color: #cccccc; font-family: 'Consolas', 'DejaVu Sans Mono', "
            "monospace; font-size: 11px;")
        self.stats_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.stats_label.setWordWrap(False)
        sl.addWidget(self.stats_label, stretch=1)

        body.addWidget(stats_frame, stretch=1)

        root.addLayout(body, stretch=1)

        self.status_label = QLabel("Idle — pick source, scene, then Start.")
        self.status_label.setStyleSheet(
            "color: #777; font-size: 11px; padding: 4px 8px;")
        root.addWidget(self.status_label)

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

    def _populate_devices(self):
        self.device_combo.clear()
        for i in range(10):
            if os.path.exists(f"/dev/video{i}"):
                self.device_combo.addItem(f"/dev/video{i}", i)
        if self.device_combo.count() == 0:
            self.device_combo.addItem("(no /dev/video*)", -1)

    def _on_mode_changed(self, idx):
        is_device = (idx == 1)
        is_file = (idx == 2)
        self.device_combo.setEnabled(is_device)
        self.refresh_dev_btn.setEnabled(is_device)
        self.browse_btn.setEnabled(is_file)
        # Tear down any owned player so switching modes doesn't leave
        # a thread reading frames into a stopped engine.
        self._stop_player()

    def _on_browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose drone video",
            os.path.expanduser("~"),
            "Video files (*.mp4 *.mov *.avi *.mkv);;All files (*)")
        if path:
            self.file_label.setText(os.path.basename(path))
            self.file_label.setToolTip(path)
            self._chosen_file = path
        else:
            self._chosen_file = None

    def _on_start_toggle(self, checked):
        if checked:
            self._start()
        else:
            self._stop()

    def _start(self):
        scene = self.scene_combo.currentText()
        variant = self.variant_combo.currentText()
        fov = self.fov_spin.value()
        mode = self.mode_combo.currentIndex()

        if self.engine.is_running:
            self.engine.stop()
        self.engine.start(scene=scene, variant=variant, fov_deg=fov,
                          frame_long_edge=640, depth_every=1,
                          map_size_m=40.0, cell_m=0.10,
                          view_radius_m=15.0)

        if mode == 0:
            # Piggyback on whatever device MainWindow has opened.
            self.live_mode_changed.emit(True)
            self.status_label.setText(
                "MainWindow tap: SLAM consuming the GCS video stream "
                "(models loading...)")
            self.start_btn.setText("Stop")
            return

        if mode == 1:
            dev_idx = self.device_combo.currentData()
            if dev_idx is None or dev_idx < 0:
                self.status_label.setText("Pick a /dev/video* device first.")
                self.start_btn.setChecked(False)
                self.engine.stop()
                return
            self.player = VideoDevicePlayer(dev_idx)
            self.player.frame_ready.connect(self.engine.process_frame)
            self.player.finished_playback.connect(self._on_playback_done)
            self.player.opened.connect(self._on_device_opened)
            self.player.error.connect(self._on_engine_error)
            self.player.start()
            self.status_label.setText(
                f"Device: /dev/video{dev_idx} (models loading...)")
            self.start_btn.setText("Stop")
            return

        # mode == 2: offline file
        path = self._chosen_file
        if not path or not os.path.exists(path):
            self.status_label.setText("Pick a video file first.")
            self.start_btn.setChecked(False)
            self.engine.stop()
            return
        self.player = VideoFilePlayer(path)
        self.player.frame_ready.connect(self.engine.process_frame)
        self.player.finished_playback.connect(self._on_playback_done)
        self.player.opened.connect(self._on_file_opened)
        self.player.start()
        self.status_label.setText(
            f"Offline: {os.path.basename(path)} (models loading...)")
        self.start_btn.setText("Stop")

    def _on_device_opened(self, fps, W, H):
        dev_idx = self.device_combo.currentData()
        self.status_label.setText(
            f"Device: /dev/video{dev_idx}  {W}x{H} @ {fps:.1f} fps")

    def _on_file_opened(self, fps, W, H):
        name = os.path.basename(self._chosen_file or "")
        self.status_label.setText(
            f"Offline: {name}  {W}x{H} @ {fps:.1f} fps")

    def _on_playback_done(self):
        self.status_label.setText("Offline playback finished.")
        self._stop_player()
        self.start_btn.setChecked(False)
        self.start_btn.setText("Start")

    def _stop_player(self):
        if self.player is not None:
            try:
                self.player.frame_ready.disconnect()
            except Exception:
                pass
            self.player.stop()
            self.player = None

    def _stop(self):
        self.live_mode_changed.emit(False)
        self._stop_player()
        self.engine.stop()
        self.start_btn.setText("Start")
        self.status_label.setText("Stopped.")

    def _on_reset(self):
        self.engine.reset_map()
        self.status_label.setText("Map reset.")

    def _on_snapshot(self):
        if self._latest_map is None:
            self.status_label.setText("No map to save yet.")
            return
        snap_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        path = os.path.join(
            snap_dir, f"slam_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        cv2.imwrite(path, self._latest_map)
        self.status_label.setText(f"Saved {path}")

    def _on_models_loaded(self, ok):
        if ok:
            self.status_label.setText("Models loaded. SLAM running.")
        else:
            self.status_label.setText("Model load FAILED — see console.")
            self.start_btn.setChecked(False)
            self.start_btn.setText("Start")

    def _on_engine_error(self, msg):
        self.status_label.setText(f"SLAM error: {msg}")

    def _on_update(self, fpv_bgr, map_bgr, stats):
        self._latest_map = map_bgr
        pm = _bgr_to_pixmap(fpv_bgr).scaled(
            self.fpv_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.fpv_label.setPixmap(pm)
        mp = _bgr_to_pixmap(map_bgr).scaled(
            self.map_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.map_label.setPixmap(mp)
        lines = [f"{k:<12} {v}" for k, v in stats.items()]
        self.stats_label.setText("\n".join(lines))

    def cleanup(self):
        self._stop_player()
        self.engine.stop()
