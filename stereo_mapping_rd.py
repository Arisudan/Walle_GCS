"""R&D: 2D occupancy mapping with visual odometry, from RealSense stereo.

Standalone — does not touch the GCS. Pipeline per frame:

    D4xx aligned RGB+depth
        ->  Open3D RGB-D odometry vs previous frame  -> dT_prev_curr
        ->  pose_world_curr = pose_world_prev @ dT_prev_curr
        ->  per-image-column nearest in-band hit     -> 1D range "scan"
        ->  raycast each ray from cam_world to hit_world into a
            persistent log-odds grid (FREE along ray, OCC at hit)
        ->  OpenCV display + nearest-obstacle HUD

Why this shape (drone context):
  * Real persistent map: cells the drone has driven past stay mapped
    even when the camera looks elsewhere.
  * Range-scan abstraction (one ray per image column at its closest
    in-band hit) is exactly what 2D SLAM stacks consume, so this map
    is reusable downstream.
  * Log-odds occupancy lets noisy single-frame hits decay until
    they're confirmed by repeat observations.
  * Open3D RGB-D odometry handles small inter-frame motion without
    needing an IMU. For aggressive drone motion or low-texture scenes
    you'd want to fuse FC attitude + position (MAVLink) into
    pose_world_curr; see HOOK marked below.

Known limits — read before flying:
  * D435 has no IMU; pure-VO drift accumulates over long traverses.
  * If odometry confidence tanks the frame is dropped from the map
    rather than corrupting it (HUD shows "ODOM LOST").
  * Camera assumed roughly level — Y band is the drone-altitude slice
    around the optical axis; tune Y_BAND_* for your mount.

Controls (viewer):
    s     - save world map + last color/depth snapshot
    r     - reset pose + clear the world map
    q/ESC - quit (auto-saves on the way out)

Headless:
    python3 stereo_mapping_rd.py --headless 5
        capture ~5 seconds, save world map PNG + log-odds .npy, print stats.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Open3D's legacy GLFW/GLEW viewer (not used here, but Open3D may try
# to spin up an OpenGL context during odometry init) fails under
# native Wayland. Force XWayland so any OpenGL fall-back stays sane.
if os.environ.get("XDG_SESSION_TYPE") == "wayland" and os.environ.get("DISPLAY"):
    os.environ.pop("WAYLAND_DISPLAY", None)
    os.environ["XDG_SESSION_TYPE"] = "x11"

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


# ─── Capture ────────────────────────────────────────────────────────
WIDTH, HEIGHT, FPS = 848, 480, 30

# ─── Depth gating ───────────────────────────────────────────────────
DEPTH_NEAR = 0.3
DEPTH_FAR  = 6.0

# ─── Height band (camera frame, +Y points DOWN) ─────────────────────
Y_BAND_UP   = -0.5
Y_BAND_DOWN = +0.5

# ─── World grid ────────────────────────────────────────────────────
WORLD_W_M   = 10.0     # extent along world X (left <-> right)
WORLD_D_M   = 10.0     # extent along world Z (back <-> forward)
WORLD_RES_M = 0.05     # 5 cm/cell -> 200x200 grid

# ─── Log-odds ───────────────────────────────────────────────────────
L_OCC = +0.85          # increment on a hit
L_FREE = -0.40         # decrement on a ray pass-through
L_MIN = -2.0
L_MAX = +3.5
L_OCC_THRESH = +0.85   # cells above this render as obstacle
L_FREE_THRESH = -0.40  # cells below this render as free

# ─── Scan ───────────────────────────────────────────────────────────
COL_STRIDE = 2         # subsample image columns for the per-frame scan
FORWARD_CONE_DEG = 60.0

# ─── Odometry downsample ──────────────────────────────────────────
# Open3D RGB-D odometry is the per-frame bottleneck. Halving each
# dimension cuts pixel count 4x with a small accuracy hit and brings
# end-to-end throughput up to ~15 fps on this box; bump back to 1
# if you have GPU/CPU to spare.
ODOM_DOWNSCALE = 2


# ─── RealSense ──────────────────────────────────────────────────────

def configure_pipeline() -> tuple[rs.pipeline, rs.align, rs.pipeline_profile]:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, FPS)
    cfg.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    return pipe, align, profile


def describe(profile: rs.pipeline_profile
             ) -> tuple[rs.intrinsics, float, o3d.camera.PinholeCameraIntrinsic]:
    intr = (profile.get_stream(rs.stream.color)
            .as_video_stream_profile().get_intrinsics())
    scale = profile.get_device().first_depth_sensor().get_depth_scale()
    s = ODOM_DOWNSCALE
    intr_o3d = o3d.camera.PinholeCameraIntrinsic(
        intr.width // s, intr.height // s,
        intr.fx / s, intr.fy / s, intr.ppx / s, intr.ppy / s,
    )
    print(f"intrinsics: fx={intr.fx:.1f} fy={intr.fy:.1f} "
          f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}  "
          f"depth_scale={scale:.4f} m/unit  "
          f"odom_res={intr.width//s}x{intr.height//s}")
    return intr, scale, intr_o3d


# ─── World grid spec ────────────────────────────────────────────────

class WorldGrid:
    """World-frame top-down occupancy grid; camera starts at the centre,
    +world-X is right, +world-Z is forward (image-up)."""
    def __init__(self, width_m: float, depth_m: float, res_m: float):
        self.res = res_m
        self.cols = int(round(width_m / res_m))
        self.rows = int(round(depth_m / res_m))
        self.origin_col = self.cols // 2
        self.origin_row = self.rows // 2
        self.width_m = width_m
        self.depth_m = depth_m
        self.log_odds = np.zeros((self.rows, self.cols), dtype=np.float32)

    def reset(self) -> None:
        self.log_odds.fill(0.0)

    def world_to_cell(self, x_w: float | np.ndarray,
                      z_w: float | np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
        col = np.round(np.asarray(x_w) / self.res + self.origin_col)
        row = np.round(self.origin_row - np.asarray(z_w) / self.res)
        return col.astype(np.int32), row.astype(np.int32)


# ─── RGBD odometry wrapper ──────────────────────────────────────────

_ODOM_OPT = o3d.pipelines.odometry.OdometryOption()
_ODOM_METHOD = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()


def make_rgbd(color_bgr: np.ndarray, depth_raw: np.ndarray,
              depth_scale_m: float) -> o3d.geometry.RGBDImage:
    if ODOM_DOWNSCALE != 1:
        new_w = color_bgr.shape[1] // ODOM_DOWNSCALE
        new_h = color_bgr.shape[0] // ODOM_DOWNSCALE
        color_bgr = cv2.resize(color_bgr, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)
        # uint16 depth: INTER_NEAREST preserves valid/invalid pixels.
        depth_raw = cv2.resize(depth_raw, (new_w, new_h),
                               interpolation=cv2.INTER_NEAREST)
    color_o3d = o3d.geometry.Image(np.ascontiguousarray(color_bgr[:, :, ::-1]))
    depth_o3d = o3d.geometry.Image(depth_raw)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0 / depth_scale_m,
        depth_trunc=DEPTH_FAR,
        convert_rgb_to_intensity=False,
    )


class PoseTracker:
    """Holds pose_world_curr and updates it from RGB-D odometry."""
    def __init__(self, intr_o3d: o3d.camera.PinholeCameraIntrinsic):
        self.intr = intr_o3d
        self.pose = np.eye(4, dtype=np.float32)
        self.prev_rgbd: o3d.geometry.RGBDImage | None = None
        self.trail: list[tuple[float, float]] = [(0.0, 0.0)]

    def reset(self) -> None:
        self.pose = np.eye(4, dtype=np.float32)
        self.prev_rgbd = None
        self.trail = [(0.0, 0.0)]

    def step(self, rgbd: o3d.geometry.RGBDImage) -> tuple[bool, float]:
        """Update self.pose using rgbd vs the previous RGBD.
        Returns (ok, confidence). On failure, pose is unchanged."""
        if self.prev_rgbd is None:
            self.prev_rgbd = rgbd
            return True, 1.0
        ok, T, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd, self.prev_rgbd, self.intr, np.eye(4),
            _ODOM_METHOD, _ODOM_OPT,
        )
        if not ok:
            return False, 0.0
        # T maps points in curr frame to coords in prev frame; pose_world
        # of curr camera = pose_world_prev @ T.
        self.pose = (self.pose @ T).astype(np.float32)
        self.prev_rgbd = rgbd
        # HOOK: if you have flight-controller pose (MAVLink LOCAL_POSITION
        # + ATTITUDE), this is the place to overwrite or KF-fuse self.pose.
        self.trail.append((float(self.pose[0, 3]), float(self.pose[2, 3])))
        # Use the rotational diag of the info matrix as a coarse confidence.
        return True, float(info[5, 5])

    @property
    def position_xz(self) -> tuple[float, float]:
        return float(self.pose[0, 3]), float(self.pose[2, 3])

    @property
    def yaw_rad(self) -> float:
        R = self.pose[:3, :3]
        return float(np.arctan2(R[0, 2], R[0, 0]))


# ─── Range-scan + map integration ───────────────────────────────────

def compute_scan(depth_raw: np.ndarray, depth_scale: float,
                 intr: rs.intrinsics
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-image-column nearest in-band hit.
    Returns (u_cols, x_cam, z_cam) for the valid columns only."""
    H, W = depth_raw.shape
    fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

    z = depth_raw.astype(np.float32) * depth_scale
    z[z == 0]         = np.nan
    z[z < DEPTH_NEAR] = np.nan
    z[z > DEPTH_FAR]  = np.nan

    v = np.arange(H, dtype=np.float32).reshape(-1, 1)
    Y = (v - cy) * z / fy
    in_band = (Y >= Y_BAND_UP) & (Y <= Y_BAND_DOWN)
    z_band = np.where(in_band, z, np.nan)

    # Subsample columns for speed.
    z_sub = z_band[:, ::COL_STRIDE]
    with np.errstate(all="ignore"):
        z_min = np.nanmin(z_sub, axis=0)               # (W/STRIDE,)
    valid = np.isfinite(z_min)

    u_sub = np.arange(0, W, COL_STRIDE, dtype=np.float32)
    x_cam = (u_sub - cx) * z_min / fx
    z_cam = z_min
    return u_sub[valid], x_cam[valid], z_cam[valid]


def integrate_scan(world: WorldGrid, pose_T_world_cam: np.ndarray,
                   x_cam: np.ndarray, z_cam: np.ndarray) -> int:
    """Raycast each scan hit into the world log-odds grid.
    Returns the number of rays actually drawn."""
    if x_cam.size == 0:
        return 0

    # Hits in world frame (Y=0; ignore vertical).
    n = x_cam.shape[0]
    pts_cam = np.stack([x_cam, np.zeros(n, np.float32),
                        z_cam,  np.ones(n,  np.float32)], axis=1)  # (n,4)
    pts_world = pts_cam @ pose_T_world_cam.T                       # (n,4)

    cam_x, cam_z = pose_T_world_cam[0, 3], pose_T_world_cam[2, 3]
    cam_col, cam_row = world.world_to_cell(cam_x, cam_z)
    cam_col = int(cam_col); cam_row = int(cam_row)
    if not (0 <= cam_col < world.cols and 0 <= cam_row < world.rows):
        return 0  # camera left the mapped region

    hit_cols, hit_rows = world.world_to_cell(pts_world[:, 0],
                                             pts_world[:, 2])

    # Build a per-frame delta mask (uint8: 1=ray pass-through, 2=hit).
    delta = np.zeros_like(world.log_odds, dtype=np.uint8)
    drawn = 0
    for i in range(n):
        hc, hr = int(hit_cols[i]), int(hit_rows[i])
        if not (0 <= hc < world.cols and 0 <= hr < world.rows):
            continue
        cv2.line(delta, (cam_col, cam_row), (hc, hr), 1, 1)
        drawn += 1
    # Stamp the hits AFTER the lines so the hit cell wins.
    in_grid = ((hit_cols >= 0) & (hit_cols < world.cols)
               & (hit_rows >= 0) & (hit_rows < world.rows))
    delta[hit_rows[in_grid], hit_cols[in_grid]] = 2

    # Apply log-odds updates in bulk.
    world.log_odds[delta == 1] += L_FREE
    world.log_odds[delta == 2] += L_OCC
    np.clip(world.log_odds, L_MIN, L_MAX, out=world.log_odds)
    return drawn


def nearest_in_cone(x_cam: np.ndarray, z_cam: np.ndarray,
                    cone_deg: float) -> tuple[float, float]:
    if z_cam.size == 0:
        return float("nan"), float("nan")
    bearings = np.degrees(np.arctan2(x_cam, z_cam))
    in_cone = np.abs(bearings) <= cone_deg / 2.0
    if not np.any(in_cone):
        return float("nan"), float("nan")
    z_in = np.where(in_cone, z_cam, np.inf)
    i = int(np.argmin(z_in))
    return float(z_cam[i]), float(bearings[i])


# ─── Rendering ──────────────────────────────────────────────────────

def render_world(world: WorldGrid, tracker: PoseTracker,
                 nearest_m: float, nearest_b: float,
                 fps: float, odom_ok: bool) -> np.ndarray:
    L = world.log_odds
    img = np.full((world.rows, world.cols, 3), 128, dtype=np.uint8)  # unknown
    img[L <= L_FREE_THRESH] = (245, 245, 245)
    img[L >= L_OCC_THRESH]  = (40, 40, 220)   # red obstacle

    # Distance rings (every metre) around current camera position.
    cam_col, cam_row = world.world_to_cell(*tracker.position_xz)
    cam_col, cam_row = int(cam_col), int(cam_row)
    if 0 <= cam_col < world.cols and 0 <= cam_row < world.rows:
        for r_m in range(1, int(min(world.width_m, world.depth_m) / 2) + 1):
            cv2.circle(img, (cam_col, cam_row), int(r_m / world.res),
                       (90, 90, 90), 1, cv2.LINE_AA)

    # Trail of past camera positions.
    if len(tracker.trail) > 1:
        pts = np.array(tracker.trail, dtype=np.float32)
        cs, rs_ = world.world_to_cell(pts[:, 0], pts[:, 1])
        for i in range(1, len(cs)):
            cv2.line(img, (int(cs[i-1]), int(rs_[i-1])),
                     (int(cs[i]),   int(rs_[i])),
                     (0, 180, 0), 1, cv2.LINE_AA)

    # Drone marker: triangle pointing along current yaw.
    yaw = tracker.yaw_rad
    fwd = np.array([np.sin(yaw), -np.cos(yaw)])  # image-frame forward
    side = np.array([-fwd[1], fwd[0]])
    if 0 <= cam_col < world.cols and 0 <= cam_row < world.rows:
        tip   = np.array([cam_col, cam_row]) + fwd  * 8
        back_l = np.array([cam_col, cam_row]) - fwd * 4 + side * 4
        back_r = np.array([cam_col, cam_row]) - fwd * 4 - side * 4
        tri = np.array([tip, back_l, back_r], dtype=np.int32)
        cv2.fillPoly(img, [tri], (255, 180, 0))

    # Upscale so the grid is readable.
    SCALE = 3
    img = cv2.resize(img, None, fx=SCALE, fy=SCALE,
                     interpolation=cv2.INTER_NEAREST)

    # HUD.
    if not odom_ok:
        hud = f"ODOM LOST   ({fps:4.1f} fps)"
        color = (0, 0, 255)
    elif np.isfinite(nearest_m):
        hud = (f"nearest: {nearest_m:5.2f} m @ {nearest_b:+5.1f}°   "
               f"pose=({tracker.position_xz[0]:+5.2f},{tracker.position_xz[1]:+5.2f}) m "
               f"yaw={np.degrees(tracker.yaw_rad):+5.1f}°   "
               f"({fps:4.1f} fps)")
        color = (0, 255, 0)
    else:
        hud = (f"cone clear   pose=({tracker.position_xz[0]:+5.2f},"
               f"{tracker.position_xz[1]:+5.2f}) m   ({fps:4.1f} fps)")
        color = (0, 255, 0)
    cv2.rectangle(img, (8, 8), (8 + 9 * len(hud), 36), (0, 0, 0), -1)
    cv2.putText(img, hud, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def colorize_depth(depth_raw: np.ndarray, depth_scale: float) -> np.ndarray:
    z = depth_raw.astype(np.float32) * depth_scale
    z = np.clip(z, 0.0, DEPTH_FAR) / DEPTH_FAR
    out = cv2.applyColorMap((z * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[depth_raw == 0] = 0
    return out


# ─── Save helpers ───────────────────────────────────────────────────

def save_map(world: WorldGrid, tracker: PoseTracker,
             out_dir: Path, tag: str = "map") -> Path:
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    img = render_world(world, tracker, float("nan"), float("nan"),
                       0.0, odom_ok=True)
    path_png = out_dir / f"{tag}_{stamp}.png"
    path_npy = out_dir / f"{tag}_{stamp}_logodds.npy"
    cv2.imwrite(str(path_png), img)
    np.save(path_npy, world.log_odds)
    print(f"saved map -> {path_png}")
    print(f"saved log-odds -> {path_npy}  "
          f"(occ_cells={(world.log_odds >= L_OCC_THRESH).sum()}, "
          f"free_cells={(world.log_odds <= L_FREE_THRESH).sum()})")
    return path_png


# ─── Runs ───────────────────────────────────────────────────────────

def run_headless(seconds: float) -> int:
    pipe, align, profile = configure_pipeline()
    try:
        intr, scale, intr_o3d = describe(profile)
        world = WorldGrid(WORLD_W_M, WORLD_D_M, WORLD_RES_M)
        tracker = PoseTracker(intr_o3d)
        print(f"world grid: {world.cols} x {world.rows} cells "
              f"({WORLD_W_M} m x {WORLD_D_M} m, {WORLD_RES_M*100:.0f} cm/cell)")

        for _ in range(5):
            pipe.wait_for_frames()

        n_frames = 0
        n_lost = 0
        t_start = time.time()
        t_end = t_start + seconds
        while time.time() < t_end:
            frames = align.process(pipe.wait_for_frames())
            d, c = frames.get_depth_frame(), frames.get_color_frame()
            if not d or not c:
                continue
            depth_raw = np.asanyarray(d.get_data())
            color = np.asanyarray(c.get_data())
            rgbd = make_rgbd(color, depth_raw, scale)
            ok, conf = tracker.step(rgbd)
            if not ok:
                n_lost += 1
                continue
            _, x_cam, z_cam = compute_scan(depth_raw, scale, intr)
            drawn = integrate_scan(world, tracker.pose, x_cam, z_cam)
            n_frames += 1
            if n_frames % 10 == 0:
                near_m, near_b = nearest_in_cone(x_cam, z_cam,
                                                 FORWARD_CONE_DEG)
                px, pz = tracker.position_xz
                print(f"  frame {n_frames:3d}: rays={drawn:4d} "
                      f"pos=({px:+5.2f},{pz:+5.2f}) m "
                      f"yaw={np.degrees(tracker.yaw_rad):+5.1f}°  "
                      f"nearest={near_m:5.2f} m @ {near_b:+5.1f}°  "
                      f"conf={conf:.1f}")

        elapsed = time.time() - t_start
        print(f"done: {n_frames} frames integrated, {n_lost} lost, "
              f"{n_frames/elapsed:.1f} fps")
        save_map(world, tracker, Path("out_mapping"))
        return 0
    finally:
        pipe.stop()


def run_viewer() -> int:
    pipe, align, profile = configure_pipeline()
    intr, scale, intr_o3d = describe(profile)
    world = WorldGrid(WORLD_W_M, WORLD_D_M, WORLD_RES_M)
    tracker = PoseTracker(intr_o3d)

    win     = "Map  (s save · r reset · q quit)"
    win_cam = "Camera  (color | depth)"
    cv2.namedWindow(win,     cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(win_cam, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(win_cam, 0,   30)
    cv2.moveWindow(win,     0,  530)

    out_dir = Path("out_mapping")
    last_t = time.time()
    fps = 0.0
    odom_ok_last = True

    for _ in range(5):
        pipe.wait_for_frames()

    try:
        while True:
            frames = align.process(pipe.wait_for_frames())
            d, c = frames.get_depth_frame(), frames.get_color_frame()
            if not d or not c:
                continue
            depth_raw = np.asanyarray(d.get_data())
            color = np.asanyarray(c.get_data())

            rgbd = make_rgbd(color, depth_raw, scale)
            odom_ok, _conf = tracker.step(rgbd)
            odom_ok_last = odom_ok
            if odom_ok:
                _, x_cam, z_cam = compute_scan(depth_raw, scale, intr)
                integrate_scan(world, tracker.pose, x_cam, z_cam)
                near_m, near_b = nearest_in_cone(x_cam, z_cam,
                                                 FORWARD_CONE_DEG)
            else:
                near_m, near_b = float("nan"), float("nan")

            now = time.time()
            dt = now - last_t
            last_t = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0.0)

            world_img = render_world(world, tracker, near_m, near_b,
                                     fps, odom_ok)
            depth_img = colorize_depth(depth_raw, scale)

            CAM_H = 360
            def fit(im):
                h, w = im.shape[:2]
                return cv2.resize(im, (int(w * CAM_H / h), CAM_H))
            cv2.imshow(win_cam, np.hstack([fit(color), fit(depth_img)]))
            cv2.imshow(win, world_img)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("s"):
                save_map(world, tracker, out_dir)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(out_dir / f"color_{stamp}.png"), color)
                cv2.imwrite(str(out_dir / f"depth_{stamp}.png"), depth_img)
            if k == ord("r"):
                world.reset()
                tracker.reset()
                print("[reset] world map cleared and pose zeroed")
    finally:
        pipe.stop()
        cv2.destroyAllWindows()
    save_map(world, tracker, out_dir)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--headless", type=float, default=0.0, metavar="SEC",
                    help="capture SEC seconds, save world map PNG + npy, exit")
    args = ap.parse_args()
    if args.headless > 0:
        return run_headless(args.headless)
    return run_viewer()


if __name__ == "__main__":
    raise SystemExit(main())
