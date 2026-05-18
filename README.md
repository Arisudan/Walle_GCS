# Walle GCS — MAX edition (`walle_gcs_max.py`)

Ground Control Station for a Betaflight + ELRS FPV drone. Talks to a RadioMaster
TX over USB (CRSF telemetry), shows the DJI O4 video feed from an HDMI capture
card, and runs a stack of AI engines (object detection + depth, face
recognition, human tracking, 2D SLAM, 3D reconstruction) on the live feed.

The **MAX edition** is the same GCS as `walle_gcs.py` but wired to the bigger,
more accurate AI models. It keeps its own config files so the two builds don't
overwrite each other's settings.

> ⚠️ This repo is mid-reorganization. `walle_gcs_max.py` was recently moved into
> this `walle_gcs/` folder; `slam_engine.py` and the model/data files still live
> in the project root (`rosetta/`). The notes below reflect the **current**
> layout. See [Redundant / questionable references](#redundant--questionable-references)
> at the bottom.

---

## How to run it

Use the launcher, not `python walle_gcs_max.py` directly — the launcher `cd`s
into the project root first so the bare relative paths (`yolov8n-pose.pt`, the
`human_tracker_v3/` package, `slam_engine.py`) resolve:

```bash
./walle-gcs-max            # from rosetta/walle_gcs/
```

`walle-gcs-max.desktop` is the double-click-from-file-manager version of the
same thing. It runs the project venv at `rosetta/venv/bin/python`.

---

## Files it uses

### 1. Sibling Python modules (imported directly)

These four `import` lines at the top of `walle_gcs_max.py` are the AI engine
back-ends. Three live next to it in `walle_gcs/`; one lives in the project root.

| Module | Class(es) imported | What it does |
|---|---|---|
| `face_recognition_engine.py` | `FaceRecognitionEngine` | SCRFD face detection + ArcFace embeddings (InsightFace `buffalo_l`, all ONNX). Async worker thread. |
| `human_tracker_engine.py` | `HumanTrackerEngine` | Async wrapper around the `human_tracker_v3/` package — YOLOv8-pose + DeepSort + Kalman trajectory prediction. |
| `slam_engine.py` *(in project root)* | `SlamEngine`, `SlamTab` | GPU 2D SLAM — Depth Anything V2 + ORB keyframe VO + occupancy grid. `walle_gcs_max.py` adds the project root to `sys.path` so this import resolves. |
| `reconstruction_engine.py` | `ReconstructionTab` | Subprocess wrapper around external **LingBot-Map** `demo.py` for streaming 3D reconstruction. Pure subprocess + Qt — imports no torch itself. |

### 2. Config & data files (read/written at runtime)

| File | Used by | Notes |
|---|---|---|
| `walle_settings_max.json` | `SettingsTab.load_settings` / `_save_settings`, `_toggle_detection` | App settings + alert config. Written on save. MAX-specific copy. |
| `walle_users_max.json` | `_load_users`, `_save_users`, `_verify_password` (`USERS_FILE`) | User accounts (salted SHA-256 hashes). Auto-created with a default `admin` user if missing. MAX-specific copy. |
| `embeddings_arcface.pkl` | `face_recognition_engine.py` (`EMBEDDING_CACHE_FILE`) | Cached face embeddings, keyed next to the engine module. |
| `known_faces/` | `face_recognition_engine.py` (`KNOWN_FACES_DIR`) | Source images for face recognition. Rebuilds the `.pkl` cache when changed. |

### 3. Pulled in indirectly by the engines (must exist in the project root)

`walle_gcs_max.py` never names these, but the engines it imports do — and the
launcher runs from the project root so they resolve:

- `human_tracker_v3/` — the tracker package (`config/`, `core/`, `ui/` sub-modules). Required by `human_tracker_engine.py`.
- `yolov8n-pose.pt` — default YOLOv8-pose weights for the human tracker.
- `gpu_2d_slam.py` + the MASt3R-SLAM checkout — `slam_engine.py` vendors its `OccupancyGrid2D` / `DepthAnythingV2Wrap` / `KeyframeVO` classes from here.
- An external `lingbot-map/` checkout (with its own venv + `checkpoints/lingbot-map-long.pt`) — `reconstruction_engine.py` shells out to it. Optional; the UI flags it if missing.

### 4. AI models — downloaded, not local files

The detection/depth models are pulled from HuggingFace on first run and cached
by `transformers` / `rfdetr`. They are **not** the `.pth` files sitting in the
repo root:

- **RF-DETR Medium** — `RFDETRMedium()`, called with no weights path → library default checkpoint.
- **Depth Anything V2 Metric-Indoor-Base** — `depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf` (the `"Base"` entry in `DEPTH_MODELS`).
- **InsightFace `buffalo_l`** — downloaded by `face_recognition_engine.py`.

### 5. Hardware / runtime I/O

| Device | How |
|---|---|
| RadioMaster TX (CRSF telemetry) | USB serial, auto-baud probe (400000/420000/921600/…), `SerialReader` thread |
| ESP32 PPM generator | USB serial @ 115200; `_send_channels_to_esp32` writes `roll,pitch,throttle,yaw,switch`. Companion firmware: `ppm_generator.ino` (project root) |
| DJI O4 video | HDMI capture card at `/dev/video*`, `VideoCapture` thread |
| Joystick / gamepad | `pygame.joystick`, polled by `_poll_joystick` |

### 6. Files it writes

- `~/walle_snapshot_<timestamp>.png` — snapshots (`_take_snapshot`)
- User-chosen `.csv` exports — detection events (`_export_events`) and flight logs (`_export_csv`)
- Recorded video — `_toggle_recording` / `_export_video`

---

## Functions & classes (file map)

`walle_gcs_max.py` is ~4400 lines. Logical groupings:

### Protocol & I/O threads
- **`CRSFTelemetry`** — holds parsed telemetry state; `parse_frame()` decodes GPS / battery / link / attitude / vario / RC-channel / flight-mode CRSF frames.
- **`SerialReader(QThread)`** — background CRSF reader; `run()` auto-detects baud (`_find_sync`, `_count_crsf_frames`), parses frames, emits `telemetry_updated`.
- **`VideoCapture(QThread)`** — grabs frames from the capture card, emits `frame_ready`.

### AI detection + depth engine
- Module helpers: `_normalize_depth`, `_colorize_depth`, `_zone_closeness`, `_depth_decision` — depth map → colorized panel + a STOP/AVOID/CLEAR zone decision (display only — see redundancy notes).
- **`ObjectDetectionEngine(QObject)`** — RF-DETR + Depth Anything V2 in worker threads. `load_models`, `_depth_worker` / `_predict_depth`, `_detection_worker`, `process_frame` (non-blocking, latest-frame-wins), `_sample_depth` / `_compute_distance` (per-box distance estimate), `_draw_detections`, `get_depth_map`.

### Joystick
- **`JoystickCalibration`** — per-axis min/center/max; `apply()` maps raw → calibrated.
- **`CalibrationDialog`**, **`JoystickMonitorDialog`** — calibration wizard + live axis/channel monitor.

### Custom Qt widgets
- **`StickWidget`**, **`ChannelBar`**, **`VirtualStickPad`** — control-input visualizers.
- **`NotificationToast`** — transient toast messages.
- **`VideoFeedWidget(QLabel)`** — the video pane; `display_frame` + the HUD painters (`_paint_hud`, `_draw_artificial_horizon`, `_draw_compass_tape`, `_draw_altitude_tape`, `_draw_speed_tape`).

### Tabs
- **`ControlPanelTab`** — arm/safety controls, preflight status, telemetry readout.
- **`DetectionFeedTab`** — live AI event log, stat cards, filters, CSV export.
- **`FlightLogsTab`** — flight history table, stats, CSV/video export.
- **`SettingsTab`** — settings + user management UI; `load_settings` / `_save_settings`, `_add_user` / `_delete_user` / `_change_password`.
- (`SlamTab` and `ReconstructionTab` come from the engine modules.)

### Auth
- `_hash_password`, `_load_users`, `_save_users`, `_verify_password` — salted SHA-256 against `walle_users_max.json`.
- **`LoginDialog(QDialog)`** — login screen; `_attempt_login`.

### Application shell
- **`SidebarNav(QFrame)`** — left-rail navigation.
- **`MainWindow(QMainWindow)`** — ties everything together: builds the UI, owns the engines, the connect/disconnect toggles (`_toggle_video`, `_toggle_serial`, `_toggle_esp32`), the per-feature toggles (`_toggle_detection`, `_toggle_depth_overlay`, `_toggle_face_rec`, `_toggle_track_human`), the frame pipeline (`_update_video`), telemetry routing (`_update_telemetry`), and the timers.
- **`main()`** — boots Qt, shows `LoginDialog`, then `MainWindow`.

---

## Redundant / questionable references

Things in this file (and around it) that look like duplication, dead code, or
stale references — worth a cleanup pass:

1. **`walle_gcs.py` vs `walle_gcs_max.py` are ~99% identical.** Only ~40 lines
   differ: the model classes (`RFDETRMedium` vs `RFDETRBase`), the depth variant
   (`"Base"` vs `"Small"`), the version string, and the four engine imports.
   Every bug fix currently has to be made twice. This could collapse into one
   file with a `--max` flag or a config key.

2. **`DEPTH_MODELS` still defines a `"Small"` entry** even though the MAX edition
   is hard-wired to `depth_variant="Base"`. The `"Small"` line is dead in this
   file.

3. **`DANGER_CLASSES` includes `"stairs"` and `"obstacle"`** — neither is a COCO
   class, and `COCO_NAMES` never produces those strings, so those two entries
   can never match a detection. Dead set members.

4. **The depth zone-avoidance decision is display-only.** `_depth_decision`
   produces a `STOP` / `AVOID_L` / `AVOID_R` / `CLEAR` mode, but `_update_video`
   only draws it as a banner and stores it on `video_widget.hud_depth_mode` — it
   never feeds back into the RC channels or the ESP32 output. If actual
   obstacle avoidance is the goal, that wiring is missing; if it's just an
   advisory HUD, the naming oversells it.

5. **The `.pth` files in the repo root are not used by this script.**
   `rf-detr-medium.pth` (and `rf-detr-base.pth`) are ~400 MB each, but
   `RFDETRMedium()` is called with no weights argument, so it downloads its own
   checkpoint. Those files look like leftovers from `train_drone_rfdetr.py` and
   are unrelated to running the GCS.

6. **Many sibling prototype scripts in the project root are *not* imported here.**
   `gpu_2d_slam.py`, `gpu_2d_slam_da3.py`, `slam3d.py`, `mono_slam.py`,
   `felix_vo.py`, `mona_fast.py`, `obstacle_avoidance.py`, `indoor_avoidance.py`,
   `drone_mavlink_avoidance.py`, `FaceRecNMK.py`, `drone_dash_v2.py`, etc. —
   `walle_gcs_max.py` imports none of them. The only real link is that
   `slam_engine.py` *vendors* (copy-pastes) classes out of `gpu_2d_slam.py`, and
   `FaceRecNMK.py` appears to be the predecessor of `face_recognition_engine.py`.
   Lots of overlapping/superseded experiments share the folder; it's not obvious
   which are live.

7. **Two near-identical `DJI_0029_tracked` videos** sit in the project root —
   `DJI_0029_tracked.mp4` and `DJI_0029_tracked.MP4` (~46–51 MB each, same name,
   different case extension). Almost certainly accidental duplicates.

8. **The non-MAX configs sit right next to the MAX ones.** `walle_settings.json`
   and `walle_users.json` are in this folder too, but belong to `walle_gcs.py`,
   not this script. Easy to edit the wrong one — the `_max` suffix is the only
   thing keeping them apart.
