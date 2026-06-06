# SLAM Output Evaluation Protocol — Walle GCS

Evaluation framework for the two map products this GCS generates:

| Product | Engine | Artifact(s) | Format |
|---|---|---|---|
| **2D occupancy map** | `slam_engine.py` (DA V2 depth + ORB keyframe VO + GPU log-odds grid) | `out_mapping/map_<ts>.png`, `out_mapping/map_<ts>_logodds.npy` | 8-bit render + float32 log-odds grid |
| **3D reconstruction** | `reconstruction_engine.py` (Lingbot streaming map model) | `<name>.ply` | Fused point cloud (XYZ + RGB + conf) |

The goal of this document is to turn "the map looks about right" into a repeatable, numeric pass/fail. Each section gives: **what to measure → how to measure it → acceptance threshold**.

---

## 0. Test set & ground truth

Before any number means anything, fix the inputs.

**Test sequences** — keep a frozen set so scores are comparable across builds. For each recorded run, log:

- Sensor (camera model, resolution, HFOV used in `make_intrinsics`).
- Scene class (`indoor` / `outdoor`) and depth variant (`small` / `base`) — must match the DA V2 model selected.
- Trajectory type: loop (returns to start), out-and-back, straight corridor, exploratory.
- Lighting, motion speed, presence of dynamic objects (people/vehicles).

**Ground truth sources** (use whatever you can get, in order of preference):

1. **Surveyed metric** — tape/laser-measured distances between fixed landmarks in the scene. The cheapest reliable GT for 2D.
2. **Reference scan** — a LiDAR or photogrammetry scan of the space, for 3D comparison.
3. **Loop-closure self-consistency** — if the trajectory returns to its start, the start/end pose gap *is* a free error signal (see §1.2). No external GT needed.
4. **CAD / floor plan** — building drawings give wall positions and room dimensions for 2D occupancy overlay.

> Record at least: **1 loop sequence**, **1 out-and-back**, **1 long straight run** per scene class. Drift behaves differently in each.

---

## 1. 2D occupancy map evaluation

The 2D map is a log-odds grid. Load the raw grid, not the PNG, for metrics:

```python
import numpy as np
logodds = np.load("out_mapping/map_20260528_145933_logodds.npy")
prob = 1.0 / (1.0 + np.exp(-logodds))   # sigmoid → occupancy probability
occ  = prob > 0.60      # matches "Cells known" threshold in slam_engine.py
free = prob < 0.40      # matches free_thresh in OccupancyGrid2D.render
unknown = ~(occ | free)
```

### 1.1 Geometric accuracy

| Metric | How to compute | Target |
|---|---|---|
| **Scale error** | Measure a known landmark-to-landmark distance on the grid (cells × resolution m/cell) vs. surveyed truth. `(d_map − d_true)/d_true`. | **≤ 2%** |
| **Wall straightness** | Fit a line to a known-straight wall's occupied cells; RMS perpendicular residual. | **≤ 1 cell** |
| **Squareness** | Angle between two walls known to be 90°. | **±3°** |
| **Map completeness** | Known-traversed area that ended up `occ ∨ free` vs. `unknown`. | **≥ 90%** of swept area |

### 1.2 Trajectory / drift accuracy

The `stats` dict emits live pose (`Pose x/y/z`, `Yaw`, `Distance`). Log it per frame to a CSV during the run.

| Metric | How to compute | Target |
|---|---|---|
| **Loop-closure gap** | Euclidean distance between first and last pose on a loop sequence, ÷ total `Distance` travelled. | **≤ 2%** of path length |
| **Yaw drift** | `Yaw` at loop end vs. start (should be ~0 after a full loop back to start heading). | **≤ 5°** |
| **ATE (if GT track)** | Align estimated track to GT (Umeyama), RMS position error. | scene-dependent; report it |
| **Scale consistency** | `Distance` accumulated vs. surveyed path length. | **≤ 3%** |

### 1.3 VO health (process quality, not output quality)

These come straight from the `stats` dict and explain *why* a map is good or bad:

- **Inlier ratio** = `Inliers / Matches`. Sustained **< 0.5** → unreliable pose, expect drift.
- **VO status** — count frames in `LOST` / fallback states. **< 5%** of frames lost is healthy.
- **Keyframe cadence** — `KF id` growth vs. distance. Too few KFs on a long path → undersampling.
- **Tracking continuity** — any frame gaps where the worker dropped frames under CPU load (cross-ref the CPU bottleneck notes).

### 1.4 Occupancy quality

| Metric | How to compute | Target |
|---|---|---|
| **Free-space precision** | Sample cells marked `free`; fraction actually traversable in GT. | **≥ 95%** (false-free is dangerous) |
| **Obstacle recall** | Known obstacles that appear as `occ`. | **≥ 90%** |
| **Ghosting / smearing** | Visual: dynamic objects (people) leaving occupied trails. Count smear artifacts. | qualitative, note count |
| **Floor handling** | `Floor y` stable across run (not jumping). | < 0.2 m variance |

---

## 2. 3D map (point cloud) evaluation

Load the `.ply` (e.g. with `open3d` or `numpy`+`plyfile`). Note the demo logs `Exported PLY (N points, conf>X)` — record **N** and the confidence cutoff.

### 2.1 Geometric accuracy

| Metric | How to compute | Target |
|---|---|---|
| **Cloud-to-reference distance** | Register `.ply` to reference scan (ICP), compute per-point nearest-neighbour distance. Report **mean** and **RMS**. | mean **≤ 3 cm** indoor / scene-scaled outdoor |
| **Scale error** | Known object dimension (door height, table length) measured in cloud vs. truth. | **≤ 2%** |
| **Planarity** | Fit a plane to a known-flat surface (wall/floor); RMS residual. | **≤ 2 cm** |
| **Completeness** | Fraction of reference surface within τ of a reconstructed point. | **≥ 85%** |
| **Outlier rate** | Points with NN-distance > 5× median (floaters/flyers). | **≤ 2%** |

### 2.2 Density & coverage

| Metric | Target |
|---|---|
| **Point density** (pts/m² on surfaces) | uniform; no large holes in swept regions |
| **Hole count** | gaps > 0.25 m² in observed surfaces — list and screenshot each |
| **Sky/background leakage** | with sky filter on, count stray sky points | near zero |

### 2.3 Color / confidence

- **Color fidelity** — RGB on the cloud matches scene (no smearing across depth edges).
- **Confidence distribution** — histogram the per-point conf; report fraction below the export cutoff that still survived.

---

## 3. Cross-cutting: performance & robustness

| Metric | Source | Target |
|---|---|---|
| **Throughput** | `FPS` in `stats` | ≥ real-time target for the scene |
| **CPU / GPU load** | external monitor (py-spy / nvidia-smi) | within budget; no thermal throttle |
| **Frame-drop rate** | worker-busy drops in `process_frame` | **< 10%** |
| **Determinism** | run same sequence ×3; spread of key metrics | scale error spread **< 1%** |
| **Failure recovery** | inject a hard turn / occlusion; does VO re-acquire? | recovers within N frames, logged |

---

## 4. Scorecard (fill one per run)

```
Run ID:            ____________________   Date: __________
Sequence:          ____________________   Scene: indoor / outdoor
Depth model:       ______ / small|base    HFOV: ______°
Artifact(s):       map_________.npy  /  _________.ply
Operator:          ____________________

── 2D OCCUPANCY ───────────────────────────────  target   measured  P/F
 Scale error                                       ≤2%      ______   ___
 Wall straightness (RMS cells)                     ≤1       ______   ___
 Loop-closure gap (% path)                         ≤2%      ______   ___
 Yaw drift (deg)                                   ≤5       ______   ___
 Map completeness (%)                              ≥90      ______   ___
 Free-space precision (%)                          ≥95      ______   ___
 Obstacle recall (%)                               ≥90      ______   ___
 Inlier ratio (median)                             ≥0.5     ______   ___
 Frames lost (%)                                   <5       ______   ___

── 3D POINT CLOUD ─────────────────────────────  target   measured  P/F
 Cloud-to-ref RMS (cm)                             ≤3       ______   ___
 Scale error (%)                                   ≤2       ______   ___
 Planarity RMS (cm)                                ≤2       ______   ___
 Completeness (%)                                  ≥85      ______   ___
 Outlier rate (%)                                  ≤2       ______   ___
 Point count (N)                                    —       ______    —

── PERFORMANCE ────────────────────────────────  target   measured  P/F
 FPS                                                —       ______   ___
 Frame-drop rate (%)                               <10      ______   ___
 Determinism spread (%)                            <1       ______   ___

OVERALL:   PASS / CONDITIONAL / FAIL
Notes / artifacts attached:
________________________________________________________________
```

**Verdict rule of thumb:** all geometric-accuracy rows green → PASS. Any single safety row red (free-space precision, obstacle recall, cloud-to-ref) → FAIL regardless of the rest. Process rows (inliers, FPS) red but outputs green → CONDITIONAL (works but fragile).

---

## 5. Suggested evaluation harness

A small reusable scorer keeps this objective and repeatable. Sketch:

```python
# eval_slam.py  (proposed — pairs with Eval_models/eval.ipynb)
def eval_occupancy(logodds_path, gt):
    grid = np.load(logodds_path)
    prob = 1/(1+np.exp(-grid))
    occ, free = prob > 0.60, prob < 0.40
    return {
        "scale_error":      scale_vs_landmarks(occ, gt.landmarks),
        "wall_rms_cells":   wall_straightness(occ, gt.walls),
        "completeness":     (occ | free)[gt.swept_mask].mean(),
        "free_precision":   traversable_fraction(free, gt.free_mask),
        "obstacle_recall":  recall(occ, gt.obstacle_mask),
    }

def eval_pointcloud(ply_path, ref_path):
    pc, ref = load_ply(ply_path), load_ply(ref_path)
    pc = icp_register(pc, ref)
    d = nn_distances(pc, ref)
    return {
        "rms_cm":       100*np.sqrt((d**2).mean()),
        "completeness": coverage(ref, pc, tau=0.03),
        "outlier_rate": (d > 5*np.median(d)).mean(),
        "n_points":     len(pc),
    }
```

Drop the per-run dicts into a CSV and you get build-over-build trend lines for free — the same regression-net philosophy as `tests/`.

---

## Appendix — quick reference: key thresholds in the code

- Occupancy "known" cutoff: `sigmoid(log_odds) > 0.6` (`slam_engine.py`, "Cells known").
- Free-space cutoff: `prob < free_thresh` with `free_thresh = 0.40` (`OccupancyGrid2D.render`).
- Pose / VO telemetry: the `stats` dict emitted by `update_ready` — log it to CSV per frame.
- 3D export line to parse: `Exported PLY (N points, conf>X) to <path>`.
