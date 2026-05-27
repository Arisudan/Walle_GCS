#!/usr/bin/env python3
"""
Face Recognition Engine for Walle GCS.

Wraps SCRFD (InsightFace) face detection + DeepFace/ArcFace embeddings with
track-based caching and dynamic far-face zoom. Designed to plug into a Qt
video loop: call ``process_frame(bgr)`` once ``is_ready`` is True.

Heavy model loading runs in a background thread; ``models_loaded(bool)`` is
emitted on completion (Qt queues the signal to the main thread automatically).
"""

import os
import time
import pickle
import threading
from collections import deque, defaultdict

import cv2
import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from core.logging import get_logger

log = get_logger("FaceRec")


# ─── Config ─────────────────────────────────────────────────────────
# buffalo_l ships SCRFD-10g detector + ResNet50 ArcFace recognition,
# all in ONNX (no TensorFlow). The original buffalo_sc + DeepFace path
# triggered TF JIT failures on Blackwell GPUs (sm_120 / RTX 5090).
DETECTOR_NAME = "buffalo_l"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_CACHE_FILE = os.path.join(BASE_DIR, "embeddings_arcface.pkl")
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")
VALID_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

FRAME_SKIP = 3                  # only run detection every Nth frame
DOWNSCALE = 0.8
MIN_FACE_SIZE = 12
MIN_DETECTION_SCORE = 0.3

# Multi-scale detection: each scale costs one full SCRFD forward pass, so
# 4 scales = 4x the per-frame detector cost. [1.0, 2.0] keeps the 1.0x pass
# for normal faces and the 2.0x pass for tiny/far faces (which is what
# FAR_FACE_THRESHOLD targets anyway), giving ~2x speedup vs. the old 4-scale
# sweep with negligible recall loss in flight footage.
ENABLE_MULTISCALE = True
SCALES = [1.0, 2.0]

ENABLE_FAR_FACE_ZOOM = True
FAR_FACE_THRESHOLD = 50
ZOOM_CONTEXT_MULTIPLIER = 2.5
ZOOM_TARGET_SIZE = 160

SIMILARITY_THRESHOLD = 0.50
EMBED_REFRESH_FRAMES = 10
IOU_THRESHOLD = 0.3

ENABLE_QUALITY_FILTER = False
MIN_BLUR_THRESHOLD = 100


# ─── Utility ────────────────────────────────────────────────────────
def _iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    if x2 < x1 or y2 < y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-6)


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


# ─── Helpers ────────────────────────────────────────────────────────
class _FarFaceZoom:
    def __init__(self, mult=ZOOM_CONTEXT_MULTIPLIER, target=ZOOM_TARGET_SIZE):
        self.mult = mult
        self.target = target

    def extract(self, frame, bbox, landmarks):
        x1, y1, x2, y2 = bbox
        fw, fh = x2 - x1, y2 - y1
        size = min(fw, fh)
        if size >= FAR_FACE_THRESHOLD:
            return None, landmarks, False

        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ew, eh = fw * self.mult, fh * self.mult
        nx1 = int(max(0, cx - ew / 2))
        ny1 = int(max(0, cy - eh / 2))
        nx2 = int(min(frame.shape[1], cx + ew / 2))
        ny2 = int(min(frame.shape[0], cy + eh / 2))

        region = frame[ny1:ny2, nx1:nx2].copy()
        if region.size == 0:
            return None, landmarks, False

        new_lm = landmarks.copy().astype(np.float32)
        new_lm[:, 0] -= nx1
        new_lm[:, 1] -= ny1

        aw, ah = nx2 - nx1, ny2 - ny1
        if aw > 0 and ah > 0:
            scale = self.target / max(aw, ah)
            region = cv2.resize(region, (int(aw * scale), int(ah * scale)),
                                interpolation=cv2.INTER_CUBIC)
            new_lm *= scale
        return region, new_lm, True


class _Preproc:
    def __init__(self):
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    def _gamma(self, img):
        luma = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
        g = 1.5 if luma < 100 else 1.2 if luma < 150 else 1.0
        if g == 1.0:
            return img
        inv = 1.0 / g
        tbl = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(img, tbl)

    def process(self, img):
        img = self._gamma(img)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        if img.shape[0] != 112 or img.shape[1] != 112:
            img = cv2.resize(img, (112, 112))
        return img


class _FaceTrack:
    def __init__(self, tid, bbox, score, frame_num):
        self.tid = tid
        self.bbox = bbox
        self.score = score
        self.last_seen = frame_num
        self.last_embed_frame = -999
        self.cached_emb = None
        self.names_hist = deque(maxlen=8)
        self.sims_hist = deque(maxlen=8)
        self.stable_name = "Unknown"
        self.stable_sim = 0.0
        self.frames_seen = 0
        self.zoom_used = False
        self.zoom_count = 0
        self.bbox_history = deque(maxlen=5)
        self.bbox_history.append(bbox)

    def _bbox_changed(self):
        if len(self.bbox_history) < 2:
            return False
        cur, prev = self.bbox_history[-1], self.bbox_history[-2]
        ca = (cur[2] - cur[0]) * (cur[3] - cur[1])
        pa = (prev[2] - prev[0]) * (prev[3] - prev[1])
        return abs(ca - pa) / (pa + 1e-6) > 0.15

    def needs_embed(self, frame_num):
        if frame_num - self.last_embed_frame >= EMBED_REFRESH_FRAMES:
            return True
        return self._bbox_changed()

    def update_embed(self, emb, name, sim, frame_num, zoomed=False):
        self.cached_emb = emb
        self.last_embed_frame = frame_num
        self.zoom_used = zoomed
        if zoomed:
            self.zoom_count += 1
        self.names_hist.append(name)
        self.sims_hist.append(sim)
        if len(self.names_hist) >= 2:
            scores = defaultdict(float)
            for i, (n, s) in enumerate(zip(self.names_hist, self.sims_hist)):
                w = (i + 1) / len(self.names_hist)
                scores[n] += s * w
            self.stable_name = max(scores.items(), key=lambda x: x[1])[0]
            self.stable_sim = float(np.mean(list(self.sims_hist)[-3:]))

    def update_bbox(self, bbox, score, frame_num):
        self.bbox = bbox
        self.score = score
        self.bbox_history.append(bbox)
        self.last_seen = frame_num
        self.frames_seen += 1


# ─── Engine ─────────────────────────────────────────────────────────
class FaceRecognitionEngine(QObject):
    """SCRFD + ArcFace face recognition with track caching + far-face zoom.

    Emits ``models_loaded(bool)`` when ``load_models()`` finishes. Call
    ``process_frame(bgr)`` once ``is_ready`` is True; returns an annotated
    BGR frame (same shape as input).
    """

    models_loaded = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self._loaded = False
        self._loading = False

        self._detector = None
        self._rec_model = None
        self._face_align = None

        self._zoom = _FarFaceZoom()
        self._preproc = _Preproc()

        # Shared state — accessed from both worker thread and GUI thread.
        self._lock = threading.Lock()
        self._tracks = {}
        self._next_tid = 0
        self._frame_num = 0
        self._frame_idx = 0
        self._known = {}

        # Async inference worker
        self._infer_input_frame = None
        self._infer_pending = threading.Event()
        self._infer_stop = threading.Event()
        self._infer_thread = None

        self._last_proc_ms = 0.0
        self._stats = {
            'faces_detected': 0,
            'far_faces': 0,
            'zoom_applied': 0,
            'embeddings_generated': 0,
            'embeddings_cached': 0,
            'quality_rejected': 0,
        }

    @property
    def is_ready(self):
        return self._loaded

    @property
    def known_count(self):
        return len(self._known)

    # ── Loading ───────────────────────────────────────────────────
    def load_models(self):
        if self._loaded or self._loading:
            return
        self._loading = True

        def _run():
            try:
                os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
                from insightface.app import FaceAnalysis
                from insightface.utils import face_align
                self._face_align = face_align

                log.info("Initializing %s (SCRFD + ArcFace ONNX) ...",
                         DETECTOR_NAME)
                # Only load the two ONNX models we actually use. The default
                # FaceAnalysis bundle also pulls in landmark_2d_106, 1k3d68,
                # and genderage — they cost ~150 MB VRAM and add GPU contention
                # for no benefit since we never read those face attributes.
                self._detector = FaceAnalysis(
                    name=DETECTOR_NAME,
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                    allowed_modules=['detection', 'recognition'],
                )
                # 640x640 is the canonical SCRFD-10g training size and is
                # ~2.2x faster than 960x960 with comparable mAP for the face
                # sizes we see at typical drone altitudes. The multi-scale
                # sweep above already covers tiny faces via the 2.0x pass.
                self._detector.prepare(ctx_id=0, det_size=(640, 640))
                self._rec_model = self._detector.models.get('recognition')
                if self._rec_model is None:
                    raise RuntimeError(
                        f"{DETECTOR_NAME} did not load a recognition model")

                log.info("Loading known-faces cache ...")
                self._known = self._load_or_build_cache()
                log.info("Ready — %d known faces", len(self._known))

                # Warmup: a single dummy detect+embed pass primes the ONNX
                # CUDA kernels so the first live frame doesn't pay the
                # algo-selection cost (which can be 2-5 s on first invocation).
                try:
                    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                    self._detector.get(dummy)
                    # Touch the recognition model too — its kernels load lazily
                    # on first call to get_feat().
                    blank = np.zeros((112, 112, 3), dtype=np.uint8)
                    self._rec_model.get_feat(self._preproc.process(blank))
                    log.info("Warmup complete")
                except Exception as e:
                    log.warning("Warmup skipped: %s", e)

                # Start the async inference worker so process_frame() never blocks.
                self._infer_thread = threading.Thread(
                    target=self._inference_worker, daemon=True)
                self._infer_thread.start()

                self._loaded = True
                self._loading = False
                self.models_loaded.emit(True)
            except Exception as e:
                log.exception("Load failed: %s", e)
                self._loading = False
                self.models_loaded.emit(False)

        threading.Thread(target=_run, daemon=True).start()

    def _load_or_build_cache(self):
        if os.path.exists(EMBEDDING_CACHE_FILE):
            try:
                with open(EMBEDDING_CACHE_FILE, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                log.warning("Cache load failed (%s); rebuilding", e)
        return self._build_cache()

    def _build_cache(self):
        cache = {}
        if not os.path.isdir(KNOWN_FACES_DIR):
            return cache
        for f in sorted(os.listdir(KNOWN_FACES_DIR)):
            if not f.lower().endswith(VALID_EXT):
                continue
            path = os.path.join(KNOWN_FACES_DIR, f)
            img = cv2.imread(path)
            if img is None:
                continue
            try:
                faces = self._detector.get(img)
                if not faces:
                    log.warning("  %s: no face found", f)
                    continue
                face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                aligned = self._face_align.norm_crop(img, face.kps)
                pre = self._preproc.process(aligned)
                emb = np.asarray(self._rec_model.get_feat(pre)).flatten()
                cache[os.path.splitext(f)[0]] = emb
                log.info("  %s OK", f)
            except Exception as e:
                log.error("  %s FAILED: %s", f, e)
        if cache:
            try:
                with open(EMBEDDING_CACHE_FILE, "wb") as f:
                    pickle.dump(cache, f)
            except Exception as e:
                log.error("Could not save cache: %s", e)
        return cache

    # ── Matching ──────────────────────────────────────────────────
    def _match(self, emb):
        best_name = "Unknown"
        best_sim = -1.0
        for name, kemb in self._known.items():
            s = _cosine(emb, kemb)
            if s > best_sim:
                best_sim = s
                best_name = name
        return best_name, best_sim

    # ── Per-frame processing (NON-BLOCKING) ───────────────────────
    def process_frame(self, frame):
        """Submit frame for async inference, return current frame with cached overlays.

        Inference runs on the worker thread. The GUI thread only does
        a fast cv2 draw using whatever tracks the worker most recently produced,
        so this returns in O(1) regardless of model speed.
        """
        if not self._loaded or frame is None:
            return frame

        # Submit the latest frame to the inference worker. If the worker is
        # still busy on a previous frame, we just overwrite — that's the
        # "latest frame wins" semantics that keeps display FPS at camera FPS.
        with self._lock:
            self._infer_input_frame = frame
        self._infer_pending.set()

        return self._draw(frame)

    def _inference_worker(self):
        """Background loop: pulls latest input frame, runs detection + embedding."""
        while not self._infer_stop.is_set():
            self._infer_pending.wait(timeout=0.5)
            self._infer_pending.clear()

            with self._lock:
                frame = self._infer_input_frame
                self._infer_input_frame = None
            if frame is None:
                continue

            # Honor FRAME_SKIP: only run inference every Nth submitted frame.
            self._frame_num += 1
            self._frame_idx += 1
            if (self._frame_idx % FRAME_SKIP) != 0:
                continue

            t0 = time.perf_counter()
            try:
                self._run_detection(frame)
            except Exception as e:
                log.error("worker error: %s", e)
            self._last_proc_ms = (time.perf_counter() - t0) * 1000

            # Drop stale tracks under lock so _draw can iterate safely.
            with self._lock:
                stale = [tid for tid, tr in self._tracks.items()
                         if self._frame_num - tr.last_seen > 20]
                for tid in stale:
                    del self._tracks[tid]

    def stop(self):
        """Signal the inference worker to exit. Safe to call multiple times."""
        self._infer_stop.set()
        self._infer_pending.set()

    def _run_detection(self, frame):
        small = cv2.resize(frame, None, fx=DOWNSCALE, fy=DOWNSCALE)

        all_faces = []
        for sc in (SCALES if ENABLE_MULTISCALE else [1.0]):
            scaled = cv2.resize(small, None, fx=sc, fy=sc) if sc != 1.0 else small
            try:
                for face in self._detector.get(scaled):
                    bbox = face.bbox.copy()
                    kps = face.kps.copy()
                    if sc != 1.0:
                        bbox /= sc
                        kps /= sc
                    all_faces.append({
                        'bbox': bbox, 'kps': kps,
                        'score': float(face.det_score),
                    })
            except Exception:
                pass

        # NMS
        if len(all_faces) > 1:
            all_faces.sort(key=lambda x: x['score'], reverse=True)
            kept = []
            for f in all_faces:
                if any(_iou(f['bbox'], k['bbox']) > 0.5 for k in kept):
                    continue
                kept.append(f)
            all_faces = kept

        detections = []
        for f in all_faces:
            if f['score'] < MIN_DETECTION_SCORE:
                continue
            x1 = int(f['bbox'][0] / DOWNSCALE)
            y1 = int(f['bbox'][1] / DOWNSCALE)
            x2 = int(f['bbox'][2] / DOWNSCALE)
            y2 = int(f['bbox'][3] / DOWNSCALE)
            w, h = x2 - x1, y2 - y1
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue
            self._stats['faces_detected'] += 1
            if w < 40 or h < 40:
                self._stats['far_faces'] += 1

            kps_full = f['kps'] / DOWNSCALE
            bbox_full = (x1, y1, x2, y2)

            if ENABLE_FAR_FACE_ZOOM:
                region, adj_kps, zoomed = self._zoom.extract(frame, bbox_full, kps_full)
                if region is not None and zoomed:
                    aligned = self._face_align.norm_crop(region, adj_kps)
                    self._stats['zoom_applied'] += 1
                else:
                    aligned = self._face_align.norm_crop(frame, kps_full)
                    zoomed = False
            else:
                aligned = self._face_align.norm_crop(frame, kps_full)
                zoomed = False

            if ENABLE_QUALITY_FILTER:
                gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_BLUR_THRESHOLD:
                    self._stats['quality_rejected'] += 1
                    continue

            pre = self._preproc.process(aligned)
            detections.append((bbox_full, pre, f['score'], zoomed))

        # Track update — under lock since GUI thread reads tracks while drawing.
        track_results = []
        with self._lock:
            for bbox, pre, score, zoomed in detections:
                best_iou = 0.0
                best_tid = None
                for tid, tr in self._tracks.items():
                    i = _iou(bbox, tr.bbox)
                    if i > best_iou and i > IOU_THRESHOLD:
                        best_iou = i
                        best_tid = tid
                if best_tid is not None:
                    tr = self._tracks[best_tid]
                    tr.update_bbox(bbox, score, self._frame_num)
                    track_results.append(
                        (tr, pre, tr.needs_embed(self._frame_num), zoomed))
                else:
                    tr = _FaceTrack(self._next_tid, bbox, score, self._frame_num)
                    self._tracks[self._next_tid] = tr
                    self._next_tid += 1
                    track_results.append((tr, pre, True, zoomed))

        # Embed only the ones that need it
        for tr, pre, need_embed, zoomed in track_results:
            if need_embed:
                try:
                    emb = np.asarray(self._rec_model.get_feat(pre)).flatten()
                    self._stats['embeddings_generated'] += 1
                    name, sim = self._match(emb)
                    tr.update_embed(emb, name, sim, self._frame_num, zoomed)
                except Exception as e:
                    log.error("embed error: %s", e)
            else:
                self._stats['embeddings_cached'] += 1

    def _draw(self, frame):
        # Snapshot tracks under lock — the worker may be mutating self._tracks
        # right now. We copy refs (cheap) and then iterate without the lock.
        with self._lock:
            tracks_snapshot = list(self._tracks.values())
        for tr in tracks_snapshot:
            if tr.frames_seen < 2:
                continue
            label = tr.stable_name
            sim = tr.stable_sim
            if sim >= 0.65:
                color = (0, 255, 0); conf = "HIGH"
            elif sim >= 0.55:
                color = (0, 200, 255); conf = "MED"
            elif sim >= SIMILARITY_THRESHOLD:
                color = (100, 200, 255); conf = "LOW"
            else:
                label = "UNKNOWN"; color = (0, 0, 255); conf = "?"

            x1, y1, x2, y2 = tr.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            zi = " [Z]" if tr.zoom_used else ""
            text = f"{label} ({conf}){zi}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
            cv2.putText(frame, text, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            fs = min(x2 - x1, y2 - y1)
            cv2.putText(frame, f"sim {sim:.2f} | {fs}px",
                        (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (220, 220, 220), 1, cv2.LINE_AA)
        return frame

    # ── External API ──────────────────────────────────────────────
    def get_stats(self):
        s = dict(self._stats)
        s['proc_ms'] = self._last_proc_ms
        s['proc_fps'] = (1000.0 / self._last_proc_ms) if self._last_proc_ms > 0 else 0.0
        s['known_count'] = len(self._known)
        s['tracks'] = len(self._tracks)
        return s

    def reload_known_faces(self):
        """Force rebuild of the embeddings cache from KNOWN_FACES_DIR."""
        try:
            if os.path.exists(EMBEDDING_CACHE_FILE):
                os.remove(EMBEDDING_CACHE_FILE)
        except Exception:
            pass
        self._known = self._build_cache()
        return len(self._known)
