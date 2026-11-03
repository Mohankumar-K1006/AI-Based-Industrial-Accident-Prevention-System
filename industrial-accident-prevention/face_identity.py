# -*- coding: utf-8 -*-
"""face_identity.py — Real-time face identification for danger-zone alerts.

FIXES APPLIED (v2):
  1. Uses YOLO pose keypoints (nose, eyes, ears) to extract a properly
     centred face crop instead of the crude "top 40% of body bbox" which
     produced wide, flat strips that DeepFace couldn't match.
  2. Uses SQLite database (employee_db.py) as the authoritative source of
     face templates, ensuring persistence across restarts.
  3. Matches against ALL templates per employee (multiple enrollment photos)
     and takes the MINIMUM distance, improving recognition accuracy.
  4. Caches results per worker-tracking-ID to avoid running DeepFace on
     every frame.
"""

import json
import os
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── paths ──
_ROOT = Path(__file__).resolve().parent

# ── tuneable parameters ──
RECHECK_INTERVAL = 0.5   # seconds between re-attempts for unidentified workers (fast recheck)
CACHE_TTL = 300.0         # seconds before refreshing a positive match (5 min)
MIN_FACE_SIZE = 30        # minimum face crop dimension (pixels)

# ── import threshold and model name ──
MODEL_NAME = os.environ.get("FACE_MODEL", "Facenet")
MATCH_DISTANCE_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.60"))

# ── safe DeepFace import ──
import builtins
import sys

_real_print = builtins.print

def _safe_print(*args, **kwargs):
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    safe = []
    for a in args:
        s = str(a)
        try:
            s.encode(enc)
            safe.append(s)
        except (UnicodeEncodeError, LookupError):
            safe.append(s.encode("ascii", errors="replace").decode("ascii"))
    try:
        _real_print(*safe, **kwargs)
    except (UnicodeEncodeError, LookupError):
        pass

builtins.print = _safe_print
try:
    from deepface import DeepFace
    _HAS_DF = True
except ImportError:
    _HAS_DF = False
    _real_print("[face_identity] WARNING: deepface not installed.")
except Exception as e:
    _HAS_DF = False
    _real_print(f"[face_identity] WARNING: failed to load deepface: {e}")
finally:
    builtins.print = _real_print


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE STATE
# ══════════════════════════════════════════════════════════════════════════════

_lock = threading.Lock()

# Template cache: list of {employee_id, full_name, embedding}
_templates: list[dict] = []
_templates_loaded_at: float = 0.0
TEMPLATE_RELOAD_INTERVAL = 5.0  # seconds between checking for new templates

# Per-worker identity cache
_identity_cache: dict[int, dict] = {}


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ══════════════════════════════════════════════════════════════════════════════
#  TEMPLATE LOADING (from SQLite)
# ══════════════════════════════════════════════════════════════════════════════

def _load_templates():
    """Load active face templates from SQLite database."""
    global _templates, _templates_loaded_at
    now = time.time()

    # Don't reload too frequently
    if now - _templates_loaded_at < TEMPLATE_RELOAD_INTERVAL:
        return

    try:
        import employee_db
        employee_db.init()
        raw = employee_db.get_all_active_templates()
        _templates = raw
        _templates_loaded_at = now

        # Count unique employees
        emp_ids = set(t["employee_id"] for t in _templates)
        print(f"[{_ts()}][face_identity] Loaded {len(_templates)} template(s) "
              f"for {len(emp_ids)} employee(s) from database")
    except Exception as e:
        print(f"[{_ts()}][face_identity] ERROR loading templates: {e}")
        # If database fails, try legacy pickle as fallback
        _load_templates_fallback()


def _load_templates_fallback():
    """Fallback: load from face_encodings.pkl + employees.json."""
    global _templates, _templates_loaded_at

    pkl_path = _ROOT / "face_encodings.pkl"
    json_path = _ROOT / "employees.json"

    if not pkl_path.exists():
        return

    try:
        with open(pkl_path, "rb") as f:
            encodings = pickle.load(f)

        employees = {}
        if json_path.exists():
            with open(json_path, "r", encoding="utf-8") as f:
                for emp in json.load(f):
                    employees[emp.get("employee_id", "")] = emp

        templates = []
        for eid, emb_raw in encodings.items():
            emb = np.asarray(emb_raw, dtype=np.float64)
            if emb.ndim >= 1 and emb.size > 0 and np.isfinite(emb).all():
                emp = employees.get(eid, {})
                templates.append({
                    "employee_id": eid,
                    "full_name": emp.get("full_name", eid),
                    "embedding": emb,
                    "model_name": MODEL_NAME,
                    "embedding_dim": emb.shape[0],
                })

        _templates = templates
        _templates_loaded_at = time.time()
        print(f"[{_ts()}][face_identity] Fallback: loaded {len(templates)} "
              f"templates from legacy files")
    except Exception as e:
        print(f"[{_ts()}][face_identity] ERROR in fallback loading: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  FACE CROP EXTRACTION (keypoint-based)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_face_crop(frame: np.ndarray, bbox: tuple,
                       keypoints: list = None) -> np.ndarray | None:
    """Extract a face crop using YOLO pose keypoints for accurate centring.

    This is the critical fix: instead of blindly taking "top 40% of body bbox"
    (which produces a wide, flat strip containing mostly shoulders), we use
    the YOLO pose nose/eyes/ears keypoints to centre a square crop on the
    actual face.

    Args:
        frame     : full BGR frame
        bbox      : (x1, y1, x2, y2) person bounding box
        keypoints : list of (x, y, conf) from YOLO pose (17 keypoints)

    Returns:
        BGR face crop (numpy array) or None if no valid crop possible.
    """
    h_frame, w_frame = frame.shape[:2]
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    # ── Strategy 1: Use facial keypoints for precise crop ──
    if keypoints and len(keypoints) >= 5:
        face_kps = []
        # COCO pose keypoints: 0=nose, 1=left_eye, 2=right_eye, 3=left_ear, 4=right_ear
        for idx in range(5):
            if idx < len(keypoints):
                kx, ky, kconf = keypoints[idx]
                kx, ky = float(kx), float(ky)
                if kconf > 0.3 and kx > 0 and ky > 0:
                    face_kps.append((kx, ky))

        if len(face_kps) >= 2:
            # Calculate face centre from visible keypoints
            xs = [p[0] for p in face_kps]
            ys = [p[1] for p in face_kps]
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)

            # Estimate face size from keypoint spread
            spread_x = max(xs) - min(xs) if len(xs) > 1 else 30
            spread_y = max(ys) - min(ys) if len(ys) > 1 else 30
            face_size = max(spread_x, spread_y, 30)

            # Create a generous square crop around the face
            # Margin = 1.8x face size to include forehead and chin
            margin = max(int(face_size * 1.8), 50)

            fc_x1 = max(0, int(cx - margin))
            fc_y1 = max(0, int(cy - margin * 0.8))  # less above (forehead)
            fc_x2 = min(w_frame, int(cx + margin))
            fc_y2 = min(h_frame, int(cy + margin * 1.2))  # more below (chin)

            crop = frame[fc_y1:fc_y2, fc_x1:fc_x2]
            ch, cw = crop.shape[:2]
            if cw >= MIN_FACE_SIZE and ch >= MIN_FACE_SIZE:
                return crop.copy()

    # ── Strategy 2: Use nose keypoint only (if available) ──
    if keypoints and len(keypoints) >= 1:
        nx, ny, nconf = keypoints[0]  # nose
        if nconf > 0.3 and float(nx) > 0 and float(ny) > 0:
            nx, ny = float(nx), float(ny)
            body_w = x2 - x1
            body_h = y2 - y1
            # Face is roughly 1/4 of body height
            face_radius = max(int(body_h * 0.18), 40)

            fc_x1 = max(0, int(nx - face_radius))
            fc_y1 = max(0, int(ny - face_radius))
            fc_x2 = min(w_frame, int(nx + face_radius))
            fc_y2 = min(h_frame, int(ny + face_radius * 1.3))

            crop = frame[fc_y1:fc_y2, fc_x1:fc_x2]
            ch, cw = crop.shape[:2]
            if cw >= MIN_FACE_SIZE and ch >= MIN_FACE_SIZE:
                return crop.copy()

    # ── Strategy 3: Fallback — upper portion of bbox ──
    body_h = y2 - y1
    body_w = x2 - x1
    # Take top 35% of bbox with slight expansion
    pad = int(body_w * 0.08)
    head_h = max(int(body_h * 0.35), 50)

    fc_x1 = max(0, x1 - pad)
    fc_y1 = max(0, y1 - pad)
    fc_x2 = min(w_frame, x2 + pad)
    fc_y2 = min(h_frame, y1 + head_h)

    crop = frame[fc_y1:fc_y2, fc_x1:fc_x2]
    ch, cw = crop.shape[:2]
    if cw >= MIN_FACE_SIZE and ch >= MIN_FACE_SIZE:
        return crop.copy()

    return None


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDING & MATCHING
# ══════════════════════════════════════════════════════════════════════════════

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten().astype(float)
    b = b.flatten().astype(float)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 2.0
    return 1.0 - (dot / norm)


def _get_embedding(bgr_image: np.ndarray) -> np.ndarray | None:
    """Extract a face embedding from a BGR image crop."""
    if not _HAS_DF or bgr_image is None or bgr_image.size == 0:
        return None

    # Try with face detection enabled first (best quality)
    try:
        results = DeepFace.represent(
            img_path=bgr_image,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend="opencv",
        )
        if results and len(results) > 0:
            return np.array(results[0]["embedding"], dtype=np.float64)
    except Exception:
        pass

    # Fallback: try with skip_detection (treat whole crop as face)
    # Only do this if the crop is small enough to likely be a face
    h, w = bgr_image.shape[:2]
    aspect = w / max(h, 1)
    if 0.5 < aspect < 2.0 and w < 400 and h < 400:
        try:
            results = DeepFace.represent(
                img_path=bgr_image,
                model_name=MODEL_NAME,
                enforce_detection=False,
                detector_backend="skip",
            )
            if results and len(results) > 0:
                return np.array(results[0]["embedding"], dtype=np.float64)
        except Exception:
            pass

    return None


def _match_against_templates(probe: np.ndarray) -> tuple[str | None, str | None, float]:
    """Compare probe embedding against ALL active templates.

    For employees with multiple templates, we use the MINIMUM distance
    (best match) across all their templates.

    Returns (employee_id, full_name, distance) or (None, None, best_dist).
    """
    if not _templates:
        return None, None, float("inf")

    # Group distances by employee
    emp_best: dict[str, tuple[float, str]] = {}  # emp_id -> (best_dist, full_name)

    for tmpl in _templates:
        eid = tmpl["employee_id"]
        dist = _cosine_distance(probe, tmpl["embedding"])

        if eid not in emp_best or dist < emp_best[eid][0]:
            emp_best[eid] = (dist, tmpl["full_name"])

    if not emp_best:
        return None, None, float("inf")

    # Find overall best match
    best_eid = min(emp_best, key=lambda k: emp_best[k][0])
    best_dist, best_name = emp_best[best_eid]

    if best_dist <= MATCH_DISTANCE_THRESHOLD:
        return best_eid, best_name, best_dist
    return None, None, best_dist


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def init():
    """Pre-load face templates from database. Call once at startup."""
    global _templates_loaded_at
    _templates_loaded_at = 0.0  # force reload
    _load_templates()

    emp_ids = set(t["employee_id"] for t in _templates)
    print(f"[{_ts()}][face_identity] Initialised: {len(emp_ids)} employees, "
          f"{len(_templates)} templates, model={MODEL_NAME}, "
          f"threshold={MATCH_DISTANCE_THRESHOLD}")


def reload_templates():
    """Force-reload templates from database (call after new enrollment)."""
    global _templates_loaded_at
    _templates_loaded_at = 0.0
    _load_templates()


def identify_worker(frame: np.ndarray, bbox: tuple, worker_id: int,
                    keypoints: list = None) -> dict:
    """Attempt to identify a worker by face recognition.

    Args:
        frame      : full BGR frame
        bbox       : (x1, y1, x2, y2) bounding box of the detected person
        worker_id  : stable tracker ID for this person
        keypoints  : list of (x, y, conf) from YOLO pose (17 keypoints)

    Returns:
        dict with keys:
            "employee_id"  : str or None
            "full_name"    : str or None
            "display_name" : str  (always set)
            "distance"     : float
            "from_cache"   : bool
    """
    now = time.time()

    # ── Check cache first ──
    cached = _identity_cache.get(worker_id)
    if cached:
        age = now - cached["identified_at"]
        since_last = now - cached["last_attempt"]

        # If we have a positive match and it's fresh, return cached
        if cached["employee_id"] is not None and age < CACHE_TTL:
            return {
                "employee_id": cached["employee_id"],
                "full_name": cached["full_name"],
                "display_name": cached["full_name"],
                "distance": cached["distance"],
                "from_cache": True,
            }

        # If no positive match, wait for RECHECK_INTERVAL before retrying
        if cached["employee_id"] is None and since_last < RECHECK_INTERVAL:
            return {
                "employee_id": None,
                "full_name": None,
                "display_name": f"Worker {worker_id}",
                "distance": cached["distance"],
                "from_cache": True,
            }

    # ── Reload templates if needed ──
    _load_templates()

    if not _templates or not _HAS_DF:
        return {
            "employee_id": None,
            "full_name": None,
            "display_name": f"Worker {worker_id}",
            "distance": -1.0,
            "from_cache": False,
        }

    # ── Extract face crop using keypoints ──
    face_crop = _extract_face_crop(frame, bbox, keypoints)

    emp_id = None
    full_name = None
    distance = -1.0

    if face_crop is not None:
        crop_h, crop_w = face_crop.shape[:2]
        probe = _get_embedding(face_crop)
        if probe is not None:
            emp_id, full_name, distance = _match_against_templates(probe)
            if emp_id:
                print(f"[{_ts()}][face_identity] Worker {worker_id} IDENTIFIED: "
                      f"'{full_name}' (emp_id={emp_id}, dist={distance:.4f}, "
                      f"crop={crop_w}x{crop_h})")
            else:
                print(f"[{_ts()}][face_identity] Worker {worker_id} not matched "
                      f"(best_dist={distance:.4f}, threshold={MATCH_DISTANCE_THRESHOLD}, "
                      f"crop={crop_w}x{crop_h})")

    # ── Build result ──
    display_name = full_name if full_name else f"Worker {worker_id}"

    # ── Update cache ──
    _identity_cache[worker_id] = {
        "employee_id": emp_id,
        "full_name": full_name,
        "distance": distance,
        "identified_at": now,
        "last_attempt": now,
    }

    return {
        "employee_id": emp_id,
        "full_name": full_name,
        "display_name": display_name,
        "distance": distance,
        "from_cache": False,
    }


def get_cached_identity(worker_id: int) -> dict:
    """Return cached identity without running recognition."""
    cached = _identity_cache.get(worker_id)
    if cached and cached["employee_id"] is not None:
        age = time.time() - cached["identified_at"]
        if age < CACHE_TTL:
            return {
                "employee_id": cached["employee_id"],
                "full_name": cached["full_name"],
                "display_name": cached["full_name"],
                "distance": cached["distance"],
                "from_cache": True,
            }
    return {
        "employee_id": None,
        "full_name": None,
        "display_name": f"Worker {worker_id}",
        "distance": -1.0,
        "from_cache": True,
    }


def clear_worker(worker_id: int):
    """Remove cached identity for a worker."""
    _identity_cache.pop(worker_id, None)


def clear_all():
    """Clear all cached identities."""
    _identity_cache.clear()


def get_employee_info(emp_id: str) -> dict | None:
    """Look up full employee record by employee_id."""
    try:
        import employee_db
        return employee_db.get_employee(emp_id)
    except Exception:
        return None
