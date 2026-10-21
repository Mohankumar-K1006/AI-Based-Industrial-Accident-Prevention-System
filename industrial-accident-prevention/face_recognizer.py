# -*- coding: utf-8 -*-
"""face_recognizer.py — Employee face recognition against enrolled encodings (deepface)."""

import csv
import json
import os
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from deepface import DeepFace
    _HAS_DF = True
except ImportError:
    _HAS_DF = False
    print("[face_recognizer] WARNING: deepface not installed. Recognition disabled.")

_ROOT = Path(__file__).resolve().parent
EMPLOYEES_FILE  = str(_ROOT / "employees.json")
ENCODINGS_FILE  = str(_ROOT / "face_encodings.pkl")
MATCH_LOG_FILE  = str(_ROOT / "face_match_log.csv")

# DeepFace model to use for embeddings — Facenet is a good accuracy/speed balance
MODEL_NAME = os.environ.get("FACE_MODEL", "Facenet")

# Cosine distance threshold (Facenet standard recommended is 0.55-0.60; lower = stricter)
MATCH_DISTANCE_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.60"))

# Per-(employee_id, zone_id) cooldown reusing the same cooldown value as notifier
COOLDOWN_SECS = int(os.environ.get("NOTIFIER_COOLDOWN", "600"))

_lock = threading.Lock()
_last_notified: dict[str, float] = {}  # "emp_id::zone_id" -> epoch


# ── data loading ──

def _load_employees() -> list[dict]:
    if not os.path.exists(EMPLOYEES_FILE):
        return []
    try:
        with open(EMPLOYEES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[face_recognizer] Error reading {EMPLOYEES_FILE}: {e}")
        return []


def _load_encodings() -> dict[str, np.ndarray]:
    """Returns {employee_id: embedding_vector} dict."""
    if not os.path.exists(ENCODINGS_FILE):
        return {}
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[face_recognizer] Error reading {ENCODINGS_FILE}: {e}")
        return {}


def _get_employee_by_id(emp_id: str) -> dict | None:
    for emp in _load_employees():
        if emp.get("employee_id") == emp_id:
            return emp
    return None


# ── match logging ──

def _log_match(employee_id: str, distance: float, zone_id: str, matched: bool):
    os.makedirs(os.path.dirname(MATCH_LOG_FILE) or ".", exist_ok=True)
    write_header = not os.path.exists(MATCH_LOG_FILE)
    with open(MATCH_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "employee_id", "distance", "zone_id", "matched"])
        w.writerow([
            datetime.now().isoformat(),
            employee_id,
            f"{distance:.4f}",
            zone_id,
            "YES" if matched else "NO",
        ])
        f.flush()


# ── distance helpers ──

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite."""
    a = a.flatten().astype(float)
    b = b.flatten().astype(float)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 2.0
    return 1.0 - (dot / norm)


def get_embedding(bgr_image: np.ndarray) -> np.ndarray | None:
    """Extract face embedding from a BGR image using DeepFace.

    Returns the embedding vector or None if no face detected.
    """
    if not _HAS_DF or bgr_image is None or bgr_image.size == 0:
        return None

    # Step 1: try standard face detection
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

    # Step 2: if strict detection failed (e.g. tilted face / partial angle), extract from crop directly
    try:
        results = DeepFace.represent(
            img_path=bgr_image,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend="opencv",
        )
        if results and len(results) > 0:
            return np.array(results[0]["embedding"], dtype=np.float64)
    except Exception as e:
        print(f"[face_recognizer] Embedding error: {e}")

    return None


# ── core recognition ──

def identify_face(bgr_crop: np.ndarray) -> tuple[str | None, float]:
    """Match a BGR face crop against enrolled encodings.

    Returns (employee_id, distance) if matched, or (None, best_distance).
    If no encodings enrolled or no face detected in crop, returns (None, -1.0).
    """
    if not _HAS_DF:
        return None, -1.0

    encodings_db = _load_encodings()
    if not encodings_db:
        return None, -1.0

    probe = get_embedding(bgr_crop)
    if probe is None:
        return None, -1.0

    best_id = None
    best_dist = float("inf")

    for emp_id, stored_enc in encodings_db.items():
        dist = _cosine_distance(probe, stored_enc)
        if dist < best_dist:
            best_dist = dist
            best_id = emp_id

    if best_dist <= MATCH_DISTANCE_THRESHOLD:
        return best_id, best_dist
    return None, best_dist


# ── background recognition + notification (called from alert_system) ──

def recognize_and_notify(frame: np.ndarray, bbox: tuple, zone_id: str, img_path: str):
    """Run face recognition in a background thread and send employee email if matched.

    Args:
        frame   : full BGR frame at the moment of DANGER entry
        bbox    : (x1, y1, x2, y2) bounding box of the detected person
        zone_id : which danger zone was breached
        img_path: path to the saved incident snapshot (for email attachment)
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.1)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    body_crop = frame[cy1:cy2, cx1:cx2].copy()

    # Head crop (top 40% of body box) for reliable face recognition
    bh = cy2 - cy1
    head_cy2 = min(cy2, cy1 + max(int(bh * 0.45), 60))
    head_crop = frame[cy1:head_cy2, cx1:cx2].copy()

    def _worker():
        # Try head crop first (cleanest face), then full body crop
        emp_id, distance = identify_face(head_crop)
        if emp_id is None and distance > MATCH_DISTANCE_THRESHOLD:
            body_id, body_dist = identify_face(body_crop)
            if body_id is not None or (body_dist >= 0 and (distance < 0 or body_dist < distance)):
                emp_id = body_id
                distance = body_dist

        matched = emp_id is not None

        _log_match(emp_id or "UNIDENTIFIED", distance, zone_id, matched)

        if not matched:
            # Manager notification already sent by alert_system
            print(f"[face_recognizer] UNIDENTIFIED person in zone {zone_id} — manager already notified")
            return

        # Cooldown check per (employee, zone)
        key = f"{emp_id}::{zone_id}"
        now = time.time()
        with _lock:
            last = _last_notified.get(key, 0.0)
            if now - last < COOLDOWN_SECS:
                print(f"[face_recognizer] Cooldown active for {emp_id} in {zone_id}, skipping email")
                return
            _last_notified[key] = now

        emp = _get_employee_by_id(emp_id)
        if not emp:
            return

        _send_employee_alert_email(emp, zone_id, img_path)

    threading.Thread(target=_worker, daemon=True).start()


# ── employee email ──

def _send_employee_alert_email(emp: dict, zone_id: str, img_path: str):
    """Send emergency alert to the employee's personal email. Reuses SMTP logic from notifier."""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders as enc
    from dotenv import load_dotenv
    from notifier import _log_notification, get_manager

    load_dotenv(_ROOT / ".env", override=True)
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    user = os.environ.get("SMTP_USER", "")
    pwd  = os.environ.get("SMTP_PASSWORD", "")
    if not user or not pwd:
        print("[face_recognizer] SMTP credentials not set, skipping employee email")
        return

    manager = get_manager(zone_id)
    machine = manager.get("machine_name", zone_id) if manager else zone_id
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    to_addr = emp["personal_email"]
    subject = f"Emergency Alert - You were detected in restricted zone {zone_id}"

    msg = MIMEMultipart("related")
    msg["From"]    = user
    msg["To"]      = to_addr
    msg["Subject"] = subject

    alt = MIMEMultipart("alternative")
    msg.attach(alt)

    text_body = (
        f"EMERGENCY ALERT\n\n"
        f"Name        : {emp['full_name']}\n"
        f"Employee ID : {emp['employee_id']}\n"
        f"Timestamp   : {ts}\n"
        f"Zone        : {zone_id}\n"
        f"Machine     : {machine}\n\n"
        f"You have been detected inside a restricted danger zone.\n"
        f"Please leave the area immediately and report to your supervisor."
    )
    alt.attach(MIMEText(text_body, "plain"))

    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #222;">
        <div style="background-color: #d9534f; color: white; padding: 12px 16px; border-radius: 4px; font-size: 18px; font-weight: bold;">
          EMERGENCY - You were detected in a restricted zone
        </div>
        <div style="padding: 16px; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; background-color: #fafafa;">
          <p><strong>Name:</strong> {emp['full_name']}</p>
          <p><strong>Employee ID:</strong> {emp['employee_id']}</p>
          <p><strong>Timestamp:</strong> {ts}</p>
          <p><strong>Zone:</strong> {zone_id}</p>
          <p><strong>Machine:</strong> {machine}</p>
          <p style="color: #d9534f; font-size: 15px; font-weight: bold;">
            Please leave the restricted area immediately!
          </p>
          <div style="margin-top: 15px;">
            <h4>Incident Camera Snapshot:</h4>
            <img src="cid:danger_snapshot" style="max-width: 100%; border: 2px solid #d9534f; border-radius: 4px;" alt="Incident Snapshot" />
          </div>
        </div>
      </body>
    </html>
    """
    alt.attach(MIMEText(html_body, "html"))

    # Attach incident image
    if img_path and os.path.exists(img_path):
        try:
            with open(img_path, "rb") as fp:
                part = MIMEBase("image", "jpeg")
                part.set_payload(fp.read())
                enc.encode_base64(part)
                part.add_header("Content-Disposition", "attachment",
                                filename=os.path.basename(img_path))
                part.add_header("Content-ID", "<danger_snapshot>")
                msg.attach(part)
        except Exception as e:
            print(f"[face_recognizer] Warning: Failed to attach image: {e}")

    sent = False
    last_err = None
    attempts = [(host, port)]
    alt_port = 587 if port == 465 else 465
    attempts.append((host, alt_port))

    for h, p in attempts:
        try:
            if p == 465:
                with smtplib.SMTP_SSL(h, p, timeout=25) as srv:
                    srv.login(user, pwd)
                    srv.sendmail(user, [to_addr], msg.as_string())
            else:
                with smtplib.SMTP(h, p, timeout=20) as srv:
                    srv.starttls()
                    srv.login(user, pwd)
                    srv.sendmail(user, [to_addr], msg.as_string())
            sent = True
            break
        except Exception as exc:
            last_err = exc

    if sent:
        _log_notification(True, "employee_email", zone_id, f"sent to {to_addr} ({emp['employee_id']})")
        print(f"[face_recognizer] Employee email sent to {to_addr}")
    else:
        _log_notification(False, "employee_email", zone_id, str(last_err))
        print(f"[face_recognizer] Employee email FAILED to {to_addr}: {last_err}")
