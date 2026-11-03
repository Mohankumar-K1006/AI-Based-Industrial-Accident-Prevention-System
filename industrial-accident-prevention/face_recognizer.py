# -*- coding: utf-8 -*-
"""face_recognizer.py — Employee face recognition against enrolled encodings (deepface)."""

import csv
import json
import os
import pickle
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

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
    _real_print("[face_recognizer] WARNING: deepface not installed. Recognition disabled.")
except Exception as e:
    _HAS_DF = False
    _real_print(f"[face_recognizer] WARNING: failed to load deepface: {e}")
finally:
    builtins.print = _real_print


_ROOT = Path(__file__).resolve().parent
EMPLOYEES_FILE  = str(_ROOT / "employees.json")
ENCODINGS_FILE  = str(_ROOT / "face_encodings.pkl")
MATCH_LOG_FILE  = str(_ROOT / "face_match_log.csv")

# DeepFace model to use for embeddings — Facenet is a good accuracy/speed balance
MODEL_NAME = os.environ.get("FACE_MODEL", "Facenet")

# Cosine distance threshold (Facenet standard recommended is 0.55-0.60; lower = stricter)
MATCH_DISTANCE_THRESHOLD = float(os.environ.get("FACE_MATCH_THRESHOLD", "0.60"))

# Per-(employee_id, zone_id) cooldown for employee alert emails (2 minutes)
COOLDOWN_SECS = int(os.environ.get("EMPLOYEE_ALERT_COOLDOWN", "120"))

# Minimum crop dimensions (pixels) — crops smaller than this are unlikely to contain
# a recognizable face and would produce garbage embeddings.
MIN_CROP_WIDTH  = 40
MIN_CROP_HEIGHT = 40

_lock = threading.Lock()
_last_notified: dict[str, float] = {}  # "emp_id::zone_id" -> epoch (successful sends only)
_in_progress: set[str] = set()          # keys currently being sent (prevents duplicate sends)


def _ts() -> str:
    """Formatted timestamp for log messages."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


# ── data loading ──

def _load_employees() -> list[dict]:
    if not os.path.exists(EMPLOYEES_FILE):
        print(f"[{_ts()}][face_recognizer] employees.json not found at {EMPLOYEES_FILE}")
        return []
    try:
        with open(EMPLOYEES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[{_ts()}][face_recognizer] Loaded {len(data)} employee(s) from employees.json")
        return data
    except Exception as e:
        print(f"[{_ts()}][face_recognizer] Error reading {EMPLOYEES_FILE}: {e}")
        return []


def _load_encodings() -> dict[str, np.ndarray]:
    """Returns {employee_id: embedding_vector} dict.

    Tries SQLite database first, falls back to legacy pickle file.
    """
    # Try loading from SQLite database first
    try:
        import employee_db
        employee_db.init()
        templates = employee_db.get_all_active_templates()
        if templates:
            # Group by employee_id, keep the one with most templates
            result = {}
            for tmpl in templates:
                eid = tmpl["employee_id"]
                # Store each template — identify_face will match against all
                if eid not in result:
                    result[eid] = []
                result[eid].append(tmpl["embedding"])
            print(f"[{_ts()}][face_recognizer] Loaded {len(templates)} template(s) "
                  f"for {len(result)} employee(s) from database")
            return result
    except Exception as e:
        print(f"[{_ts()}][face_recognizer] Database load failed, using pickle: {e}")

    # Fallback to legacy pickle
    if not os.path.exists(ENCODINGS_FILE):
        print(f"[{_ts()}][face_recognizer] face_encodings.pkl not found at {ENCODINGS_FILE}")
        return {}
    try:
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)
        valid = {}
        for emp_id, enc in data.items():
            arr = np.asarray(enc)
            if arr.ndim >= 1 and arr.size > 0 and np.isfinite(arr).all():
                valid[emp_id] = [arr]  # wrap in list for consistency
            else:
                print(f"[{_ts()}][face_recognizer] WARNING: invalid encoding for '{emp_id}'")
        print(f"[{_ts()}][face_recognizer] Loaded {len(valid)} valid embedding(s) from face_encodings.pkl")
        return valid
    except Exception as e:
        print(f"[{_ts()}][face_recognizer] Error reading {ENCODINGS_FILE}: {e}")
        return {}


def _get_employee_by_id(emp_id: str) -> dict | None:
    try:
        import employee_db
        emp = employee_db.get_employee(emp_id)
        if emp:
            return emp
    except Exception:
        pass
    # Fallback to JSON
    for emp in _load_employees():
        if emp.get("employee_id") == emp_id:
            return emp
    return None


def _valid_email(email: str) -> bool:
    """Basic email format validation."""
    return bool(re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", email))


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


def _is_valid_crop(crop: np.ndarray) -> bool:
    """Check if a crop is large enough to potentially contain a face."""
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    return w >= MIN_CROP_WIDTH and h >= MIN_CROP_HEIGHT


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
        print(f"[{_ts()}][face_recognizer] Embedding error: {e}")

    return None


# ── core recognition ──

def identify_face(bgr_crop: np.ndarray) -> tuple[str | None, float]:
    """Match a BGR face crop against enrolled encodings.

    Matches against ALL templates per employee and takes the MINIMUM
    distance (best match).

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

    for emp_id, templates in encodings_db.items():
        # templates is a list of embedding vectors
        if not isinstance(templates, list):
            templates = [templates]
        for stored_enc in templates:
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
    print(f"[{_ts()}][face_recognizer] === DANGER-ZONE ENTRY DETECTED === zone={zone_id}")
    print(f"[{_ts()}][face_recognizer] Starting face-recognition task for bbox={bbox}")

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * 0.1)
    pad_y = int((y2 - y1) * 0.1)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    body_crop = frame[cy1:cy2, cx1:cx2].copy()

    # Head crop (top 45% of body box) for reliable face recognition
    bh = cy2 - cy1
    head_cy2 = min(cy2, cy1 + max(int(bh * 0.45), 60))
    head_crop = frame[cy1:head_cy2, cx1:cx2].copy()

    def _worker():
        try:
            _do_recognize_and_notify(head_crop, body_crop, zone_id, img_path)
        except Exception as e:
            print(f"[{_ts()}][face_recognizer] FATAL error in recognition worker: {e}")
            traceback.print_exc()

    threading.Thread(target=_worker, daemon=True).start()


def _do_recognize_and_notify(head_crop: np.ndarray, body_crop: np.ndarray,
                              zone_id: str, img_path: str):
    """Core recognition + notification logic, runs inside the worker thread."""

    # ── Step 1: Try head crop ──
    head_h, head_w = head_crop.shape[:2] if head_crop is not None and head_crop.size > 0 else (0, 0)
    print(f"[{_ts()}][face_recognizer] Head crop dimensions: {head_w}x{head_h}px")

    emp_id = None
    distance = -1.0

    if _is_valid_crop(head_crop):
        print(f"[{_ts()}][face_recognizer] Attempting face recognition on head crop...")
        emp_id, distance = identify_face(head_crop)
        if emp_id is not None:
            print(f"[{_ts()}][face_recognizer] Head crop MATCHED: emp_id={emp_id}, distance={distance:.4f}, "
                  f"threshold={MATCH_DISTANCE_THRESHOLD}")
        else:
            reason = "no face/embedding detected" if distance < 0 else f"best distance {distance:.4f} > threshold {MATCH_DISTANCE_THRESHOLD}"
            print(f"[{_ts()}][face_recognizer] Head crop did NOT match ({reason})")
    else:
        print(f"[{_ts()}][face_recognizer] Head crop too small ({head_w}x{head_h}), skipping")

    # ── Step 2: Fallback to body crop if head crop did NOT produce a valid match ──
    # BUG FIX: Previously the condition was `distance > MATCH_DISTANCE_THRESHOLD`, which
    # is False when distance is -1.0 (no face detected), so the fallback never ran.
    # Now we fall back whenever emp_id is None (any failed recognition, including -1.0).
    if emp_id is None:
        body_h, body_w = body_crop.shape[:2] if body_crop is not None and body_crop.size > 0 else (0, 0)
        print(f"[{_ts()}][face_recognizer] Fallback: attempting body crop ({body_w}x{body_h}px)...")

        if _is_valid_crop(body_crop):
            body_id, body_dist = identify_face(body_crop)
            if body_id is not None:
                emp_id = body_id
                distance = body_dist
                print(f"[{_ts()}][face_recognizer] Body crop MATCHED: emp_id={body_id}, "
                      f"distance={body_dist:.4f}, threshold={MATCH_DISTANCE_THRESHOLD}")
            else:
                reason = "no face/embedding detected" if body_dist < 0 else f"best distance {body_dist:.4f} > threshold {MATCH_DISTANCE_THRESHOLD}"
                print(f"[{_ts()}][face_recognizer] Body crop did NOT match ({reason})")
                # Keep the better distance for logging (prefer a positive distance over -1.0)
                if body_dist >= 0 and (distance < 0 or body_dist < distance):
                    distance = body_dist
        else:
            print(f"[{_ts()}][face_recognizer] Body crop too small ({body_w}x{body_h}), skipping")

    # ── Step 3: Determine final match status ──
    matched = emp_id is not None
    log_id = emp_id if matched else "UNKNOWN"

    print(f"[{_ts()}][face_recognizer] --- RECOGNITION RESULT ---")
    print(f"[{_ts()}][face_recognizer]   Final employee_id : {log_id}")
    print(f"[{_ts()}][face_recognizer]   Final distance    : {distance:.4f}")
    print(f"[{_ts()}][face_recognizer]   Threshold         : {MATCH_DISTANCE_THRESHOLD}")
    print(f"[{_ts()}][face_recognizer]   Match status      : {'MATCHED' if matched else 'UNKNOWN'}")

    _log_match(log_id, distance, zone_id, matched)

    if not matched:
        # Manager notification already sent by alert_system
        print(f"[{_ts()}][face_recognizer] UNKNOWN person in zone {zone_id} — manager already notified")
        return

    # ── Step 4: Look up employee record and validate email ──
    emp = _get_employee_by_id(emp_id)
    if not emp:
        print(f"[{_ts()}][face_recognizer] WARNING: Matched emp_id={emp_id} but no record in employees.json")
        return

    recipient_email = emp.get("personal_email", "").strip()
    print(f"[{_ts()}][face_recognizer] Matched employee: id={emp_id}, name={emp.get('full_name', 'N/A')}, "
          f"email={'FOUND' if recipient_email else 'MISSING'}")

    if not recipient_email:
        print(f"[{_ts()}][face_recognizer] WARNING: Employee {emp_id} has no personal_email — cannot send alert")
        return

    if not _valid_email(recipient_email):
        print(f"[{_ts()}][face_recognizer] WARNING: Employee {emp_id} has invalid email format: {recipient_email}")
        return

    # ── Step 5: Thread-safe cooldown check ──
    key = f"{emp_id}::{zone_id}"
    now = time.time()
    with _lock:
        last = _last_notified.get(key, 0.0)
        if now - last < COOLDOWN_SECS:
            remaining = int(COOLDOWN_SECS - (now - last))
            print(f"[{_ts()}][face_recognizer] COOLDOWN active for {emp_id} in {zone_id} "
                  f"({remaining}s remaining) — email SKIPPED")
            return

        # Check if a send is already in-progress for this key (prevent duplicate sends)
        if key in _in_progress:
            print(f"[{_ts()}][face_recognizer] Email already in-progress for {emp_id} in {zone_id} — SKIPPED")
            return

        # Mark as in-progress (NOT as successfully notified yet)
        _in_progress.add(key)

    # ── Step 6: Send email and record outcome ──
    print(f"[{_ts()}][face_recognizer] Sending employee alert email to {recipient_email}...")
    try:
        sent = _send_employee_alert_email(emp, zone_id, img_path)
    except Exception as e:
        sent = False
        print(f"[{_ts()}][face_recognizer] Email sending raised exception: {e}")
        traceback.print_exc()

    with _lock:
        _in_progress.discard(key)
        if sent:
            # BUG FIX: Only record the cooldown timestamp AFTER a successful send.
            _last_notified[key] = time.time()
            print(f"[{_ts()}][face_recognizer] Email status: SENT — cooldown started ({COOLDOWN_SECS}s)")
        else:
            # Do NOT update _last_notified — next attempt can retry immediately.
            print(f"[{_ts()}][face_recognizer] Email status: FAILED — cooldown NOT set (retry allowed)")


# ── employee email ──

def _send_employee_alert_email(emp: dict, zone_id: str, img_path: str) -> bool:
    """Send emergency alert to the employee's personal email. Reuses SMTP logic from notifier.

    Returns True if the email was sent successfully, False otherwise.
    """
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

    print(f"[{_ts()}][face_recognizer] SMTP config: host={host}, port={port}, "
          f"user={'SET' if user else 'MISSING'}, password={'SET' if pwd else 'MISSING'}")

    if not user or not pwd:
        print(f"[{_ts()}][face_recognizer] SMTP credentials not set — email FAILED")
        _log_notification(False, "employee_email", zone_id, "SMTP_USER or SMTP_PASSWORD not set")
        return False

    manager = get_manager(zone_id)
    machine = manager.get("machine_name", zone_id) if manager else zone_id
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    to_addr = emp.get("personal_email", "").strip()
    if not to_addr:
        print(f"[{_ts()}][face_recognizer] No recipient email for employee {emp.get('employee_id')} — email FAILED")
        _log_notification(False, "employee_email", zone_id, "no personal_email in employee record")
        return False

    print(f"[{_ts()}][face_recognizer] Composing email to {to_addr} for zone={zone_id}")

    subject = f"Emergency Alert - You were detected in restricted zone {zone_id}"

    msg = MIMEMultipart("related")
    msg["From"]    = user
    msg["To"]      = to_addr
    msg["Subject"] = subject

    alt = MIMEMultipart("alternative")
    msg.attach(alt)

    text_body = (
        f"EMERGENCY ALERT\n\n"
        f"Name        : {emp.get('full_name', 'N/A')}\n"
        f"Employee ID : {emp.get('employee_id', 'N/A')}\n"
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
          <p><strong>Name:</strong> {emp.get('full_name', 'N/A')}</p>
          <p><strong>Employee ID:</strong> {emp.get('employee_id', 'N/A')}</p>
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
            print(f"[{_ts()}][face_recognizer] Attached incident image: {img_path}")
        except Exception as e:
            print(f"[{_ts()}][face_recognizer] WARNING: Failed to attach image: {e}")
    else:
        print(f"[{_ts()}][face_recognizer] No incident image to attach (path={img_path})")

    sent = False
    last_err = None
    attempts = [(host, port)]
    alt_port = 587 if port == 465 else 465
    attempts.append((host, alt_port))

    for h, p in attempts:
        try:
            print(f"[{_ts()}][face_recognizer] SMTP connecting to {h}:{p}...")
            if p == 465:
                with smtplib.SMTP_SSL(h, p, timeout=25) as srv:
                    print(f"[{_ts()}][face_recognizer] SMTP_SSL connected, authenticating...")
                    srv.login(user, pwd)
                    print(f"[{_ts()}][face_recognizer] SMTP authenticated, sending to {to_addr}...")
                    refused = srv.sendmail(user, [to_addr], msg.as_string())
                    if refused:
                        print(f"[{_ts()}][face_recognizer] WARNING: Some recipients refused: {refused}")
                        last_err = Exception(f"Refused recipients: {refused}")
                        continue
            else:
                with smtplib.SMTP(h, p, timeout=20) as srv:
                    print(f"[{_ts()}][face_recognizer] SMTP connected, starting TLS...")
                    srv.starttls()
                    print(f"[{_ts()}][face_recognizer] TLS started, authenticating...")
                    srv.login(user, pwd)
                    print(f"[{_ts()}][face_recognizer] SMTP authenticated, sending to {to_addr}...")
                    refused = srv.sendmail(user, [to_addr], msg.as_string())
                    if refused:
                        print(f"[{_ts()}][face_recognizer] WARNING: Some recipients refused: {refused}")
                        last_err = Exception(f"Refused recipients: {refused}")
                        continue
            sent = True
            print(f"[{_ts()}][face_recognizer] SMTP sendmail completed on {h}:{p}")
            break
        except smtplib.SMTPAuthenticationError as exc:
            last_err = exc
            print(f"[{_ts()}][face_recognizer] SMTP AUTH FAILED on {h}:{p}: {exc}")
        except smtplib.SMTPConnectError as exc:
            last_err = exc
            print(f"[{_ts()}][face_recognizer] SMTP CONNECT FAILED on {h}:{p}: {exc}")
        except smtplib.SMTPRecipientsRefused as exc:
            last_err = exc
            print(f"[{_ts()}][face_recognizer] SMTP RECIPIENTS REFUSED on {h}:{p}: {exc}")
        except Exception as exc:
            last_err = exc
            print(f"[{_ts()}][face_recognizer] SMTP error on {h}:{p}: {type(exc).__name__}: {exc}")

    if sent:
        _log_notification(True, "employee_email", zone_id, f"sent to {to_addr} ({emp.get('employee_id', 'N/A')})")
        print(f"[{_ts()}][face_recognizer] === EMAIL SENT to {to_addr} ===")
        return True
    else:
        _log_notification(False, "employee_email", zone_id, str(last_err))
        print(f"[{_ts()}][face_recognizer] === EMAIL FAILED to {to_addr}: {last_err} ===")
        return False


# ── standalone test email function ──

def test_email(recipient: str = "") -> bool:
    """Send a test email to verify SMTP configuration works, without requiring a YOLO event.

    Args:
        recipient: Email address to send the test to. If empty, reads from
                   SMTP_TEST_TO env var, then falls back to the first enrolled employee.

    Returns:
        True if the test email was sent successfully, False otherwise.
    """
    import smtplib
    from email.mime.text import MIMEText
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env", override=True)
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    user = os.environ.get("SMTP_USER", "")
    pwd  = os.environ.get("SMTP_PASSWORD", "")

    print(f"[{_ts()}][test_email] SMTP config: host={host}, port={port}, "
          f"user={'SET' if user else 'MISSING'}, password={'SET' if pwd else 'MISSING'}")

    if not user or not pwd:
        print(f"[{_ts()}][test_email] SMTP credentials not configured — test FAILED")
        return False

    # Determine recipient
    if not recipient:
        recipient = os.environ.get("SMTP_TEST_TO", "").strip()
    if not recipient:
        employees = _load_employees()
        if employees:
            recipient = employees[0].get("personal_email", "").strip()
    if not recipient:
        print(f"[{_ts()}][test_email] No recipient specified and no enrolled employees — test FAILED")
        return False

    print(f"[{_ts()}][test_email] Sending test email to {recipient}...")

    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = MIMEText(
        f"This is a test email from the AI-Based Industrial Accident Prevention System.\n\n"
        f"Sent at: {ts_str}\n"
        f"SMTP Host: {host}:{port}\n"
        f"From: {user}\n\n"
        f"If you received this, the SMTP configuration is working correctly."
    )
    msg["Subject"] = f"[TEST] Industrial Safety System - SMTP Test ({ts_str})"
    msg["From"] = user
    msg["To"] = recipient

    attempts = [(host, port)]
    alt_port = 587 if port == 465 else 465
    attempts.append((host, alt_port))

    for h, p in attempts:
        try:
            print(f"[{_ts()}][test_email] Connecting to {h}:{p}...")
            if p == 465:
                with smtplib.SMTP_SSL(h, p, timeout=25) as srv:
                    srv.login(user, pwd)
                    srv.sendmail(user, [recipient], msg.as_string())
            else:
                with smtplib.SMTP(h, p, timeout=20) as srv:
                    srv.starttls()
                    srv.login(user, pwd)
                    srv.sendmail(user, [recipient], msg.as_string())
            print(f"[{_ts()}][test_email] === TEST EMAIL SENT to {recipient} via {h}:{p} ===")
            return True
        except Exception as exc:
            print(f"[{_ts()}][test_email] Failed on {h}:{p}: {type(exc).__name__}: {exc}")

    print(f"[{_ts()}][test_email] === TEST EMAIL FAILED — all attempts exhausted ===")
    return False
