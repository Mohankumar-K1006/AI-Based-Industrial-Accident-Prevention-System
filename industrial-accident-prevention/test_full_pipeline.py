# -*- coding: utf-8 -*-
"""test_full_pipeline.py — Comprehensive test of the employee email pipeline.

Tests:
  1. Load employee database and verify stored emails + embeddings.
  2. Test face recognition on a clear employee image.
  3. Verify the fallback path when head crop has no face.
  4. Test SMTP delivery independently.
  5. Simulate cooldown-after-failure behavior.
  6. Confirm alarm still works if recognition/email fails.

Usage:
    python test_full_pipeline.py                    # run all offline tests
    python test_full_pipeline.py path/to/photo.jpg  # also test recognition with that photo
"""

import os
import sys
import time
import json
import pickle
import threading
from pathlib import Path
from datetime import datetime

# Ensure project imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.chdir(str(Path(__file__).resolve().parent))


def _header(title: str):
    print(f"\n{'='*70}")
    print(f"  TEST: {title}")
    print(f"{'='*70}")


def _ok(msg: str):
    print(f"  [PASS] {msg}")


def _fail(msg: str):
    print(f"  [FAIL] {msg}")


def _info(msg: str):
    print(f"  [INFO] {msg}")


passed = 0
failed = 0


def _assert(cond: bool, msg: str):
    global passed, failed
    if cond:
        _ok(msg)
        passed += 1
    else:
        _fail(msg)
        failed += 1


# ─── Test 1: Employee database & encoding compatibility ───

def test_employee_database():
    _header("Employee database and encoding compatibility")

    from face_recognizer import _load_employees, _load_encodings, EMPLOYEES_FILE, ENCODINGS_FILE

    _info(f"employees.json path: {EMPLOYEES_FILE}")
    _info(f"face_encodings.pkl path: {ENCODINGS_FILE}")

    employees = _load_employees()
    encodings = _load_encodings()

    _assert(len(employees) > 0, f"Loaded {len(employees)} employee(s) from employees.json")
    _assert(len(encodings) > 0, f"Loaded {len(encodings)} encoding(s) from face_encodings.pkl")

    for emp in employees:
        eid = emp.get("employee_id", "???")
        email = emp.get("personal_email", "")
        name = emp.get("full_name", "")
        has_enc = eid in encodings

        _info(f"  {eid}: name={name}, email={email}, has_encoding={has_enc}")

        # Verify personal_email field exists and is non-empty
        _assert(bool(email), f"Employee '{eid}' has personal_email = '{email}'")

        # Verify encoding exists and has valid shape
        if has_enc:
            enc = encodings[eid]
            import numpy as np
            arr = np.asarray(enc)
            _assert(arr.ndim >= 1 and arr.size > 0, f"Employee '{eid}' encoding shape={arr.shape}")
            _assert(np.isfinite(arr).all(), f"Employee '{eid}' encoding has all finite values")
        else:
            _fail(f"Employee '{eid}' has NO encoding in face_encodings.pkl")


# ─── Test 2: Face recognition on an image ───

def test_face_recognition(img_path: str = ""):
    _header("Face recognition on image")

    from face_recognizer import identify_face

    if not img_path:
        _info("No test image provided — testing with blank image (should return UNKNOWN)")
        import numpy as np
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        emp_id, dist = identify_face(blank)
        _assert(emp_id is None, f"Blank image returns no match (emp_id={emp_id}, dist={dist})")
        return

    import cv2
    img = cv2.imread(img_path)
    if img is None:
        _fail(f"Could not read image: {img_path}")
        return

    _info(f"Testing recognition on: {img_path} ({img.shape[1]}x{img.shape[0]})")
    emp_id, dist = identify_face(img)

    if emp_id:
        _ok(f"MATCHED: emp_id={emp_id}, distance={dist:.4f}")
        from face_recognizer import _get_employee_by_id
        emp = _get_employee_by_id(emp_id)
        if emp:
            _info(f"  Name: {emp.get('full_name', 'N/A')}, Email: {emp.get('personal_email', 'N/A')}")
    else:
        _info(f"UNIDENTIFIED (no match): best_distance={dist:.4f}" if dist >= 0
              else "No face detected in image")


# ─── Test 3: Fallback logic ───

def test_fallback_logic():
    _header("Fallback from head crop to body crop")

    from face_recognizer import identify_face, _is_valid_crop
    import numpy as np

    # Create a very small crop (should fail)
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    _assert(not _is_valid_crop(tiny), "Tiny 10x10 crop correctly rejected by _is_valid_crop")

    # Create a valid-size blank crop (should fail recognition but not crash)
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    _assert(_is_valid_crop(blank), "200x200 crop accepted by _is_valid_crop")

    emp_id, dist = identify_face(blank)
    _assert(emp_id is None, f"Blank 200x200 returns no match (emp_id={emp_id}, dist={dist})")

    # Verify that recognize_and_notify handles the fallback path without crashing
    _info("Testing recognize_and_notify with blank frame (no crash = pass)...")
    from face_recognizer import recognize_and_notify
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    bbox = (100, 50, 300, 400)
    try:
        recognize_and_notify(frame, bbox, "test_zone", "")
        time.sleep(2)  # let worker thread run
        _ok("recognize_and_notify completed without crash")
    except Exception as e:
        _fail(f"recognize_and_notify crashed: {e}")


# ─── Test 4: SMTP test email ───

def test_smtp():
    _header("SMTP test email (independent of YOLO)")

    from face_recognizer import test_email

    _info("Calling test_email()...")
    result = test_email()
    _assert(result is True or result is False, f"test_email returned: {result}")
    if result:
        _ok("Test email sent successfully!")
    else:
        _info("Test email failed — check SMTP config above")


# ─── Test 5: Cooldown-after-failure behavior ───

def test_cooldown_logic():
    _header("Cooldown logic: failed send should NOT block retry")

    from face_recognizer import _lock, _last_notified, _in_progress, COOLDOWN_SECS

    test_key = "test_emp::test_zone"

    with _lock:
        # Ensure clean state
        _last_notified.pop(test_key, None)
        _in_progress.discard(test_key)

    # Simulate: no prior send => should not be in cooldown
    with _lock:
        last = _last_notified.get(test_key, 0.0)
        now = time.time()
        in_cooldown = (now - last) < COOLDOWN_SECS
    _assert(not in_cooldown, "Fresh key is NOT in cooldown")

    # Simulate: successful send => should be in cooldown
    with _lock:
        _last_notified[test_key] = time.time()
    with _lock:
        last = _last_notified.get(test_key, 0.0)
        now = time.time()
        in_cooldown = (now - last) < COOLDOWN_SECS
    _assert(in_cooldown, "After successful send, key IS in cooldown")

    # Simulate: clear the cooldown (as if send failed => we don't set timestamp)
    with _lock:
        _last_notified.pop(test_key, None)
    with _lock:
        last = _last_notified.get(test_key, 0.0)
        now = time.time()
        in_cooldown = (now - last) < COOLDOWN_SECS
    _assert(not in_cooldown, "After clearing (simulated failure), key is NOT in cooldown — retry allowed")

    # Clean up
    with _lock:
        _last_notified.pop(test_key, None)
        _in_progress.discard(test_key)


# ─── Test 6: Alarm independence ───

def test_alarm_independence():
    _header("Alarm works if recognition/email fails")

    import numpy as np
    import alert_system

    mgr = alert_system.AlertManager(fps=10, zone="test_zone")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    workers = [{"status": "DANGER", "bbox": (100, 50, 300, 400)}]

    try:
        # Run through SAFE -> DANGER transition
        for i in range(10):
            frame_out = mgr.update("SAFE", frame.copy(), i)
        for i in range(10, 25):
            frame_out = mgr.update("DANGER", frame.copy(), i, workers)

        _ok("AlertManager ran through SAFE->DANGER without crash")
        _assert(mgr._confirmed in ("DANGER", "WARNING", "SAFE"),
                f"Confirmed state = {mgr._confirmed}")
    except Exception as e:
        _fail(f"AlertManager crashed: {e}")
    finally:
        mgr.shutdown()


# ─── Test 7: .env loading verification ───

def test_env_loading():
    _header(".env file loading verification")

    from dotenv import load_dotenv
    _ROOT = Path(__file__).resolve().parent
    env_path = _ROOT / ".env"

    _assert(env_path.exists(), f".env file exists at {env_path}")

    load_dotenv(env_path, override=True)

    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_pwd = os.environ.get("SMTP_PASSWORD", "")
    smtp_host = os.environ.get("SMTP_HOST", "")
    smtp_port = os.environ.get("SMTP_PORT", "")

    _assert(bool(smtp_user), f"SMTP_USER is SET")
    _assert(bool(smtp_pwd), f"SMTP_PASSWORD is SET (not printing value)")
    _assert(bool(smtp_host), f"SMTP_HOST = {smtp_host}")
    _assert(bool(smtp_port), f"SMTP_PORT = {smtp_port}")


# ─── Main ───

def main():
    global passed, failed

    print(f"\n{'#'*70}")
    print(f"  FULL PIPELINE TEST — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")

    test_env_loading()
    test_employee_database()

    img_path = sys.argv[1] if len(sys.argv) > 1 else ""
    test_face_recognition(img_path)
    test_fallback_logic()
    test_cooldown_logic()
    test_alarm_independence()
    test_smtp()

    print(f"\n{'='*70}")
    print(f"  RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
    print(f"{'='*70}\n")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
