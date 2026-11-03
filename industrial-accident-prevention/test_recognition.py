# -*- coding: utf-8 -*-
"""test_recognition.py — Automated tests for the face recognition bug fixes.

Tests:
  1. Database initialization and migration
  2. Idempotent re-init (no duplicates)
  3. Employee CRUD operations
  4. Template storage and retrieval
  5. Multi-template matching
  6. Persistence across simulated restarts
  7. Enrollment pipeline consistency
  8. Face crop extraction with keypoints
  9. Unknown face rejection
  10. Identity isolation between workers
"""

import os
import sys
import json
import time
import sqlite3
import shutil
import tempfile
from pathlib import Path

import numpy as np

# Ensure project modules are importable
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import employee_db

PASS = 0
FAIL = 0
ERRORS = []

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
        ERRORS.append(name)

def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
#  TEST 1: Database Initialization & Migration
# ══════════════════════════════════════════════════════════════

separator("TEST 1: Database Initialization & Migration")

c, t = employee_db.init()
test("Database initialised", c > 0, f"employees={c}, templates={t}")
test("Database file exists", employee_db.DB_PATH.exists())
test("Database file has content", os.path.getsize(employee_db.DB_PATH) > 0)

# Verify all employees from JSON were migrated
json_path = _ROOT / "employees.json"
if json_path.exists():
    with open(json_path, "r", encoding="utf-8") as f:
        json_emps = json.load(f)
    test("All JSON employees migrated", c >= len(json_emps),
         f"db={c}, json={len(json_emps)}")


# ══════════════════════════════════════════════════════════════
#  TEST 2: Idempotent Re-init (No Duplicates)
# ══════════════════════════════════════════════════════════════

separator("TEST 2: Idempotent Re-init")

c2, t2 = employee_db.init()
test("Employee count unchanged after re-init", c == c2,
     f"before={c}, after={c2}")
test("Template count unchanged after re-init", t == t2,
     f"before={t}, after={t2}")

# Third init
c3, t3 = employee_db.init()
test("Employee count stable on third init", c == c3)


# ══════════════════════════════════════════════════════════════
#  TEST 3: Employee CRUD
# ══════════════════════════════════════════════════════════════

separator("TEST 3: Employee CRUD")

# Add test employee
test_emp_id = "_TEST_EMP_001"
added = employee_db.add_employee(test_emp_id, "Test Person", "QA",
                                  "test@example.com", ["zone_1"])
test("Add new employee", added)

# Verify it exists
emp = employee_db.get_employee(test_emp_id)
test("Retrieve employee", emp is not None)
test("Employee name correct", emp and emp["full_name"] == "Test Person")
test("Employee email correct", emp and emp["personal_email"] == "test@example.com")

# Duplicate add should fail
added2 = employee_db.add_employee(test_emp_id, "Duplicate", "QA",
                                   "dup@example.com")
test("Duplicate add returns False", not added2)

# Count should be +1
new_count = employee_db.count_employees()
test("Count increased by 1", new_count == c + 1)

# Delete
employee_db.hard_delete_employee(test_emp_id)
test("Delete removes employee",
     employee_db.get_employee(test_emp_id) is None)
test("Count restored after delete",
     employee_db.count_employees() == c)


# ══════════════════════════════════════════════════════════════
#  TEST 4: Template Storage & Retrieval
# ══════════════════════════════════════════════════════════════

separator("TEST 4: Template Storage & Retrieval")

# Add test employee + template
employee_db.add_employee("_TEST_T1", "Template Test", "QA", "t1@test.com")
fake_embedding = np.random.randn(128).astype(np.float64)
tid = employee_db.add_template("_TEST_T1", fake_embedding, "Facenet")
test("Template added", tid is not None)

# Retrieve
templates = employee_db.get_templates_for_employee("_TEST_T1")
test("Template retrievable", len(templates) == 1)

if templates:
    retrieved = templates[0]["embedding"]
    dist = np.linalg.norm(fake_embedding - retrieved)
    test("Embedding preserved exactly", dist < 1e-10, f"dist={dist}")
    test("Embedding dim correct", templates[0]["embedding_dim"] == 128)

# Add second template (multi-template support)
fake_embedding_2 = np.random.randn(128).astype(np.float64)
tid2 = employee_db.add_template("_TEST_T1", fake_embedding_2, "Facenet")
test("Second template added", tid2 is not None)

templates2 = employee_db.get_templates_for_employee("_TEST_T1")
test("Multi-template support works", len(templates2) == 2)

# Invalid embedding rejection
bad = np.array([np.nan, np.inf, 0.0])
tid_bad = employee_db.add_template("_TEST_T1", bad, "Facenet")
test("Invalid embedding rejected", tid_bad is None)

# Cleanup
employee_db.hard_delete_employee("_TEST_T1")


# ══════════════════════════════════════════════════════════════
#  TEST 5: Multi-Template Matching
# ══════════════════════════════════════════════════════════════

separator("TEST 5: Multi-Template Matching")

templates = employee_db.get_all_active_templates()
test("Templates loaded from database", len(templates) > 0,
     f"count={len(templates)}")

# Verify template structure
if templates:
    t0 = templates[0]
    test("Template has employee_id", "employee_id" in t0)
    test("Template has full_name", "full_name" in t0)
    test("Template has embedding", "embedding" in t0)
    test("Embedding is numpy array", isinstance(t0["embedding"], np.ndarray))
    test("Embedding is finite", np.isfinite(t0["embedding"]).all())


# ══════════════════════════════════════════════════════════════
#  TEST 6: Persistence Across Simulated Restart
# ══════════════════════════════════════════════════════════════

separator("TEST 6: Persistence Across Restart")

# Record current counts
pre_emp = employee_db.count_employees()
pre_tmpl = employee_db.count_templates()

# Force a "restart" by resetting the module's initialized flag
employee_db._initialized = False

# Re-init (simulates application restart)
post_emp, post_tmpl = employee_db.init()
test("Employee count persists across restart",
     pre_emp == post_emp, f"before={pre_emp}, after={post_emp}")
test("Template count persists across restart",
     pre_tmpl == post_tmpl, f"before={pre_tmpl}, after={post_tmpl}")


# ══════════════════════════════════════════════════════════════
#  TEST 7: Enrollment Pipeline
# ══════════════════════════════════════════════════════════════

separator("TEST 7: Transactional Enrollment")

embs = [np.random.randn(128).astype(np.float64) for _ in range(3)]
success = employee_db.enroll_employee(
    "_TEST_ENROLL", "Enroll Test", embs,
    department="Testing", personal_email="enroll@test.com",
    zone_ids=["zone_1"], model_name="Facenet"
)
test("Enrollment succeeds", success)

emp = employee_db.get_employee("_TEST_ENROLL")
test("Enrolled employee exists", emp is not None)
test("Enrolled employee name", emp and emp["full_name"] == "Enroll Test")

tmpls = employee_db.get_templates_for_employee("_TEST_ENROLL")
test("All 3 templates stored", len(tmpls) == 3)

# Re-enroll same employee (should update, not duplicate)
embs2 = [np.random.randn(128).astype(np.float64)]
success2 = employee_db.enroll_employee(
    "_TEST_ENROLL", "Enroll Test Updated", embs2,
    personal_email="updated@test.com"
)
test("Re-enrollment succeeds", success2)

emp2 = employee_db.get_employee("_TEST_ENROLL")
test("Name updated on re-enroll", emp2 and emp2["full_name"] == "Enroll Test Updated")

tmpls2 = employee_db.get_templates_for_employee("_TEST_ENROLL")
test("Templates accumulated (3+1=4)", len(tmpls2) == 4)

# Empty enrollment should fail
success3 = employee_db.enroll_employee("_TEST_EMPTY", "Empty", [])
test("Empty enrollment rejected", not success3)

# Cleanup
employee_db.hard_delete_employee("_TEST_ENROLL")
employee_db.hard_delete_employee("_TEST_EMPTY")


# ══════════════════════════════════════════════════════════════
#  TEST 8: Face Crop Extraction with Keypoints
# ══════════════════════════════════════════════════════════════

separator("TEST 8: Face Crop Extraction")

import face_identity

# Create a dummy frame (640x480 black image with a white rectangle as "face")
frame = np.zeros((480, 640, 3), dtype=np.uint8)
# Draw a white square where a face would be
cv2_available = True
try:
    import cv2
    cv2.rectangle(frame, (280, 100), (360, 200), (255, 255, 255), -1)
except ImportError:
    cv2_available = False

bbox = (250, 80, 390, 450)  # full body bbox

# Test with keypoints (nose at 320,150, eyes at 300,130 and 340,130)
keypoints = [
    (320, 150, 0.9),  # nose
    (300, 130, 0.85), # left eye
    (340, 130, 0.85), # right eye
    (280, 145, 0.7),  # left ear
    (360, 145, 0.7),  # right ear
] + [(0, 0, 0)] * 12  # remaining body keypoints

crop_kp = face_identity._extract_face_crop(frame, bbox, keypoints)
test("Keypoint crop extracted", crop_kp is not None)
if crop_kp is not None:
    h, w = crop_kp.shape[:2]
    aspect = w / max(h, 1)
    test("Keypoint crop has reasonable aspect ratio",
         0.4 < aspect < 2.5, f"aspect={aspect:.2f}, size={w}x{h}")
    test("Keypoint crop is not too wide/flat",
         h > 40, f"height={h}")

# Test without keypoints (fallback)
crop_fallback = face_identity._extract_face_crop(frame, bbox, None)
test("Fallback crop extracted", crop_fallback is not None)

# Test with low-confidence keypoints (should fallback)
bad_kps = [(0, 0, 0.01)] * 17  # all below threshold
crop_bad = face_identity._extract_face_crop(frame, bbox, bad_kps)
test("Low-conf keypoints produce fallback crop", crop_bad is not None)


# ══════════════════════════════════════════════════════════════
#  TEST 9: Identity Isolation Between Workers
# ══════════════════════════════════════════════════════════════

separator("TEST 9: Identity Isolation")

face_identity.clear_all()

# Simulate two different workers
face_identity._identity_cache[1] = {
    "employee_id": "emp_A",
    "full_name": "Alice",
    "distance": 0.2,
    "identified_at": time.time(),
    "last_attempt": time.time(),
}
face_identity._identity_cache[2] = {
    "employee_id": "emp_B",
    "full_name": "Bob",
    "distance": 0.3,
    "identified_at": time.time(),
    "last_attempt": time.time(),
}

id1 = face_identity.get_cached_identity(1)
id2 = face_identity.get_cached_identity(2)
test("Worker 1 identity is Alice",
     id1["display_name"] == "Alice")
test("Worker 2 identity is Bob",
     id2["display_name"] == "Bob")
test("Identities don't leak between workers",
     id1["employee_id"] != id2["employee_id"])

# Clear one worker
face_identity.clear_worker(1)
id1_after = face_identity.get_cached_identity(1)
test("Cleared worker returns generic name",
     id1_after["display_name"] == "Worker 1")
test("Other worker unaffected by clear",
     face_identity.get_cached_identity(2)["display_name"] == "Bob")

face_identity.clear_all()


# ══════════════════════════════════════════════════════════════
#  TEST 10: Legacy File Sync
# ══════════════════════════════════════════════════════════════

separator("TEST 10: Legacy File Sync")

employee_db.sync_to_legacy_files()

json_path = _ROOT / "employees.json"
pkl_path = _ROOT / "face_encodings.pkl"

test("employees.json exists after sync", json_path.exists())
test("face_encodings.pkl exists after sync", pkl_path.exists())

if json_path.exists():
    with open(json_path, "r", encoding="utf-8") as f:
        synced = json.load(f)
    db_count = employee_db.count_employees()
    test("JSON employee count matches DB",
         len(synced) == db_count, f"json={len(synced)}, db={db_count}")

if pkl_path.exists():
    import pickle
    with open(pkl_path, "rb") as f:
        synced_enc = pickle.load(f)
    test("Pickle has encodings", len(synced_enc) > 0)


# ══════════════════════════════════════════════════════════════
#  RESULTS
# ══════════════════════════════════════════════════════════════

separator("RESULTS")
total = PASS + FAIL
print(f"\n  Total:  {total}")
print(f"  Passed: {PASS}")
print(f"  Failed: {FAIL}")
if ERRORS:
    print(f"\n  Failed tests:")
    for e in ERRORS:
        print(f"    - {e}")
print()

sys.exit(0 if FAIL == 0 else 1)
