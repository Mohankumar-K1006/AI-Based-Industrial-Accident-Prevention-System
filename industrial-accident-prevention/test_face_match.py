# -*- coding: utf-8 -*-
"""test_face_match.py — Offline face recognition test (no camera needed).

Usage:
    python test_face_match.py                     # tests with zero enrolled employees
    python test_face_match.py path/to/photo.jpg   # tests matching against enrolled employees
"""

import os
import sys
from pathlib import Path

# Ensure project imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    from face_recognizer import identify_face, _load_employees, _load_encodings

    employees = _load_employees()
    encodings = _load_encodings()

    print(f"Enrolled employees : {len(employees)}")
    print(f"Stored encodings   : {len(encodings)}")
    if employees:
        for e in employees:
            print(f"  - {e['employee_id']}: {e['full_name']} ({e['personal_email']})")
    print()

    if len(sys.argv) < 2:
        print("No test image provided.")
        print("Testing with a blank image (should always return UNIDENTIFIED)...")
        import numpy as np
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        emp_id, dist = identify_face(blank)
        print(f"  Result: {'MATCHED ' + emp_id if emp_id else 'UNIDENTIFIED'}")
        print(f"  Distance: {dist}")
        print()
        print("OK - No crash with zero/empty enrollment")
        return

    img_path = sys.argv[1]
    if not os.path.exists(img_path):
        print(f"ERROR: File not found: {img_path}")
        sys.exit(1)

    import cv2
    img = cv2.imread(img_path)
    if img is None:
        print(f"ERROR: Could not decode image: {img_path}")
        sys.exit(1)

    print(f"Testing image: {img_path}")
    emp_id, dist = identify_face(img)

    if emp_id:
        emp = None
        for e in employees:
            if e["employee_id"] == emp_id:
                emp = e
                break
        print(f"  MATCHED: {emp_id}")
        print(f"     Name    : {emp['full_name'] if emp else 'N/A'}")
        print(f"     Distance: {dist:.4f}")
    else:
        print(f"  UNIDENTIFIED (no match within threshold)")
        print(f"     Best distance: {dist:.4f}" if dist >= 0 else "     No face detected in image")


if __name__ == "__main__":
    main()
