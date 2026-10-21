# -*- coding: utf-8 -*-
"""enroll_dashboard.py — Streamlit employee enrollment for face recognition (deepface)."""

import json
import os
import pickle
import re
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

try:
    from deepface import DeepFace
    _HAS_DF = True
except ImportError:
    _HAS_DF = False

_ROOT = Path(__file__).resolve().parent
EMPLOYEES_FILE = str(_ROOT / "employees.json")
ENCODINGS_FILE = str(_ROOT / "face_encodings.pkl")
MANAGERS_FILE  = str(_ROOT / "managers.json")

# Must match face_recognizer.py
MODEL_NAME = os.environ.get("FACE_MODEL", "Facenet")


# ── helpers ──

def _load_employees() -> list[dict]:
    if not os.path.exists(EMPLOYEES_FILE):
        return []
    with open(EMPLOYEES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_employees(data: list[dict]):
    with open(EMPLOYEES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_encodings() -> dict[str, np.ndarray]:
    if not os.path.exists(ENCODINGS_FILE):
        return {}
    with open(ENCODINGS_FILE, "rb") as f:
        return pickle.load(f)


def _save_encodings(data: dict[str, np.ndarray]):
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)


def _load_zone_ids() -> list[str]:
    """Get zone_ids from managers.json for the dropdown."""
    if not os.path.exists(MANAGERS_FILE):
        return []
    try:
        with open(MANAGERS_FILE, "r", encoding="utf-8") as f:
            managers = json.load(f)
        return list({m["zone_id"] for m in managers if "zone_id" in m})
    except Exception:
        return []


def _valid_email(email: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$", email))


def _get_embedding(bgr_image: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Extract face embedding. Returns (embedding, error_message)."""
    try:
        results = DeepFace.represent(
            img_path=bgr_image,
            model_name=MODEL_NAME,
            enforce_detection=True,
            detector_backend="opencv",
        )
        if results and len(results) > 0:
            if len(results) > 1:
                return None, f"Multiple faces detected ({len(results)}). Use a photo with only one face."
            return np.array(results[0]["embedding"], dtype=np.float64), ""
        return None, "No face detected."
    except ValueError as e:
        if "Face could not be detected" in str(e):
            return None, "No face detected. Use a clear, front-facing photo."
        return None, str(e)
    except Exception as e:
        return None, str(e)


# ── main UI ──

def main():
    st.set_page_config(page_title="Employee Face Enrollment", page_icon="Person", layout="wide")
    st.title("Employee Face Enrollment")

    if not _HAS_DF:
        st.error("deepface library not installed. Run: pip install deepface tf-keras")
        return

    tab_enroll, tab_manage = st.tabs(["Enroll New Employee", "Manage Enrolled Employees"])

    # ── Tab 1: Enroll ──
    with tab_enroll:
        st.subheader("Register a New Employee")

        zone_ids = _load_zone_ids()

        col1, col2 = st.columns(2)
        with col1:
            emp_id = st.text_input("Employee ID *", placeholder="e.g. EMP001")
            full_name = st.text_input("Full Name *", placeholder="e.g. John Doe")
            department = st.text_input("Department", placeholder="e.g. Maintenance")
        with col2:
            personal_email = st.text_input("Personal Email *", placeholder="e.g. john@example.com")
            if zone_ids:
                selected_zones = st.multiselect("Authorised Zone(s)", zone_ids,
                                                 help="Which danger zone(s) this employee works near")
            else:
                selected_zones_text = st.text_input("Zone ID(s) - comma-separated",
                                                     placeholder="zone_123,zone_456")
                selected_zones = [z.strip() for z in selected_zones_text.split(",") if z.strip()]

        st.markdown("---")
        st.markdown("**Upload 2-3 clear face photos** (front-facing, good lighting)")
        photos = st.file_uploader(
            "Upload Photos",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload 2-3 photos for better accuracy",
        )

        if st.button("Enroll Employee", type="primary"):
            # Validation
            errors = []
            if not emp_id.strip():
                errors.append("Employee ID is required")
            if not full_name.strip():
                errors.append("Full Name is required")
            if not personal_email.strip():
                errors.append("Email is required")
            elif not _valid_email(personal_email.strip()):
                errors.append("Invalid email format")
            if not photos or len(photos) < 1:
                errors.append("Upload at least 1 photo")

            # Check duplicate ID
            employees = _load_employees()
            if any(e["employee_id"] == emp_id.strip() for e in employees):
                errors.append(f"Employee ID '{emp_id.strip()}' already enrolled")

            if errors:
                for e in errors:
                    st.error(e)
            else:
                # Compute embeddings from each photo
                all_embeddings = []
                for i, photo in enumerate(photos):
                    img_bytes = photo.read()
                    nparr = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is None:
                        st.error(f"Photo {i+1} ({photo.name}): could not decode image")
                        continue

                    with st.spinner(f"Processing photo {i+1}..."):
                        embedding, err = _get_embedding(img)

                    if err:
                        st.error(f"Photo {i+1} ({photo.name}): {err}")
                        continue

                    all_embeddings.append(embedding)
                    st.success(f"Photo {i+1} ({photo.name}): face encoded successfully")

                if not all_embeddings:
                    st.error("No valid face embeddings could be computed. Enrollment aborted.")
                else:
                    # Average the embeddings for robustness
                    avg_embedding = np.mean(all_embeddings, axis=0)

                    # Save metadata
                    emp_data = {
                        "employee_id": emp_id.strip(),
                        "full_name": full_name.strip(),
                        "department": department.strip(),
                        "personal_email": personal_email.strip(),
                        "zone_ids": selected_zones,
                    }
                    employees.append(emp_data)
                    _save_employees(employees)

                    # Save encoding
                    encodings = _load_encodings()
                    encodings[emp_id.strip()] = avg_embedding
                    _save_encodings(encodings)

                    st.success(f"Employee **{full_name.strip()}** enrolled successfully "
                               f"with {len(all_embeddings)} photo(s)!")
                    st.balloons()

    # ── Tab 2: Manage ──
    with tab_manage:
        st.subheader("Enrolled Employees")

        employees = _load_employees()
        if not employees:
            st.info("No employees enrolled yet.")
        else:
            import pandas as pd
            df = pd.DataFrame(employees)
            cols = ["employee_id", "full_name", "department", "personal_email", "zone_ids"]
            existing_cols = [c for c in cols if c in df.columns]
            st.dataframe(df[existing_cols], use_container_width=True, hide_index=True)

            st.markdown("---")
            st.subheader("Delete an Employee")
            emp_ids = [e["employee_id"] for e in employees]
            delete_id = st.selectbox("Select Employee ID to delete", emp_ids)
            if st.button("Delete Employee", type="secondary"):
                employees = [e for e in employees if e["employee_id"] != delete_id]
                _save_employees(employees)

                encodings = _load_encodings()
                encodings.pop(delete_id, None)
                _save_encodings(encodings)

                st.success(f"Employee {delete_id} deleted (metadata + encoding removed)")
                st.rerun()


if __name__ == "__main__":
    main()
