# -*- coding: utf-8 -*-
"""employee_db.py — Persistent SQLite database for employee face recognition.

Provides atomic, transactional storage for:
  - Employee records (employee_id, name, email, department, zones).
  - Face templates (embeddings with model metadata, multiple per employee).
  - Safety events (danger-zone entries with recognition results).

Design decisions:
  - The database file is resolved relative to the project root directory
    (Path(__file__).resolve().parent), NOT the current working directory.
  - Schema is created idempotently (IF NOT EXISTS).
  - Foreign keys are enabled and enforced.
  - On first init, existing employees.json + face_encodings.pkl data is
    migrated automatically (idempotent — skips already-migrated records).
  - All public functions use parameterised queries.
  - Embeddings are stored as binary BLOBs (numpy .tobytes() / np.frombuffer()).
"""

import json
import os
import pickle
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Paths ──
_ROOT = Path(__file__).resolve().parent
DB_DIR = _ROOT / "data"
DB_PATH = DB_DIR / "industrial_safety.db"

# Legacy data files (for migration)
EMPLOYEES_JSON = _ROOT / "employees.json"
ENCODINGS_PKL = _ROOT / "face_encodings.pkl"

# ── Thread-local connections ──
_local = threading.local()
_init_lock = threading.Lock()
_initialized = False


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE CONNECTION & SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

def _get_conn() -> sqlite3.Connection:
    """Get a thread-local SQLite connection (one per thread)."""
    conn = getattr(_local, "conn", None)
    if conn is None:
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        _local.conn = conn
    return conn


def _create_schema():
    """Create database tables if they don't exist."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS employees (
            employee_id   TEXT PRIMARY KEY,
            full_name     TEXT NOT NULL,
            department    TEXT DEFAULT '',
            personal_email TEXT DEFAULT '',
            zone_ids      TEXT DEFAULT '[]',
            registered_at TEXT DEFAULT (datetime('now','localtime')),
            is_active     INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS face_templates (
            template_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id   TEXT NOT NULL REFERENCES employees(employee_id)
                          ON DELETE CASCADE,
            embedding     BLOB NOT NULL,
            model_name    TEXT NOT NULL DEFAULT 'Facenet',
            embedding_dim INTEGER NOT NULL DEFAULT 128,
            source_info   TEXT DEFAULT '',
            created_at    TEXT DEFAULT (datetime('now','localtime')),
            is_active     INTEGER DEFAULT 1
        );

        CREATE TABLE IF NOT EXISTS safety_events (
            event_id      INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id   TEXT,
            display_name  TEXT,
            event_type    TEXT NOT NULL,
            zone_id       TEXT,
            camera_id     TEXT,
            recognition_distance REAL,
            event_details TEXT DEFAULT '',
            created_at    TEXT DEFAULT (datetime('now','localtime'))
        );

        CREATE INDEX IF NOT EXISTS idx_tmpl_emp
            ON face_templates(employee_id);
        CREATE INDEX IF NOT EXISTS idx_tmpl_active
            ON face_templates(is_active);
        CREATE INDEX IF NOT EXISTS idx_events_emp
            ON safety_events(employee_id);
        CREATE INDEX IF NOT EXISTS idx_events_time
            ON safety_events(created_at);
    """)
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
#  MIGRATION FROM LEGACY JSON / PICKLE
# ══════════════════════════════════════════════════════════════════════════════

def _migrate_legacy_data():
    """Migrate employees.json + face_encodings.pkl into SQLite (idempotent)."""
    conn = _get_conn()

    # Load legacy employees
    employees = []
    if EMPLOYEES_JSON.exists():
        try:
            with open(EMPLOYEES_JSON, "r", encoding="utf-8") as f:
                employees = json.load(f)
            print(f"[employee_db] Found {len(employees)} employees in legacy employees.json")
        except Exception as e:
            print(f"[employee_db] WARNING: Could not read employees.json: {e}")

    # Load legacy encodings
    encodings = {}
    if ENCODINGS_PKL.exists():
        try:
            with open(ENCODINGS_PKL, "rb") as f:
                encodings = pickle.load(f)
            print(f"[employee_db] Found {len(encodings)} encodings in legacy face_encodings.pkl")
        except Exception as e:
            print(f"[employee_db] WARNING: Could not read face_encodings.pkl: {e}")

    if not employees and not encodings:
        return

    migrated_emp = 0
    migrated_tmpl = 0
    skipped_emp = 0

    for emp in employees:
        eid = emp.get("employee_id", "").strip()
        if not eid:
            continue

        # Check if already migrated
        row = conn.execute(
            "SELECT employee_id FROM employees WHERE employee_id = ?", (eid,)
        ).fetchone()

        if row is not None:
            skipped_emp += 1
            continue

        # Insert employee
        zones = emp.get("zone_ids", [])
        if isinstance(zones, list):
            zones = json.dumps(zones)

        conn.execute("""
            INSERT INTO employees (employee_id, full_name, department,
                                   personal_email, zone_ids, is_active)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (
            eid,
            emp.get("full_name", eid),
            emp.get("department", ""),
            emp.get("personal_email", ""),
            zones,
        ))
        migrated_emp += 1

        # Insert face template if encoding exists for this employee
        if eid in encodings:
            emb = np.asarray(encodings[eid], dtype=np.float64)
            if emb.ndim >= 1 and emb.size > 0 and np.isfinite(emb).all():
                conn.execute("""
                    INSERT INTO face_templates
                        (employee_id, embedding, model_name, embedding_dim, source_info)
                    VALUES (?, ?, ?, ?, 'migrated_from_pkl')
                """, (eid, emb.tobytes(), "Facenet", emb.shape[0]))
                migrated_tmpl += 1

    # Handle encodings for IDs not in employees.json (orphan encodings)
    for eid, emb_raw in encodings.items():
        eid = str(eid).strip()
        if not eid:
            continue

        row = conn.execute(
            "SELECT employee_id FROM employees WHERE employee_id = ?", (eid,)
        ).fetchone()

        if row is None:
            # Create a minimal employee record for orphan encoding
            conn.execute("""
                INSERT INTO employees (employee_id, full_name, is_active)
                VALUES (?, ?, 1)
            """, (eid, eid))
            migrated_emp += 1

        # Check if template already exists
        tmpl = conn.execute("""
            SELECT template_id FROM face_templates
            WHERE employee_id = ? AND source_info = 'migrated_from_pkl'
        """, (eid,)).fetchone()

        if tmpl is None:
            emb = np.asarray(emb_raw, dtype=np.float64)
            if emb.ndim >= 1 and emb.size > 0 and np.isfinite(emb).all():
                conn.execute("""
                    INSERT INTO face_templates
                        (employee_id, embedding, model_name, embedding_dim, source_info)
                    VALUES (?, ?, ?, ?, 'migrated_from_pkl')
                """, (eid, emb.tobytes(), "Facenet", emb.shape[0]))
                migrated_tmpl += 1

    conn.commit()

    if migrated_emp > 0 or migrated_tmpl > 0:
        print(f"[employee_db] Migration complete: "
              f"{migrated_emp} employees, {migrated_tmpl} templates migrated, "
              f"{skipped_emp} already existed")
    else:
        print(f"[employee_db] Migration: all {skipped_emp} employees already in database")


# ══════════════════════════════════════════════════════════════════════════════
#  INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def init():
    """Initialise the database: create schema, run migrations.

    Safe to call multiple times — idempotent.
    Returns (employee_count, template_count).
    """
    global _initialized
    with _init_lock:
        if _initialized:
            c = count_employees()
            t = count_templates()
            return c, t

        print(f"[employee_db] Database path: {DB_PATH}")
        _create_schema()
        _migrate_legacy_data()
        _initialized = True

        c = count_employees()
        t = count_templates()
        print(f"[employee_db] Initialised: {c} active employees, {t} face templates")
        return c, t


# ══════════════════════════════════════════════════════════════════════════════
#  EMPLOYEE CRUD
# ══════════════════════════════════════════════════════════════════════════════

def add_employee(employee_id: str, full_name: str, department: str = "",
                 personal_email: str = "", zone_ids: list = None) -> bool:
    """Add a new employee. Returns True if added, False if already exists."""
    conn = _get_conn()
    try:
        zones_json = json.dumps(zone_ids or [])
        conn.execute("""
            INSERT INTO employees (employee_id, full_name, department,
                                   personal_email, zone_ids, is_active)
            VALUES (?, ?, ?, ?, ?, 1)
        """, (employee_id, full_name, department, personal_email, zones_json))
        conn.commit()
        print(f"[employee_db] Added employee: {employee_id} ({full_name})")
        return True
    except sqlite3.IntegrityError:
        print(f"[employee_db] Employee {employee_id} already exists")
        return False


def get_employee(employee_id: str) -> dict | None:
    """Get a single employee by ID."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM employees WHERE employee_id = ? AND is_active = 1",
        (employee_id,)
    ).fetchone()
    if row is None:
        return None
    return _row_to_emp(row)


def get_all_employees() -> list[dict]:
    """Get all active employees."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM employees WHERE is_active = 1 ORDER BY full_name"
    ).fetchall()
    return [_row_to_emp(r) for r in rows]


def delete_employee(employee_id: str) -> bool:
    """Soft-delete an employee (set is_active = 0). Also deactivates templates."""
    conn = _get_conn()
    conn.execute(
        "UPDATE employees SET is_active = 0 WHERE employee_id = ?",
        (employee_id,)
    )
    conn.execute(
        "UPDATE face_templates SET is_active = 0 WHERE employee_id = ?",
        (employee_id,)
    )
    conn.commit()
    return True


def hard_delete_employee(employee_id: str) -> bool:
    """Permanently delete an employee and all templates."""
    conn = _get_conn()
    conn.execute("DELETE FROM face_templates WHERE employee_id = ?", (employee_id,))
    conn.execute("DELETE FROM employees WHERE employee_id = ?", (employee_id,))
    conn.commit()
    return True


def count_employees() -> int:
    conn = _get_conn()
    row = conn.execute("SELECT COUNT(*) FROM employees WHERE is_active = 1").fetchone()
    return row[0] if row else 0


def employee_exists(employee_id: str) -> bool:
    conn = _get_conn()
    row = conn.execute(
        "SELECT 1 FROM employees WHERE employee_id = ?", (employee_id,)
    ).fetchone()
    return row is not None


def _row_to_emp(row) -> dict:
    """Convert a sqlite3.Row to an employee dict."""
    d = dict(row)
    # Parse zone_ids JSON
    try:
        d["zone_ids"] = json.loads(d.get("zone_ids", "[]"))
    except (json.JSONDecodeError, TypeError):
        d["zone_ids"] = []
    return d


# ══════════════════════════════════════════════════════════════════════════════
#  FACE TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

def add_template(employee_id: str, embedding: np.ndarray,
                 model_name: str = "Facenet", source_info: str = "") -> int | None:
    """Add a face template. Returns template_id or None on error."""
    conn = _get_conn()

    # Validate embedding
    emb = np.asarray(embedding, dtype=np.float64).flatten()
    if emb.size == 0 or not np.isfinite(emb).all():
        print(f"[employee_db] ERROR: Invalid embedding for {employee_id}")
        return None

    try:
        cur = conn.execute("""
            INSERT INTO face_templates
                (employee_id, embedding, model_name, embedding_dim, source_info)
            VALUES (?, ?, ?, ?, ?)
        """, (employee_id, emb.tobytes(), model_name, emb.shape[0], source_info))
        conn.commit()
        tid = cur.lastrowid
        print(f"[employee_db] Added template {tid} for {employee_id} "
              f"(dim={emb.shape[0]}, model={model_name})")
        return tid
    except sqlite3.IntegrityError as e:
        print(f"[employee_db] ERROR: Cannot add template for {employee_id}: {e}")
        return None


def get_all_active_templates() -> list[dict]:
    """Get all active face templates with employee info.

    Returns list of dicts with keys:
        template_id, employee_id, full_name, embedding (np.ndarray),
        model_name, embedding_dim
    """
    conn = _get_conn()
    rows = conn.execute("""
        SELECT t.template_id, t.employee_id, e.full_name, t.embedding,
               t.model_name, t.embedding_dim
        FROM face_templates t
        JOIN employees e ON t.employee_id = e.employee_id
        WHERE t.is_active = 1 AND e.is_active = 1
        ORDER BY t.employee_id, t.template_id
    """).fetchall()

    templates = []
    for row in rows:
        emb_blob = row["embedding"]
        dim = row["embedding_dim"]
        try:
            emb = np.frombuffer(emb_blob, dtype=np.float64)
            if emb.shape[0] != dim:
                print(f"[employee_db] WARNING: Template {row['template_id']} "
                      f"dim mismatch: stored={dim}, actual={emb.shape[0]}")
                continue
            if not np.isfinite(emb).all():
                print(f"[employee_db] WARNING: Template {row['template_id']} "
                      f"has non-finite values, skipping")
                continue
        except Exception as e:
            print(f"[employee_db] WARNING: Failed to decode template "
                  f"{row['template_id']}: {e}")
            continue

        templates.append({
            "template_id": row["template_id"],
            "employee_id": row["employee_id"],
            "full_name": row["full_name"],
            "embedding": emb,
            "model_name": row["model_name"],
            "embedding_dim": dim,
        })

    return templates


def get_templates_for_employee(employee_id: str) -> list[dict]:
    """Get all active templates for a specific employee."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT template_id, embedding, model_name, embedding_dim
        FROM face_templates
        WHERE employee_id = ? AND is_active = 1
    """, (employee_id,)).fetchall()

    templates = []
    for row in rows:
        try:
            emb = np.frombuffer(row["embedding"], dtype=np.float64)
            templates.append({
                "template_id": row["template_id"],
                "embedding": emb,
                "model_name": row["model_name"],
                "embedding_dim": row["embedding_dim"],
            })
        except Exception:
            continue
    return templates


def count_templates() -> int:
    conn = _get_conn()
    row = conn.execute(
        "SELECT COUNT(*) FROM face_templates WHERE is_active = 1"
    ).fetchone()
    return row[0] if row else 0


def delete_templates_for_employee(employee_id: str):
    """Deactivate all templates for an employee."""
    conn = _get_conn()
    conn.execute(
        "UPDATE face_templates SET is_active = 0 WHERE employee_id = ?",
        (employee_id,)
    )
    conn.commit()


# ══════════════════════════════════════════════════════════════════════════════
#  SAFETY EVENTS
# ══════════════════════════════════════════════════════════════════════════════

def log_event(employee_id: str | None, display_name: str,
              event_type: str, zone_id: str = "",
              recognition_distance: float = None,
              event_details: str = "") -> int:
    """Log a safety event. Returns event_id."""
    conn = _get_conn()
    cur = conn.execute("""
        INSERT INTO safety_events
            (employee_id, display_name, event_type, zone_id,
             recognition_distance, event_details)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (employee_id, display_name, event_type, zone_id,
          recognition_distance, event_details))
    conn.commit()
    return cur.lastrowid


def get_recent_events(limit: int = 50) -> list[dict]:
    """Get most recent safety events."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT * FROM safety_events
        ORDER BY created_at DESC LIMIT ?
    """, (limit,)).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
#  LEGACY FILE SYNC (keep JSON/pkl updated for backward compat)
# ══════════════════════════════════════════════════════════════════════════════

def sync_to_legacy_files():
    """Write current database state back to employees.json and face_encodings.pkl.

    This ensures backward compatibility with any code that still reads
    the legacy files directly.
    """
    try:
        employees = get_all_employees()
        emp_list = []
        for emp in employees:
            emp_list.append({
                "employee_id": emp["employee_id"],
                "full_name": emp["full_name"],
                "department": emp.get("department", ""),
                "personal_email": emp.get("personal_email", ""),
                "zone_ids": emp.get("zone_ids", []),
            })
        with open(EMPLOYEES_JSON, "w", encoding="utf-8") as f:
            json.dump(emp_list, f, indent=2)

        # For face_encodings.pkl, store one encoding per employee
        # (the most recent active template)
        templates = get_all_active_templates()
        enc_dict = {}
        for tmpl in templates:
            eid = tmpl["employee_id"]
            # Last template wins (most recently added)
            enc_dict[eid] = tmpl["embedding"]

        with open(ENCODINGS_PKL, "wb") as f:
            pickle.dump(enc_dict, f)

        print(f"[employee_db] Synced to legacy files: "
              f"{len(emp_list)} employees, {len(enc_dict)} encodings")
    except Exception as e:
        print(f"[employee_db] WARNING: Failed to sync legacy files: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  ENROLL EMPLOYEE (TRANSACTIONAL)
# ══════════════════════════════════════════════════════════════════════════════

def enroll_employee(employee_id: str, full_name: str,
                    embeddings: list[np.ndarray],
                    department: str = "", personal_email: str = "",
                    zone_ids: list = None,
                    model_name: str = "Facenet") -> bool:
    """Transactionally enroll an employee with face templates.

    If the employee already exists, adds new templates without duplicating
    the employee record.

    Returns True on success, False on failure.
    """
    conn = _get_conn()

    if not embeddings:
        print(f"[employee_db] ERROR: No valid embeddings for enrollment of {employee_id}")
        return False

    try:
        # Start transaction
        conn.execute("BEGIN")

        # Upsert employee
        existing = conn.execute(
            "SELECT employee_id FROM employees WHERE employee_id = ?",
            (employee_id,)
        ).fetchone()

        if existing is None:
            zones_json = json.dumps(zone_ids or [])
            conn.execute("""
                INSERT INTO employees (employee_id, full_name, department,
                                       personal_email, zone_ids, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
            """, (employee_id, full_name, department, personal_email, zones_json))
        else:
            # Update existing record
            zones_json = json.dumps(zone_ids or [])
            conn.execute("""
                UPDATE employees SET full_name = ?, department = ?,
                    personal_email = ?, zone_ids = ?, is_active = 1
                WHERE employee_id = ?
            """, (full_name, department, personal_email, zones_json, employee_id))

        # Add face templates
        added = 0
        for emb in embeddings:
            emb = np.asarray(emb, dtype=np.float64).flatten()
            if emb.size == 0 or not np.isfinite(emb).all():
                continue
            conn.execute("""
                INSERT INTO face_templates
                    (employee_id, embedding, model_name, embedding_dim, source_info)
                VALUES (?, ?, ?, ?, 'enrollment')
            """, (employee_id, emb.tobytes(), model_name, emb.shape[0]))
            added += 1

        if added == 0:
            conn.execute("ROLLBACK")
            print(f"[employee_db] ERROR: No valid embeddings saved for {employee_id}")
            return False

        conn.execute("COMMIT")
        print(f"[employee_db] Enrolled {employee_id} ({full_name}) "
              f"with {added} template(s)")

        # Sync to legacy files for backward compat
        sync_to_legacy_files()
        return True

    except Exception as e:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        print(f"[employee_db] ERROR: Enrollment failed for {employee_id}: {e}")
        return False
