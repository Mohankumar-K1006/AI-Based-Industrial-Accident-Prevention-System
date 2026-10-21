# -*- coding: utf-8 -*-
"""manager_dashboard.py — Streamlit UI to manage zone-manager assignments."""

import json
import os
import re
from pathlib import Path

import streamlit as st

_ROOT = Path(__file__).resolve().parent
MANAGERS_FILE = str(_ROOT / "managers.json")
ZONES_FILE    = str(_ROOT / "zones_config.json")


def _load() -> list[dict]:
    if not os.path.exists(MANAGERS_FILE):
        return []
    with open(MANAGERS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_zone_ids() -> list[str]:
    if not os.path.exists(ZONES_FILE):
        return ["default"]
    try:
        with open(ZONES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Find all DANGER zones or just all zones
            zones = [z.get("id", z.get("name")) for z in data.get("zones", [])]
            return zones if zones else ["default"]
    except Exception:
        return ["default"]


def _save(data: list[dict]):
    with open(MANAGERS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    try:
        os.utime(MANAGERS_FILE, None)
    except Exception:
        pass


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
PHONE_RE = re.compile(r"^\+?\d{7,15}$")

st.set_page_config(page_title="Manager Config", page_icon="👷")
st.title("👷 Zone Manager Configuration")

managers = _load()
available_zones = _load_zone_ids()

# ── Add / Edit form ──
st.subheader("Add / Edit Manager")
with st.form("mgr_form", clear_on_submit=True):
    zone_id      = st.selectbox("Zone ID (Select a drawn zone)", available_zones)
    manager_name = st.text_input("Manager Name")
    email        = st.text_input("Email")
    phone        = st.text_input("Phone (with country code)", placeholder="+1234567890")
    machine_name = st.text_input("Machine Name")
    submitted    = st.form_submit_button("Save")

if submitted:
    errors = []
    if not zone_id.strip():
        errors.append("Zone ID is required.")
    if not manager_name.strip():
        errors.append("Manager name is required.")
    if not EMAIL_RE.match(email.strip()):
        errors.append("Invalid email format.")
    if not PHONE_RE.match(phone.strip().replace(" ", "")):
        errors.append("Invalid phone format (7-15 digits, optional leading +).")
    if errors:
        for e in errors:
            st.error(e)
    else:
        entry = {
            "zone_id": zone_id.strip(),
            "manager_name": manager_name.strip(),
            "email": email.strip(),
            "phone": phone.strip(),
            "machine_name": machine_name.strip(),
        }
        # upsert by zone_id
        found = False
        for i, m in enumerate(managers):
            if m["zone_id"] == entry["zone_id"]:
                managers[i] = entry
                found = True
                break
        if not found:
            managers.append(entry)
        _save(managers)
        st.success(f"Saved manager for zone '{entry['zone_id']}'")
        st.rerun()

# ── Current entries table ──
st.subheader("Current Managers")
if not managers:
    st.info("No managers configured yet.")
else:
    for i, m in enumerate(managers):
        c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
        with c1:
            st.write(f"**{m['zone_id']}** — {m['manager_name']} "
                     f"| `{m['email']}` | {m['phone']} | Machine: *{m['machine_name']}*")
        with c2:
            if st.button("✉️ Test", key=f"test_{i}"):
                try:
                    from datetime import datetime
                    from notifier import _send_email
                    ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S (TEST)")
                    _send_email(m, m["zone_id"], m.get("machine_name", m["zone_id"]), ts_now, "")
                    st.success(f"Test sent to {m['email']}!")
                except Exception as e:
                    st.error(f"Error: {e}")
        with c3:
            if st.button("Edit", key=f"edit_{i}"):
                st.session_state["edit_zone"] = m["zone_id"]
                st.rerun()
        with c4:
            if st.button("Delete", key=f"del_{i}"):
                managers.pop(i)
                _save(managers)
                st.rerun()
