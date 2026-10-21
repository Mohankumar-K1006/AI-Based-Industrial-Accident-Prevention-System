# -*- coding: utf-8 -*-
"""test_notify.py — Send a test email+SMS for the first manager in managers.json."""

import os
import sys
from pathlib import Path

# ensure imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent))

from notifier import _load_managers, _send_email, _send_sms  # noqa: E402

managers = _load_managers()
if not managers:
    print("No managers in managers.json — add one first.")
    sys.exit(1)

m = managers[0]
zone = m["zone_id"]
machine = m.get("machine_name", zone)
ts = "2026-01-01 00:00:00 (TEST)"

# use latest real incident camera snapshot if available
alerts_dir = Path(__file__).resolve().parent / "alerts"
real_images = sorted(
    [p for p in alerts_dir.glob("*.jpg") if p.name != "test_notify.jpg"],
    key=os.path.getmtime,
    reverse=True
) if alerts_dir.exists() else []

if real_images:
    test_img = str(real_images[0])
else:
    test_img = ""

print(f"Manager : {m['manager_name']} ({m['email']}, {m['phone']})")
print(f"Zone    : {zone}  |  Machine: {machine}")
print(f"Image   : {test_img or '(none)'}")
print()

print("— Sending test EMAIL —")
_send_email(m, zone, machine, ts, test_img)

print("— Sending test SMS —")
_send_sms(m, zone, machine, ts)

print("\nDone. Check notifications_log.csv for results.")
