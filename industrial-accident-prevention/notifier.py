# -*- coding: utf-8 -*-
"""notifier.py — Email + SMS notifications for DANGER events."""

import csv
import json
import os
import re
import smtplib
import threading
import time
from datetime import datetime
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root so credentials are always available
_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")
MANAGERS_FILE = str(_ROOT / "managers.json")
NOTIFY_LOG    = str(_ROOT / "notifications_log.csv")
COOLDOWN_SECS = int(os.environ.get("NOTIFIER_COOLDOWN", "600"))  # 10 minutes per zone

# In-memory cooldown tracker: zone_id -> last notification epoch
_last_managers_mtime: float = 0.0
_last_notified: dict[str, float] = {}
_last_notified_email: dict[str, str] = {}
_lock = threading.Lock()


# ── manager lookup ──

def _load_managers() -> list[dict]:
    if not os.path.exists(MANAGERS_FILE):
        return []
    try:
        with open(MANAGERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[notifier] Error reading {MANAGERS_FILE}: {e}")
        return []


def get_manager(zone_id: str) -> dict | None:
    managers = _load_managers()
    if not managers:
        return None
    # 1. Exact match
    for m in managers:
        if m.get("zone_id") == zone_id:
            return m
    # 2. Match with or without "zone_" prefix
    clean_id = str(zone_id).replace("zone_", "").strip()
    for m in managers:
        m_zid = str(m.get("zone_id", "")).replace("zone_", "").strip()
        if m_zid and m_zid == clean_id:
            return m
    # 3. Fallback to first manager with real email
    for m in managers:
        if m.get("email") and "example.com" not in m.get("email", ""):
            return m
    return managers[0]


# ── logging ──

def _log_notification(success: bool, channel: str, zone_id: str, detail: str):
    os.makedirs(os.path.dirname(NOTIFY_LOG) or ".", exist_ok=True)
    write_header = not os.path.exists(NOTIFY_LOG)
    with open(NOTIFY_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "channel", "zone_id", "success", "detail"])
        w.writerow([datetime.now().isoformat(), channel, zone_id,
                     "OK" if success else "FAIL", detail])
        f.flush()


# ── email ──

def _send_email(manager: dict, zone_id: str, machine: str, ts: str, img_path: str):
    load_dotenv(_ROOT / ".env", override=True)
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
    port = int(os.environ.get("SMTP_PORT", "465"))
    user = os.environ.get("SMTP_USER", "")
    pwd  = os.environ.get("SMTP_PASSWORD", "")
    if not user or not pwd:
        _log_notification(False, "email", zone_id, "SMTP_USER or SMTP_PASSWORD not set")
        print("[notifier] WARN: SMTP credentials not configured, skipping email")
        return
    # If img_path is empty or missing, find the most recent incident image in alerts
    if not img_path or not os.path.exists(img_path):
        alerts_dir = _ROOT / "alerts"
        if alerts_dir.exists():
            candidates = sorted(
                [p for p in alerts_dir.glob("*.jpg") if p.name != "test_notify.jpg"],
                key=os.path.getmtime,
                reverse=True
            )
            if candidates:
                img_path = str(candidates[0])

    to_addr = manager["email"]
    subject = f"DANGER ALERT - {machine} - {zone_id}"

    msg = MIMEMultipart("related")
    msg["From"]    = user
    msg["To"]      = to_addr
    msg["Subject"] = subject

    alt = MIMEMultipart("alternative")
    msg.attach(alt)

    text_body = (
        f"DANGER ALERT\n\n"
        f"Timestamp   : {ts}\n"
        f"Zone        : {zone_id}\n"
        f"Machine     : {machine}\n"
        f"Manager     : {manager.get('manager_name', '')}\n\n"
        f"A person has been detected in the restricted danger zone.\n"
        f"Please stop the machine if the situation is unsafe."
    )
    alt.attach(MIMEText(text_body, "plain"))

    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #222;">
        <div style="background-color: #d9534f; color: white; padding: 12px 16px; border-radius: 4px; font-size: 18px; font-weight: bold;">
          🚨 INDUSTRIAL SAFETY ALERT — DANGER ZONE BREACH
        </div>
        <div style="padding: 16px; border: 1px solid #ddd; border-radius: 4px; margin-top: 10px; background-color: #fafafa;">
          <p><strong>Timestamp:</strong> {ts}</p>
          <p><strong>Zone:</strong> {zone_id}</p>
          <p><strong>Machine:</strong> {machine}</p>
          <p><strong>Manager:</strong> {manager.get('manager_name', '')}</p>
          <p style="color: #d9534f; font-size: 15px; font-weight: bold;">
            ⚠️ A person has entered the restricted safety zone!
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

    # attach incident image
    if img_path and os.path.exists(img_path):
        try:
            with open(img_path, "rb") as fp:
                part = MIMEBase("image", "jpeg")
                part.set_payload(fp.read())
                encoders.encode_base64(part)
                part.add_header("Content-Disposition", "attachment",
                                filename=os.path.basename(img_path))
                part.add_header("Content-ID", "<danger_snapshot>")
                msg.attach(part)
        except Exception as e:
            print(f"[notifier] Warning: Failed to attach image: {e}")

    sent = False
    last_err = None
    # Try port 465 (SSL) and port 587 (STARTTLS) with automatic fallback
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
        _log_notification(True, "email", zone_id, f"sent to {to_addr}")
        print(f"[notifier] Email sent successfully to {to_addr}")
    else:
        _log_notification(False, "email", zone_id, str(last_err))
        print(f"[notifier] Email FAILED to {to_addr}: {last_err}")


# ── SMS via Twilio ──

def _send_sms(manager: dict, zone_id: str, machine: str, ts: str):
    load_dotenv(_ROOT / ".env", override=True)
    sid   = os.environ.get("TWILIO_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_num = os.environ.get("TWILIO_FROM_NUMBER", "")
    if not sid or not token or not from_num:
        _log_notification(False, "sms", zone_id, "Twilio env vars not set")
        print("[notifier] WARN: Twilio not configured, skipping SMS")
        return
    to_num = manager.get("phone", "").strip()
    if not to_num:
        return
    body = (f"DANGER: person detected near {machine} ({zone_id}) "
            f"at {ts}. Stop machine if unsafe.")
    try:
        from twilio.rest import Client
        client = Client(sid, token)
        client.messages.create(body=body, from_=from_num, to=to_num)
        _log_notification(True, "sms", zone_id, f"sent to {to_num}")
        print(f"[notifier] SMS sent to {to_num}")
    except ImportError:
        _log_notification(False, "sms", zone_id, "twilio package not installed")
        print("[notifier] WARN: twilio package not installed, skipping SMS")
    except Exception as exc:
        _log_notification(False, "sms", zone_id, str(exc))
        print(f"[notifier] SMS FAILED: {exc}")


# ── public entry point ──

def notify_manager(zone_id: str, img_path: str = ""):
    """Look up manager for zone_id, send email + SMS in background thread.
    Automatically detects new or updated email IDs in managers.json without restarting."""
    global _last_managers_mtime

    # Check if managers.json was updated on disk
    curr_mtime = 0.0
    if os.path.exists(MANAGERS_FILE):
        try:
            curr_mtime = os.path.getmtime(MANAGERS_FILE)
        except OSError:
            pass

    with _lock:
        if curr_mtime > 0 and curr_mtime > _last_managers_mtime:
            print("[notifier] managers.json modified on disk. Reloading managers & resetting cooldowns.")
            _last_managers_mtime = curr_mtime
            _last_notified.clear()
            _last_notified_email.clear()

    manager = get_manager(zone_id)
    if not manager:
        print(f"[notifier] No manager configured for zone '{zone_id}', skipping")
        return

    to_email = str(manager.get("email", "")).strip().lower()
    if not to_email:
        print(f"[notifier] Manager for zone '{zone_id}' has no email configured, skipping")
        return

    norm_zone = str(zone_id).replace("zone_", "").strip() or "default"
    curr_time = time.time()

    with _lock:
        last_time = _last_notified.get(norm_zone, 0.0)
        last_email = _last_notified_email.get(norm_zone, "")

        # If a new email was added or changed for this zone, immediately bypass cooldown!
        if last_email != to_email:
            print(f"[notifier] New/updated recipient '{to_email}' for zone '{zone_id}' — sending immediately!")
        elif curr_time - last_time < COOLDOWN_SECS:
            # Same recipient within cooldown period
            return

        _last_notified[norm_zone] = curr_time
        _last_notified_email[norm_zone] = to_email

    machine = manager.get("machine_name", zone_id)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _do():
        try:
            _send_email(manager, zone_id, machine, ts, img_path)
            _send_sms(manager, zone_id, machine, ts)
        except Exception as e:
            print(f"[notifier] Background thread error: {e}")

    t = threading.Thread(target=_do, daemon=True)
    t.start()

