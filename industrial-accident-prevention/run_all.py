# -*- coding: utf-8 -*-
"""
run_all.py — Unified Master Launcher for AI-Based Industrial Accident Prevention System
======================================================================================
Launches all subsystems concurrently:
  1. Safety Analytics Dashboard       -> http://localhost:8501 (Streamlit)
  2. Employee Face Enrollment         -> http://localhost:8502 (Streamlit)
  3. Zone Manager Configuration       -> http://localhost:8503 (Streamlit)
  4. Core CV & Prevention System      -> http://localhost:5000 (Flask Admin + OpenCV Window)

Keyboard controls in OpenCV window:
  Q — quit the application & all dashboard services
  R — reset worker tracking
  Z — toggle zone overlay
  S — save screenshot
  K — toggle skeleton drawing
  M — mute / unmute audio alerts
"""

import os
import sys
import time
import signal
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

# Ensure UTF-8 on Windows so Streamlit and DeepFace logs do not crash on charmap
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

BANNER = r"""
===========================================================================
    AI-BASED INDUSTRIAL ACCIDENT PREVENTION SYSTEM -- MASTER LAUNCHER
===========================================================================
  [1] Admin Panel & Live CCTV Feed   : http://localhost:5000
  [2] Safety Analytics & PDF Reports : http://localhost:8501
  [3] Employee Face Enrollment       : http://localhost:8502
  [4] Zone Manager Configuration     : http://localhost:8503
  [5] Real-Time Vision Pipeline      : Active (YOLO Pose + Tracking + Zones)
===========================================================================
"""

def main():
    print(BANNER)
    processes = []

    def cleanup():
        print("\n[Launcher] Stopping all dashboard background processes...")
        for name, p in processes:
            try:
                print(f"  Stopping {name} (PID {p.pid})...")
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        print("[Launcher] All services stopped cleanly.")

    # 1. Start Streamlit Dashboard (Port 8501)
    print("[1/4] Starting Safety Analytics Dashboard on port 8501...")
    p1 = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "dashboard.py",
         "--server.port", "8501", "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT_DIR)
    )
    processes.append(("Analytics Dashboard", p1))

    # 2. Start Employee Face Enrollment Dashboard (Port 8502)
    print("[2/4] Starting Employee Face Enrollment Dashboard on port 8502...")
    p2 = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "enroll_dashboard.py",
         "--server.port", "8502", "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT_DIR)
    )
    processes.append(("Enroll Dashboard", p2))

    # 3. Start Zone Manager Configuration Dashboard (Port 8503)
    print("[3/4] Starting Zone Manager Configuration Dashboard on port 8503...")
    p3 = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "manager_dashboard.py",
         "--server.port", "8503", "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        cwd=str(ROOT_DIR)
    )
    processes.append(("Manager Dashboard", p3))

    time.sleep(2)
    print("[4/4] Starting Core CV Pipeline & Admin Web Dashboard (Port 5000)...")
    print("\n>>> All web dashboards are LIVE. Starting camera stream now...\n")

    try:
        # Import and run main pipeline in foreground
        import main as cv_main
        cv_main.main()
    except KeyboardInterrupt:
        print("\n[Launcher] KeyboardInterrupt received.")
    finally:
        cleanup()

if __name__ == "__main__":
    main()
