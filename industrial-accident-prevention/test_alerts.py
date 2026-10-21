"""
test_alerts.py — exercises alert_system without camera or YOLO.
Cycles SAFE -> WARNING -> DANGER -> SAFE on a blank frame.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import time
import alert_system

PHASES = [
    ("SAFE",    3.0),
    ("WARNING", 5.0),
    ("DANGER",  7.0),
    ("SAFE",    5.0),
]

mgr = alert_system.AlertManager()
frame_no = 0
print("Running test. Press Q to quit early.")

try:
    for state_name, duration in PHASES:
        print(f"[test] Phase: {state_name} for {duration}s")
        t0 = time.time()
        workers = []
        if state_name == "DANGER":
            workers = [{"status": "DANGER", "bbox": (150, 100, 350, 350)}]
        elif state_name == "WARNING":
            workers = [{"status": "WARNING", "bbox": (150, 100, 350, 350)}]

        while time.time() - t0 < duration:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame_no += 1
            frame = mgr.update(state_name, frame, frame_no, workers)
            cv2.imshow("Alert System Test", frame)
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q") or key == ord("Q"):
                raise KeyboardInterrupt
            if key == ord("m") or key == ord("M"):
                mgr.toggle_mute()

    print("[test] DONE. All phases complete.")
except KeyboardInterrupt:
    print("[test] Stopped by user.")
finally:
    mgr.shutdown()
    cv2.destroyAllWindows()
