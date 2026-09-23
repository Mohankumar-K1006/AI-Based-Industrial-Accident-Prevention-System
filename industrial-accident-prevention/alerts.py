"""
alerts.py — Alert Generation Module
====================================
Responsibilities:
  1. Display on-screen warning/danger messages on the video frame.
  2. Print alert messages to the terminal (console log).
  3. Play an audible beep when a worker enters the DANGER zone.
  4. Debounce alerts using a cooldown timer so the system does NOT
     fire hundreds of beeps per second (one beep every N seconds).

Debounce / cooldown strategy:
  ┌────────────────────────────────────────────────────────────────┐
  │  Each worker ID has its own cooldown timer.                    │
  │                                                                │
  │  When danger_alert(worker_id) is called:                       │
  │    1. Check: has this worker's last beep been > COOLDOWN ago?  │
  │    2. YES → play beep in a background thread, reset timer.     │
  │    3. NO  → skip the beep (still show on-screen text).         │
  │                                                                │
  │  The beep runs in a BACKGROUND THREAD so it does not block     │
  │  the video processing loop.  Only ONE beep thread can be       │
  │  active at a time (guarded by a flag).                         │
  └────────────────────────────────────────────────────────────────┘

Windows audio:
  Uses the built-in `winsound` module (no pip install needed).
  winsound.Beep(frequency_hz, duration_ms) plays a simple tone.
  On non-Windows systems, the beep is skipped gracefully.

This module does NOT modify detector.py, tracker.py, or zones.py.
"""

import time
import threading
import cv2
import config

# ── Try to import winsound (Windows only) ──
# On Linux/Mac this will fail, so we set a flag and skip beeps.
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False
    print("[alerts] winsound not available (non-Windows OS). "
          "Audible beeps will be skipped.")


class AlertManager:
    """
    Manages audible and visual alerts with per-worker cooldown.

    Usage:
        alert_mgr = AlertManager()

        # In the frame loop, for each worker:
        if status == "WARNING":
            alert_mgr.warning_alert(worker_id)
        elif status == "DANGER":
            alert_mgr.danger_alert(worker_id)

        # Draw all alert banners on the frame:
        frame = alert_mgr.draw_alert_banner(frame, workers_with_status)
    """

    def __init__(self):
        """Initialise the alert manager with empty cooldown state."""

        # Tracks the last time a beep was triggered for each worker.
        # Key = worker_id (int), Value = timestamp (float).
        self._last_beep_time = {}
        self._last_warning_time = {}
        self._last_global_beep_time = 0.0

        # Flag to prevent overlapping beep threads.
        # Only one beep can play at a time.
        self._beep_playing = False

    # ──────────────────────────────────────────
    #  WARNING ALERT
    # ──────────────────────────────────────────

    def warning_alert(self, worker_id):
        """
        Handle a WARNING-level alert for a worker.

        Actions:
          - Print a warning message to the terminal (throttled).
          - No audible beep (warnings are visual/log only).

        Args:
            worker_id : int — the tracked worker's ID.
        """
        current_time = time.time()
        last_time = self._last_warning_time.get(worker_id, 0)
        if current_time - last_time >= config.ALERT_COOLDOWN:
            self._last_warning_time[worker_id] = current_time
            print(f"  ⚠  WARNING: Worker {worker_id} approaching danger zone")

    # ──────────────────────────────────────────
    #  DANGER ALERT
    # ──────────────────────────────────────────

    def danger_alert(self, worker_id):
        """
        Handle a DANGER-level alert for a worker.

        Actions:
          - Print a danger message to the terminal.
          - Play an audible beep IF the cooldown period has elapsed
            for this worker and globally.

        Args:
            worker_id : int — the tracked worker's ID.
        """
        current_time = time.time()
        last_worker_time = self._last_beep_time.get(worker_id, 0)

        # Check if enough time has passed both per-worker and globally
        if (current_time - last_worker_time >= config.ALERT_COOLDOWN and
                current_time - self._last_global_beep_time >= 1.5):
            self._last_beep_time[worker_id] = current_time
            self._last_global_beep_time = current_time
            print(f"  🚨 DANGER: Worker {worker_id} entered restricted zone!")
            self._play_beep()

    # ──────────────────────────────────────────
    #  AUDIBLE BEEP (runs in background thread)
    # ──────────────────────────────────────────

    def _play_beep(self):
        """
        Play a short beep sound in a background thread.

        Why a thread?
          winsound.Beep() is a BLOCKING call — it pauses execution
          for the beep's duration (e.g. 200 ms).  If we called it
          directly in the main loop, the video would freeze for
          200 ms each time.  Running it in a thread keeps the
          video smooth.

        Guard:
          The _beep_playing flag prevents stacking multiple beep
          threads on top of each other.
        """
        if not HAS_WINSOUND:
            return

        if self._beep_playing:
            return    # a beep is already in progress

        def _beep_thread():
            self._beep_playing = True
            try:
                winsound.Beep(
                    config.ALERT_BEEP_FREQUENCY,   # Hz  (e.g. 1000)
                    config.ALERT_BEEP_DURATION      # ms  (e.g. 200)
                )
            except Exception:
                pass    # fail silently if audio device is unavailable
            finally:
                self._beep_playing = False

        # Start a daemon thread so it dies when the main program exits.
        thread = threading.Thread(target=_beep_thread, daemon=True)
        thread.start()

    # ──────────────────────────────────────────
    #  ON-SCREEN ALERT BANNER
    # ──────────────────────────────────────────

    def draw_alert_banner(self, frame, workers_with_status):
        """
        Draw alert messages at the top of the frame for any worker
        that is in WARNING or DANGER status.

        Args:
            frame               : numpy.ndarray — BGR frame to draw on.
            workers_with_status : list[dict]     — each dict must have:
                "worker_id" : int
                "status"    : str ("SAFE", "WARNING", or "DANGER")

        Returns:
            frame : numpy.ndarray — the annotated frame.

        Layout:
          Alert messages stack vertically starting from the top-right
          corner of the frame:
            ┌──────────────────────────────────────────┐
            │                    ⚠ WARNING: Worker 2.. │
            │                    🚨 DANGER: Worker 1.. │
            │                                          │
            │         (video feed)                     │
            └──────────────────────────────────────────┘
        """
        frame_h, frame_w = frame.shape[:2]

        # Collect alert messages (skip SAFE workers).
        messages = []
        for w in workers_with_status:
            wid    = w["worker_id"]
            status = w["status"]

            if status == "WARNING":
                messages.append({
                    "text":  f"WARNING: Worker {wid} approaching danger zone",
                    "color": config.COLOR_WARNING,
                })
            elif status == "DANGER":
                messages.append({
                    "text":  f"DANGER: Worker {wid} entered restricted zone!",
                    "color": config.COLOR_DANGER,
                })

        if not messages:
            return frame

        # Draw each message as a banner row from top-right corner.
        y_offset = 30
        for msg in messages:
            text  = msg["text"]
            color = msg["color"]

            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, config.FONT_THICKNESS
            )

            # Position: right-aligned with a small margin.
            x_pos = frame_w - tw - 15

            # Background rectangle for readability.
            cv2.rectangle(
                frame,
                (x_pos - 5, y_offset - th - 5),
                (frame_w - 5, y_offset + baseline + 5),
                color,
                cv2.FILLED,
            )

            # White text on coloured background.
            cv2.putText(
                frame, text,
                (x_pos, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE,
                (255, 255, 255),
                config.FONT_THICKNESS,
            )

            # Move down for the next message.
            y_offset += th + baseline + 15

        return frame


# ──────────────────────────────────────────────
#  CONVENIENCE FUNCTIONS (stateless wrappers)
# ──────────────────────────────────────────────
# These allow simple usage without creating an AlertManager instance.
# They use a module-level default instance.

_default_manager = AlertManager()


def warning_alert(worker_id):
    """Log a warning alert for the given worker. No audible beep."""
    _default_manager.warning_alert(worker_id)


def danger_alert(worker_id):
    """Log a danger alert and play an audible beep (with cooldown)."""
    _default_manager.danger_alert(worker_id)


def draw_alert_banner(frame, workers_with_status):
    """Draw on-screen alert banners. Returns the annotated frame."""
    return _default_manager.draw_alert_banner(frame, workers_with_status)


# ──────────────────────────────────────────────
#  SELF-TEST  (run:  python alerts.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import time as _time
    import os
    import detector
    import tracker as tracker_module
    import zones

    print("=" * 55)
    print("  alerts.py — Self-Test  (Full Pipeline)")
    print("=" * 55)

    # Load model + tracker + alert manager
    model = detector.load_model()
    trk = tracker_module.CentroidTracker()
    alert_mgr = AlertManager()

    # Open video
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {config.VIDEO_SOURCE}")
        exit(1)

    print(f"[alerts] Video opened: {config.VIDEO_SOURCE}")
    print(f"[alerts] Beep frequency : {config.ALERT_BEEP_FREQUENCY} Hz")
    print(f"[alerts] Beep duration  : {config.ALERT_BEEP_DURATION} ms")
    print(f"[alerts] Alert cooldown : {config.ALERT_COOLDOWN} sec")
    print("[alerts] Press 'q' to quit.\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[alerts] End of video stream.")
            break

        frame_count += 1
        start_time = _time.time()

        # Pipeline: detect → track → classify → alert
        detections = detector.detect_persons(model, frame)
        tracked_workers = trk.update(detections)

        # Draw zones
        frame = zones.draw_zones(frame)

        # Classify each worker and trigger alerts
        workers_with_status = []
        for worker in tracked_workers:
            cx, cy = worker["centroid"]
            status = zones.get_worker_status((cx, cy))
            color  = zones.get_status_color(status)
            wid    = worker["worker_id"]

            workers_with_status.append({
                "worker_id": wid,
                "status":    status,
            })

            # Trigger appropriate alert
            if status == "WARNING":
                alert_mgr.warning_alert(wid)
            elif status == "DANGER":
                alert_mgr.danger_alert(wid)

            # Draw bounding box coloured by status
            x1, y1, x2, y2 = worker["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

            # Label
            label = f"Worker {wid} [{status}] {worker['confidence']:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, config.FONT_THICKNESS
            )
            cv2.rectangle(frame, (x1, y1 - th - baseline - 6),
                          (x1 + tw + 4, y1), color, cv2.FILLED)
            cv2.putText(frame, label, (x1 + 2, y1 - baseline - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                        (255, 255, 255), config.FONT_THICKNESS)

            # Centroid
            cv2.circle(frame, (cx, cy), 5, color, -1)

        # Draw alert banners
        frame = alert_mgr.draw_alert_banner(frame, workers_with_status)

        elapsed = _time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # HUD
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(frame, f"Tracked: {len(tracked_workers)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        cv2.imshow("Alerts Self-Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[alerts] Quit by user.")
            break
        elif key == ord('s'):
            os.makedirs("screenshots", exist_ok=True)
            fname = f"screenshots/alerts_frame_{frame_count}.png"
            cv2.imwrite(fname, frame)
            print(f"[alerts] Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[alerts] Self-test finished.")
