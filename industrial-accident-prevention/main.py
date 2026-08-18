"""
main.py — Application Entry Point
===================================
AI-Based Industrial Accident Prevention System
Using Real-Time Danger Zone Detection and Worker Tracking

This is the orchestrator that connects all modules into a single
real-time processing pipeline:

  ┌──────────────────────────────────────────────────────────────────┐
  │                    PROCESSING PIPELINE                          │
  │                                                                  │
  │  Video/Webcam                                                    │
  │      │                                                           │
  │      ▼                                                           │
  │  OpenCV frame capture  (cv2.VideoCapture)                        │
  │      │                                                           │
  │      ▼                                                           │
  │  YOLO person detection  (detector.py)                            │
  │      │  → bounding boxes, confidence, centroids                  │
  │      ▼                                                           │
  │  Worker ID tracking  (tracker.py)                                │
  │      │  → stable worker IDs across frames                        │
  │      ▼                                                           │
  │  Zone evaluation  (zones.py)                                     │
  │      │  → SAFE / WARNING / DANGER per worker                     │
  │      ▼                                                           │
  │  Alert generation  (alerts.py)                                   │
  │      │  → on-screen banners + audible beep                       │
  │      ▼                                                           │
  │  Visual display                                                  │
  │      → status panel, bounding boxes, zones, FPS                  │
  └──────────────────────────────────────────────────────────────────┘

Keyboard controls:
  Q — quit the application
  R — reset the tracker (clear all worker IDs)
  Z — toggle zone overlay visibility
  S — save a screenshot to the screenshots/ folder

All configurable parameters are in config.py.
"""

import cv2
import time
import os
import sys

# ── Import project modules ──
import config
import detector
import tracker
import zones
import alerts


# ──────────────────────────────────────────────
#  STATUS PANEL (HUD at the top-left of frame)
# ──────────────────────────────────────────────

def draw_status_panel(frame, fps, num_workers, num_warnings, num_dangers):
    """
    Draw a semi-transparent status panel in the top-left corner
    showing real-time system statistics.

    Layout:
      ┌─────────────────────────────────┐
      │  AI ACCIDENT PREVENTION SYSTEM  │
      │  FPS: 28.3                      │
      │  Workers: 3                     │
      │  Warnings: 1                    │
      │  Dangers: 0                     │
      │  [Q]uit [R]eset [Z]ones [S]ave  │
      └─────────────────────────────────┘

    Args:
        frame        : numpy.ndarray — BGR frame to draw on.
        fps          : float         — current frames per second.
        num_workers  : int           — number of currently tracked workers.
        num_warnings : int           — workers in WARNING zone.
        num_dangers  : int           — workers in DANGER zone.

    Returns:
        frame : numpy.ndarray — the annotated frame.
    """
    # Panel dimensions and position
    panel_w = 310
    panel_h = 170
    margin  = 10
    x1, y1  = margin, margin
    x2, y2  = x1 + panel_w, y1 + panel_h

    # ── Semi-transparent dark background ──
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # ── Border ──
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # ── Text content ──
    text_x = x1 + 12
    line_y = y1 + 25
    line_gap = 24

    # Title
    cv2.putText(frame, "AI ACCIDENT PREVENTION SYSTEM", (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
    line_y += line_gap + 4

    # FPS — colour-coded: green if ≥15, yellow if ≥8, red if <8
    fps_color = (0, 200, 0) if fps >= 15 else (0, 200, 255) if fps >= 8 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2)
    line_y += line_gap

    # Workers detected
    cv2.putText(frame, f"Workers Detected: {num_workers}", (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    line_y += line_gap

    # Active warnings
    warn_color = config.COLOR_WARNING if num_warnings > 0 else (150, 150, 150)
    cv2.putText(frame, f"Active Warnings: {num_warnings}", (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, warn_color, 1)
    line_y += line_gap

    # Active dangers
    dang_color = config.COLOR_DANGER if num_dangers > 0 else (150, 150, 150)
    cv2.putText(frame, f"Active Dangers:  {num_dangers}", (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, dang_color, 1)
    line_y += line_gap + 2

    # Keyboard controls hint
    cv2.putText(frame, "[Q]uit  [R]eset  [Z]ones  [S]ave", (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150, 150, 150), 1)

    return frame


# ──────────────────────────────────────────────
#  DRAW WORKER ANNOTATIONS
# ──────────────────────────────────────────────

def draw_worker(frame, worker, status):
    """
    Draw a single worker's bounding box, ID label, status, and centroid
    on the frame, colour-coded by their safety status.

    Args:
        frame  : numpy.ndarray — BGR frame to draw on.
        worker : dict          — from tracker.update(), has keys:
                                 worker_id, bbox, centroid, confidence
        status : str           — "SAFE", "WARNING", or "DANGER"

    Returns:
        frame : numpy.ndarray — the annotated frame.
    """
    x1, y1, x2, y2 = worker["bbox"]
    cx, cy          = worker["centroid"]
    wid             = worker["worker_id"]
    conf            = worker["confidence"]
    color           = zones.get_status_color(status)

    # ── Bounding box coloured by status ──
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

    # ── Label: "Worker 1 [SAFE] 0.87" ──
    label = f"Worker {wid} [{status}] {conf:.2f}"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE, config.FONT_THICKNESS
    )

    # Filled background behind label for readability.
    cv2.rectangle(
        frame,
        (x1, y1 - th - baseline - 8),
        (x1 + tw + 6, y1),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        frame, label,
        (x1 + 3, y1 - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE,
        (255, 255, 255),
        config.FONT_THICKNESS,
    )

    # ── Centroid dot ──
    cv2.circle(frame, (cx, cy), 5, color, -1)

    # ── Centroid coordinate text (small, below the dot) ──
    cv2.putText(
        frame, f"({cx},{cy})", (cx + 8, cy + 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1
    )

    return frame


# ──────────────────────────────────────────────
#  MAIN FUNCTION
# ──────────────────────────────────────────────

def main():
    """
    Main entry point.  Initialises all modules, opens the video
    source, and runs the real-time processing loop until the user
    presses 'Q' or the video ends.
    """

    # ═══════════════════════════════════════════
    #  STEP 1: STARTUP & INITIALISATION
    # ═══════════════════════════════════════════

    print("=" * 60)
    print("  AI-Based Industrial Accident Prevention System")
    print("  Real-Time Danger Zone Detection & Worker Tracking")
    print("=" * 60)

    # 1a. Load the YOLO model (downloads weights on first run).
    model = detector.load_model()

    # 1b. Create the centroid tracker (starts with zero tracks).
    worker_tracker = tracker.CentroidTracker()

    # 1c. Create the alert manager (manages cooldown timers).
    alert_manager = alerts.AlertManager()

    # 1d. Open the video source (webcam or file, set in config.py).
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"\n[ERROR] Cannot open video source: {config.VIDEO_SOURCE}")
        print("Possible fixes:")
        print("  • If using webcam: check that it is plugged in.")
        print("  • If using a file: check that the path in config.py is correct.")
        print("  • Try setting VIDEO_SOURCE = 0 in config.py for default webcam.")
        sys.exit(1)

    # Read video properties for logging.
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30  # fallback for webcam

    print(f"\n[main] Video source  : {config.VIDEO_SOURCE}")
    print(f"[main] Resolution    : {frame_width} x {frame_height}")
    print(f"[main] Source FPS    : {video_fps:.1f}")
    print(f"[main] YOLO model    : {config.YOLO_MODEL_PATH}")
    print(f"[main] Confidence    : {config.YOLO_CONFIDENCE_THRESHOLD}")
    print(f"[main] Tracking dist : {config.MAX_TRACKING_DISTANCE} px")
    print(f"[main] Max missing   : {config.MAX_FRAMES_MISSING} frames")
    print(f"[main] Alert cooldown: {config.ALERT_COOLDOWN} sec")
    print()
    print("[main] Controls:")
    print("         Q = Quit")
    print("         R = Reset tracker (clear all worker IDs)")
    print("         Z = Show / hide zone overlays")
    print("         S = Save screenshot")
    print()

    # ═══════════════════════════════════════════
    #  STEP 2: STATE VARIABLES
    # ═══════════════════════════════════════════

    frame_count     = 0           # total frames processed
    show_zones      = True        # toggle with 'Z' key
    fps             = 0.0         # computed each frame
    prev_time       = time.time() # for FPS calculation

    # Throttle terminal output: only print alerts every N frames
    # to avoid flooding the console at 30+ fps.
    TERMINAL_LOG_INTERVAL = 30

    # ═══════════════════════════════════════════
    #  STEP 3: MAIN PROCESSING LOOP
    # ═══════════════════════════════════════════

    while True:
        # ─────────────────────────────────────
        #  3a. Capture a frame
        # ─────────────────────────────────────
        ret, frame = cap.read()

        if not ret:
            # End of video file, or camera disconnected.
            print("\n[main] Video stream ended or camera disconnected.")
            # If it's a video file, we can loop or exit.
            # For MVP, we simply exit.
            break

        frame_count += 1

        # ─────────────────────────────────────
        #  3b. FPS calculation
        # ─────────────────────────────────────
        # Simple method: measure wall-clock time between frames.
        current_time = time.time()
        elapsed = current_time - prev_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        prev_time = current_time

        # ─────────────────────────────────────
        #  3c. YOLO person detection
        # ─────────────────────────────────────
        # detector.detect_persons() returns a list of dicts:
        #   { bbox, confidence, class_id, centroid }
        # Only "person" detections above the confidence threshold
        # are included.  The list may be empty (no people visible).
        detections = detector.detect_persons(model, frame)

        # ─────────────────────────────────────
        #  3d. Worker ID tracking
        # ─────────────────────────────────────
        # tracker.update() matches new detections to existing
        # tracks using centroid Euclidean distance, then returns:
        #   { worker_id, bbox, centroid, confidence }
        # Handles new IDs, missing workers, and stale track removal.
        tracked_workers = worker_tracker.update(detections)

        # ─────────────────────────────────────
        #  3e. Draw zone overlays (if enabled)
        # ─────────────────────────────────────
        # Zones are drawn FIRST so they appear BEHIND the workers.
        if show_zones:
            frame = zones.draw_zones(frame)

        # ─────────────────────────────────────
        #  3f. Classify each worker & trigger alerts
        # ─────────────────────────────────────
        workers_with_status = []
        num_warnings = 0
        num_dangers  = 0

        # Decide whether to print to terminal this frame.
        should_log = (frame_count % TERMINAL_LOG_INTERVAL == 0)

        for worker in tracked_workers:
            centroid = worker["centroid"]
            wid      = worker["worker_id"]

            # Check which zone the worker's centroid falls into.
            status = zones.get_worker_status(centroid)

            # Collect for alert banner and status panel.
            workers_with_status.append({
                "worker_id": wid,
                "status":    status,
            })

            # Count warnings and dangers for the status panel.
            if status == zones.STATUS_WARNING:
                num_warnings += 1
                # Trigger warning alert (terminal log only, no beep).
                # Throttled to avoid console flooding.
                if should_log:
                    alert_manager.warning_alert(wid)

            elif status == zones.STATUS_DANGER:
                num_dangers += 1
                # Trigger danger alert (terminal log + audible beep).
                # The beep is debounced by AlertManager's cooldown.
                # Terminal log is throttled here; beep fires independently.
                alert_manager.danger_alert(wid)

            # ── Draw this worker's bounding box, label, centroid ──
            frame = draw_worker(frame, worker, status)

        # ─────────────────────────────────────
        #  3g. Draw alert banners (top-right)
        # ─────────────────────────────────────
        frame = alert_manager.draw_alert_banner(frame, workers_with_status)

        # ─────────────────────────────────────
        #  3h. Draw status panel (top-left HUD)
        # ─────────────────────────────────────
        frame = draw_status_panel(
            frame, fps,
            num_workers  = len(tracked_workers),
            num_warnings = num_warnings,
            num_dangers  = num_dangers,
        )

        # ─────────────────────────────────────
        #  3i. Terminal logging (throttled)
        # ─────────────────────────────────────
        if should_log and len(tracked_workers) > 0:
            print(f"\n[Frame {frame_count:>6d}] FPS={fps:.1f}  "
                  f"Workers={len(tracked_workers)}  "
                  f"Warn={num_warnings}  Danger={num_dangers}")
            for w in tracked_workers:
                s = zones.get_worker_status(w["centroid"])
                print(f"  └─ Worker {w['worker_id']:>2d}  "
                      f"centroid={w['centroid']}  "
                      f"conf={w['confidence']:.2f}  "
                      f"status={s}")

        # ─────────────────────────────────────
        #  3j. Display the frame
        # ─────────────────────────────────────
        cv2.imshow(config.WINDOW_NAME, frame)

        # ─────────────────────────────────────
        #  3k. Keyboard input
        # ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            # ── QUIT ──
            print("\n[main] Quit by user (Q pressed).")
            break

        elif key == ord('r') or key == ord('R'):
            # ── RESET TRACKER ──
            # Clears all worker IDs.  New IDs will start from 1 again.
            worker_tracker = tracker.CentroidTracker()
            alert_manager = alerts.AlertManager()
            print("\n[main] Tracker RESET. All worker IDs cleared.")

        elif key == ord('z') or key == ord('Z'):
            # ── TOGGLE ZONES ──
            show_zones = not show_zones
            state = "ON" if show_zones else "OFF"
            print(f"\n[main] Zone overlay toggled: {state}")

        elif key == ord('s') or key == ord('S'):
            # ── SAVE SCREENSHOT ──
            os.makedirs("screenshots", exist_ok=True)
            filename = f"screenshots/frame_{frame_count}.png"
            cv2.imwrite(filename, frame)
            print(f"\n[main] Screenshot saved: {filename}")

    # ═══════════════════════════════════════════
    #  STEP 4: CLEANUP
    # ═══════════════════════════════════════════
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"  Session Summary")
    print(f"  Total frames processed : {frame_count}")
    print(f"  Total worker IDs used  : {worker_tracker.next_worker_id - 1}")
    print("=" * 60)
    print("[main] Application closed.\n")


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
