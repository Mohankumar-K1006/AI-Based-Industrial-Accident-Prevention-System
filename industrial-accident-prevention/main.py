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
  │  YOLO pose detection  (detector.py)                              │
  │      │  → bounding boxes, confidence, 17 keypoints, edges        │
  │      ▼                                                           │
  │  YOLO vehicle detection (detector.py)                            │
  │      │  → vehicle bounding boxes, person-in-vehicle matching     │
  │      ▼                                                           │
  │  Worker ID tracking  (tracker.py)                                │
  │      │  → stable worker IDs with keypoints across frames         │
  │      ▼                                                           │
  │  Multi-point zone evaluation  (zones.py)                         │
  │      │  → SAFE / WARNING / DANGER per worker (any body part)     │
  │      ▼                                                           │
  │  Alert generation  (alerts.py)                                   │
  │      │  → on-screen banners + audible beep                       │
  │      ▼                                                           │
  │  Shared state push  (shared_state.py)                            │
  │      │  → data available to Flask admin dashboard                │
  │      ▼                                                           │
  │  Visual display + Admin Dashboard                                │
  │      → CV window + web dashboard at localhost:5000               │
  └──────────────────────────────────────────────────────────────────┘

Keyboard controls:
  Q — quit the application
  R — reset the tracker (clear all worker IDs)
  Z — toggle zone overlay visibility
  S — save a screenshot to the screenshots/ folder
  K — toggle skeleton drawing

All configurable parameters are in config.py.
Admin dashboard runs at http://localhost:5000 (set in config.py).
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
import alert_system
from shared_state import state


# ──────────────────────────────────────────────
#  STATUS PANEL (HUD at the top-left of frame)
# ──────────────────────────────────────────────

def draw_status_panel(frame, fps, num_workers, num_warnings, num_dangers,
                      num_vehicles=0, admin_url=""):
    """
    Draw a semi-transparent status panel in the top-left corner
    showing real-time system statistics.

    Layout:
      ┌─────────────────────────────────────┐
      │  AI ACCIDENT PREVENTION SYSTEM      │
      │  FPS: 28.3                          │
      │  Workers: 3  Vehicles: 1            │
      │  Warnings: 1   Dangers: 0           │
      │  Admin: http://localhost:5000        │
      │  [Q]uit [R]eset [Z]ones [S]ave [K]  │
      └─────────────────────────────────────┘

    Args:
        frame        : numpy.ndarray
        fps          : float
        num_workers  : int
        num_warnings : int
        num_dangers  : int
        num_vehicles : int
        admin_url    : str

    Returns:
        frame : numpy.ndarray — the annotated frame.
    """
    panel_w = 340
    panel_h = 200
    margin  = 10
    x1, y1  = margin, margin
    x2, y2  = x1 + panel_w, y1 + panel_h

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # Text content
    text_x = x1 + 12
    line_y = y1 + 25
    line_gap = 24

    # Title
    cv2.putText(frame, "AI ACCIDENT PREVENTION SYSTEM", (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
    line_y += line_gap + 4

    # FPS — colour-coded
    fps_color = (0, 200, 0) if fps >= 15 else (0, 200, 255) if fps >= 8 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2)
    line_y += line_gap

    # Workers + Vehicles
    cv2.putText(frame, f"Workers: {num_workers}   Vehicles: {num_vehicles}",
                (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    line_y += line_gap

    # Active warnings
    warn_color = config.COLOR_WARNING if num_warnings > 0 else (150, 150, 150)
    cv2.putText(frame, f"Warnings: {num_warnings}", (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, warn_color, 1)

    # Active dangers (same line)
    dang_color = config.COLOR_DANGER if num_dangers > 0 else (150, 150, 150)
    cv2.putText(frame, f"Dangers: {num_dangers}", (text_x + 160, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, dang_color, 1)
    line_y += line_gap

    # Admin URL
    if admin_url:
        cv2.putText(frame, f"Admin: {admin_url}", (text_x, line_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 200, 255), 1)
    line_y += line_gap

    # Keyboard controls hint
    cv2.putText(frame, "[Q]uit [R]eset [Z]ones [S]ave [K]skel",
                (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150, 150, 150), 1)

    return frame


# ──────────────────────────────────────────────
#  DRAW WORKER ANNOTATIONS
# ──────────────────────────────────────────────

def draw_worker(frame, worker, status):
    """
    Draw a single worker's bounding box, ID label, status, skeleton,
    and edge points on the frame, colour-coded by their safety status.

    Args:
        frame  : numpy.ndarray
        worker : dict from tracker.update()
        status : str — "SAFE", "WARNING", or "DANGER"

    Returns:
        frame : numpy.ndarray — the annotated frame.
    """
    x1, y1, x2, y2 = worker["bbox"]
    cx, cy          = worker["centroid"]
    wid             = worker["worker_id"]
    conf            = worker["confidence"]
    in_vehicle      = worker.get("in_vehicle", False)
    vehicle_info    = worker.get("vehicle_info")
    keypoints       = worker.get("keypoints", [])
    edge_points     = worker.get("edge_points", [])
    color           = zones.get_status_color(status)

    # Bounding box coloured by status
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

    # Label: "Worker 1 [SAFE] 0.87" or "Driver 1 [SAFE] 0.87 (Car)"
    label_prefix = "Driver" if in_vehicle else "Worker"
    label = f"{label_prefix} {wid} [{status}] {conf:.2f}"
    if in_vehicle and vehicle_info:
        label += f" ({vehicle_info['class_name']})"

    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX,
        config.FONT_SCALE, config.FONT_THICKNESS
    )

    # Filled background behind label
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

    # Draw skeleton if keypoints are available
    if keypoints:
        detector.draw_skeleton(frame, keypoints, color)

    # Centroid dot
    cv2.circle(frame, (cx, cy), 5, color, -1)

    # Edge point dots (small yellow circles)
    for (ex, ey) in edge_points:
        cv2.circle(frame, (ex, ey), 3, (0, 255, 255), -1)

    # Centroid coordinate text (small, below the dot)
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
    source, starts the admin dashboard, and runs the real-time
    processing loop until the user presses 'Q' or the video ends.
    """

    # ═══════════════════════════════════════════
    #  STEP 1: STARTUP & INITIALISATION
    # ═══════════════════════════════════════════

    print("=" * 60)
    print("  AI-Based Industrial Accident Prevention System")
    print("  Real-Time Danger Zone Detection & Worker Tracking")
    print("  Multi-Point Skeleton + Vehicle Detection + Admin")
    print("=" * 60)

    # 1a. Load the YOLO pose model (person detection with keypoints).
    pose_model = detector.load_model()

    # 1b. Load the YOLO vehicle model (car, bus, truck detection).
    vehicle_model = detector.load_vehicle_model()

    # 1c. Create the centroid tracker.
    worker_tracker = tracker.CentroidTracker()

    # 1d. Create the alert managers.
    alert_manager = alerts.AlertManager()
    adv_alerts = alert_system.AlertManager()

    # 1e. Load zone configurations (from file or config defaults).
    zones.load_zones()
    active_zones = zones.get_active_zones()
    print(f"\n[main] Loaded {len(active_zones)} safety zones:")
    for z in active_zones:
        print(f"       └─ {z['type']}: {z['name']} "
              f"({len(z['points'])} vertices)")

    # 1f. Start the admin dashboard (Flask) in a background thread.
    admin_url = f"http://localhost:{config.ADMIN_PORT}"
    if config.ADMIN_ENABLED:
        import admin_app
        admin_app.start_admin_server()
    else:
        print("\n[main] Admin dashboard is DISABLED (config.ADMIN_ENABLED = False)")
        admin_url = ""

    # 1g. Open the video source.
    if isinstance(config.VIDEO_SOURCE, int) and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(config.VIDEO_SOURCE, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(config.VIDEO_SOURCE)

    if not cap.isOpened() and isinstance(config.VIDEO_SOURCE, int):
        cap = cv2.VideoCapture(config.VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"\n[ERROR] Cannot open video source: {config.VIDEO_SOURCE}")
        print("Possible fixes:")
        print("  • If using webcam: check that it is plugged in.")
        print("  • If using a file: check that the path in config.py is correct.")
        print("  • Try setting VIDEO_SOURCE = 0 in config.py for default webcam.")
        sys.exit(1)

    # Read video properties
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30

    print(f"\n[main] Video source  : {config.VIDEO_SOURCE}")
    print(f"[main] Resolution    : {frame_width} x {frame_height}")
    print(f"[main] Source FPS    : {video_fps:.1f}")
    print(f"[main] Pose model    : {config.YOLO_POSE_MODEL_PATH}")
    print(f"[main] Vehicle model : {config.YOLO_VEHICLE_MODEL_PATH}")
    print(f"[main] Confidence    : {config.YOLO_CONFIDENCE_THRESHOLD}")
    print(f"[main] Tracking dist : {config.MAX_TRACKING_DISTANCE} px")
    print(f"[main] Max missing   : {config.MAX_FRAMES_MISSING} frames")
    print(f"[main] Alert cooldown: {config.ALERT_COOLDOWN} sec")
    print(f"[main] Skeleton draw : {config.SKELETON_DRAW}")
    print(f"[main] Vehicle detect: {config.DETECT_VEHICLES}")
    if admin_url:
        print(f"[main] Admin panel   : {admin_url}")
    print()
    print("[main] Controls:")
    print("         Q = Quit")
    print("         R = Reset tracker (clear all worker IDs)")
    print("         Z = Show / hide zone overlays")
    print("         S = Save screenshot")
    print("         K = Toggle skeleton drawing")
    print()

    # ═══════════════════════════════════════════
    #  STEP 2: STATE VARIABLES
    # ═══════════════════════════════════════════

    frame_count     = 0
    show_zones      = True
    fps             = 0.0
    prev_time       = time.time()

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
            print("\n[main] Video stream ended or camera disconnected.")
            break

        frame_count += 1

        # ─────────────────────────────────────
        #  3b. FPS calculation
        # ─────────────────────────────────────
        current_time = time.time()
        elapsed = current_time - prev_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0
        prev_time = current_time

        # ─────────────────────────────────────
        #  3c. YOLO pose person detection
        # ─────────────────────────────────────
        detections = detector.detect_persons(pose_model, frame)

        # ─────────────────────────────────────
        #  3d. Vehicle detection & matching
        # ─────────────────────────────────────
        vehicles = detector.detect_vehicles(vehicle_model, frame)
        detections = detector.detect_persons_in_vehicles(detections, vehicles)

        # ─────────────────────────────────────
        #  3e. Worker ID tracking
        # ─────────────────────────────────────
        tracked_workers = worker_tracker.update(detections)

        # ─────────────────────────────────────
        #  3f. Draw zone overlays (if enabled)
        # ─────────────────────────────────────
        if show_zones:
            frame = zones.draw_zones(frame)

        # ─────────────────────────────────────
        #  3g. Classify each worker & trigger alerts
        # ─────────────────────────────────────
        workers_with_status = []
        num_warnings = 0
        num_dangers  = 0

        should_log = (frame_count % TERMINAL_LOG_INTERVAL == 0)

        for worker in tracked_workers:
            centroid = worker["centroid"]
            wid      = worker["worker_id"]
            edge_pts = worker.get("edge_points", [centroid])

            # Multi-point zone check: any body edge in a zone triggers it
            status = zones.get_worker_status_multi(edge_pts)
            breached = zones.get_breached_danger_zone(edge_pts) if status == zones.STATUS_DANGER else None
            zid = breached["id"] if breached else "default"

            # Build enriched worker data for shared state
            worker_data = {
                "worker_id": wid,
                "status":    status,
                "centroid":  centroid,
                "bbox":      worker["bbox"],
                "confidence": worker["confidence"],
                "edge_points": edge_pts,
                "in_vehicle": worker.get("in_vehicle", False),
                "vehicle_info": worker.get("vehicle_info"),
                "zone_id":   zid,
            }
            workers_with_status.append(worker_data)

            # Count warnings and dangers
            if status == zones.STATUS_WARNING:
                num_warnings += 1
                if should_log:
                    alert_manager.warning_alert(wid)
                # Push to shared state alert log
                state.add_alert(wid, "WARNING",
                                f"Worker {wid} approaching danger zone")

            elif status == zones.STATUS_DANGER:
                num_dangers += 1
                alert_manager.danger_alert(wid)
                state.add_alert(wid, "DANGER",
                                f"Worker {wid} entered restricted zone!")

            # Draw this worker's bounding box, skeleton, label, centroid
            frame = draw_worker(frame, worker, status)

        # ─────────────────────────────────────
        #  3h. Draw vehicle bounding boxes
        # ─────────────────────────────────────
        frame = detector.draw_vehicles(frame, vehicles)

        # ─────────────────────────────────────
        #  3i. Draw alert banners + advanced alerts
        # ─────────────────────────────────────
        frame = alert_manager.draw_alert_banner(frame, workers_with_status)
        # Compute highest severity for this frame
        _order = {"DANGER": 2, "WARNING": 1, "SAFE": 0}
        _highest = max((w.get("status", "SAFE") for w in workers_with_status),
                       key=lambda s: _order.get(s, 0), default="SAFE")
        frame = adv_alerts.update(_highest, frame, frame_count, workers_with_status)

        # ─────────────────────────────────────
        #  3j. Draw status panel (top-left HUD)
        # ─────────────────────────────────────
        frame = draw_status_panel(
            frame, fps,
            num_workers  = len(tracked_workers),
            num_warnings = num_warnings,
            num_dangers  = num_dangers,
            num_vehicles = len(vehicles),
            admin_url    = admin_url,
        )

        # ─────────────────────────────────────
        #  3k. Push data to shared state (for admin dashboard)
        # ─────────────────────────────────────
        state.set_frame(frame)
        state.set_workers(workers_with_status)
        state.set_vehicles(vehicles)
        state.set_stats(fps, frame_count)

        # ─────────────────────────────────────
        #  3l. Terminal logging (throttled)
        # ─────────────────────────────────────
        if should_log and len(tracked_workers) > 0:
            print(f"\n[Frame {frame_count:>6d}] FPS={fps:.1f}  "
                  f"Workers={len(tracked_workers)}  "
                  f"Vehicles={len(vehicles)}  "
                  f"Warn={num_warnings}  Danger={num_dangers}")
            for w in tracked_workers:
                edge_pts = w.get("edge_points", [w["centroid"]])
                s = zones.get_worker_status_multi(edge_pts)
                vehicle_tag = " [IN VEHICLE]" if w.get("in_vehicle") else ""
                print(f"  └─ Worker {w['worker_id']:>2d}  "
                      f"centroid={w['centroid']}  "
                      f"edges={len(edge_pts)}  "
                      f"conf={w['confidence']:.2f}  "
                      f"status={s}"
                      f"{vehicle_tag}")

        # ─────────────────────────────────────
        #  3m. Display the frame
        # ─────────────────────────────────────
        cv2.imshow(config.WINDOW_NAME, frame)

        # ─────────────────────────────────────
        #  3n. Keyboard input
        # ─────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == ord('Q'):
            print("\n[main] Quit by user (Q pressed).")
            break

        elif key == ord('r') or key == ord('R'):
            worker_tracker = tracker.CentroidTracker()
            alert_manager = alerts.AlertManager()
            print("\n[main] Tracker RESET. All worker IDs cleared.")

        elif key == ord('z') or key == ord('Z'):
            show_zones = not show_zones
            state_str = "ON" if show_zones else "OFF"
            print(f"\n[main] Zone overlay toggled: {state_str}")

        elif key == ord('s') or key == ord('S'):
            os.makedirs("screenshots", exist_ok=True)
            filename = f"screenshots/frame_{frame_count}.png"
            cv2.imwrite(filename, frame)
            print(f"\n[main] Screenshot saved: {filename}")

        elif key == ord('k') or key == ord('K'):
            config.SKELETON_DRAW = not config.SKELETON_DRAW
            state_str = "ON" if config.SKELETON_DRAW else "OFF"
            print(f"\n[main] Skeleton drawing toggled: {state_str}")

        elif key == ord('m') or key == ord('M'):
            adv_alerts.toggle_mute()

    # ═══════════════════════════════════════════
    #  STEP 4: CLEANUP
    # ═══════════════════════════════════════════
    adv_alerts.shutdown()
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print(f"  Session Summary")
    print(f"  Total frames processed : {frame_count}")
    print(f"  Total worker IDs used  : {worker_tracker.next_worker_id - 1}")
    stats = state.get_stats()
    print(f"  Total alerts generated : {stats['total_alerts']}")
    print(f"  Session uptime         : {stats['uptime_str']}")
    print("=" * 60)
    print("[main] Application closed.\n")


# ──────────────────────────────────────────────
#  ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    main()
