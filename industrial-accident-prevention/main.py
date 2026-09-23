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
import threading
import numpy as np

# ── Import project modules ──
import config
import detector
import tracker
import zones
import alerts
from shared_state import state


# ──────────────────────────────────────────────
#  THREADED CAMERA CAPTURE
# ──────────────────────────────────────────────

# ──────────────────────────────────────────────
#  THREADED CAMERA CAPTURE
# ──────────────────────────────────────────────

class ThreadedCamera:
    """
    Background thread camera capture to decouple camera I/O from AI inference.
    Ensures .read() always delivers the most recent frame without blocking.
    """
    def __init__(self, src=0, width=None, height=None):
        # Convert numeric string to int if needed
        if isinstance(src, str) and src.isdigit():
            src = int(src)

        # On Windows, DirectShow (CAP_DSHOW) opens faster and has zero driver lag
        if isinstance(src, int) and sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(src)

        if width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Minimize buffer size so we never lag behind live frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.grabbed, self.frame = self.cap.read()
        self.read_lock = threading.Lock()
        self.stopped = False
        self.started = False

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        return self

    def _capture_loop(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if not grabbed:
                time.sleep(0.01)
                continue
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.read_lock:
            grabbed = self.grabbed
            frame = self.frame.copy() if self.frame is not None else None
        return grabbed, frame

    def isOpened(self):
        return self.cap.isOpened()

    def get(self, prop_id):
        return self.cap.get(prop_id)

    def release(self):
        self.stopped = True
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()


# ──────────────────────────────────────────────
#  ASYNCHRONOUS INFERENCE PIPELINE
# ──────────────────────────────────────────────

class AsyncInferencePipeline:
    """
    Dedicated worker thread for AI inference and tracking.
    Runs YOLO pose, vehicle detection, tracking, and zone checks
    asynchronously so the camera capture and display loop runs
    buttery-smooth at full 30-60 FPS without ever hitching or freezing.
    """
    def __init__(self, pose_model, vehicle_model, worker_tracker, alert_manager):
        self.pose_model = pose_model
        self.vehicle_model = vehicle_model
        self.worker_tracker = worker_tracker
        self.alert_manager = alert_manager

        self.lock = threading.Lock()
        self.running = True

        # Input frame slot
        self.pending_frame = None
        self.has_new_frame = threading.Event()

        # Shared results protected by self.lock
        self.tracked_workers = []
        self.workers_with_status = []
        self.vehicles = []
        self.num_warnings = 0
        self.num_dangers = 0
        self.ai_fps = 0.0
        self.inference_ms = 0.0
        self.ai_cycle_count = 0

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def push_frame(self, frame):
        """Called by the main display loop to supply the freshest frame."""
        if not self.has_new_frame.is_set():
            with self.lock:
                self.pending_frame = frame.copy()
            self.has_new_frame.set()

    def get_results(self):
        """Called by the display loop to retrieve the latest AI detection state."""
        with self.lock:
            return (
                self.tracked_workers.copy(),
                self.workers_with_status.copy(),
                self.vehicles.copy(),
                self.num_warnings,
                self.num_dangers,
                self.ai_fps,
                self.inference_ms
            )

    def reset_tracker(self):
        """Thread-safe reset of tracker and alerts."""
        with self.lock:
            self.worker_tracker = tracker.CentroidTracker()
            self.alert_manager = alerts.AlertManager()
            self.tracked_workers = []
            self.workers_with_status = []
            self.vehicles = []
            self.num_warnings = 0
            self.num_dangers = 0

    def _worker_loop(self):
        prev_time = time.time()
        while self.running:
            got_frame = self.has_new_frame.wait(timeout=0.05)
            if not self.running:
                break
            if not got_frame:
                continue

            with self.lock:
                frame = self.pending_frame
                self.has_new_frame.clear()

            if frame is None:
                continue

            t_start = time.perf_counter()
            self.ai_cycle_count += 1

            # 1. Person pose & keypoint detection
            detections = detector.detect_persons(self.pose_model, frame)

            # 2. Vehicle detection (every N cycles or when vehicles present)
            v_interval = getattr(config, 'VEHICLE_INFERENCE_INTERVAL', 3)
            vehicles = []
            if config.DETECT_VEHICLES:
                if (self.ai_cycle_count % v_interval == 0) or len(self.vehicles) > 0:
                    vehicles = detector.detect_vehicles(self.vehicle_model, frame)
                    detections = detector.detect_persons_in_vehicles(detections, vehicles)
                else:
                    with self.lock:
                        vehicles = self.vehicles

            # 3. Worker ID tracking
            tracked = self.worker_tracker.update(detections)

            # 4. Multi-point zone evaluation & alerts
            workers_status = []
            warn_cnt = 0
            dang_cnt = 0

            for w in tracked:
                centroid = w["centroid"]
                wid = w["worker_id"]
                edge_pts = w.get("edge_points", [centroid])

                status = zones.get_worker_status_multi(edge_pts)
                w_data = {
                    "worker_id": wid,
                    "status": status,
                    "centroid": centroid,
                    "bbox": w["bbox"],
                    "confidence": w["confidence"],
                    "edge_points": edge_pts,
                    "in_vehicle": w.get("in_vehicle", False),
                    "vehicle_info": w.get("vehicle_info"),
                }
                workers_status.append(w_data)

                if status == zones.STATUS_WARNING:
                    warn_cnt += 1
                    self.alert_manager.warning_alert(wid)
                    state.add_alert(wid, "WARNING", f"Worker {wid} approaching danger zone")
                elif status == zones.STATUS_DANGER:
                    dang_cnt += 1
                    self.alert_manager.danger_alert(wid)
                    state.add_alert(wid, "DANGER", f"Worker {wid} entered restricted zone!")

            t_end = time.perf_counter()
            cycle_ms = (t_end - t_start) * 1000

            now = time.time()
            elapsed = now - prev_time
            prev_time = now
            inst_ai_fps = 1.0 / elapsed if elapsed > 0 else 0.0

            with self.lock:
                self.tracked_workers = tracked
                self.workers_with_status = workers_status
                self.vehicles = vehicles
                self.num_warnings = warn_cnt
                self.num_dangers = dang_cnt
                self.ai_fps = (self.ai_fps * 0.8 + inst_ai_fps * 0.2) if self.ai_fps > 0 else inst_ai_fps
                self.inference_ms = cycle_ms

    def stop(self):
        self.running = False
        self.has_new_frame.set()
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)



# ──────────────────────────────────────────────
#  STATUS PANEL (HUD at the top-left of frame)
# ──────────────────────────────────────────────

def draw_status_panel(frame, fps, num_workers, num_warnings, num_dangers,
                      num_vehicles=0, admin_url="", inference_ms=0.0, ai_fps=0.0):
    """
    Draw a semi-transparent status panel in the top-left corner
    showing real-time system statistics.
    Optimised to avoid full-frame copy via in-place ROI blending.
    """
    panel_w = 340
    panel_h = 220
    margin  = 10
    x1, y1  = margin, margin
    x2, y2  = min(frame.shape[1], x1 + panel_w), min(frame.shape[0], y1 + panel_h)

    # In-place ROI blending (no full-frame copy)
    sub = frame[y1:y2, x1:x2]
    rect = np.full(sub.shape, (30, 30, 30), dtype=np.uint8)
    cv2.addWeighted(rect, 0.75, sub, 0.25, 0, sub)

    # Border
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

    # Text content
    text_x = x1 + 12
    line_y = y1 + 25
    line_gap = 24

    # Title
    cv2.putText(frame, "AI ACCIDENT PREVENTION SYSTEM", (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
    line_y += line_gap + 2

    # FPS — colour-coded with AI latency and AI FPS
    fps_color = (0, 200, 0) if fps >= 20 else (0, 200, 255) if fps >= 10 else (0, 0, 255)
    fps_text = f"Display: {fps:.1f} FPS"
    if ai_fps > 0:
        fps_text += f" | AI: {ai_fps:.1f} FPS"
    elif inference_ms > 0:
        fps_text += f" | AI: {inference_ms:.0f}ms"
    cv2.putText(frame, fps_text, (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_color, 2)
    line_y += line_gap

    # Workers + Vehicles
    v_status = f"{num_vehicles}" if config.DETECT_VEHICLES else "OFF"
    cv2.putText(frame, f"Workers: {num_workers}   Vehicles: {v_status}",
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
    cv2.putText(frame, "[Q]uit [R]eset [Z]ones [S]ave [K]skel [V]ehicles",
                (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (160, 160, 160), 1)

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

    # 1d. Create the alert manager.
    alert_manager = alerts.AlertManager()

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

    # 1g. Open the video source (threaded capture if enabled for live streams).
    is_live_source = (
        isinstance(config.VIDEO_SOURCE, int) or
        (isinstance(config.VIDEO_SOURCE, str) and (
            config.VIDEO_SOURCE.isdigit() or
            config.VIDEO_SOURCE.startswith(('rtsp://', 'http://', 'https://'))
        ))
    )

    if getattr(config, 'THREADED_CAMERA', False) and is_live_source:
        cap = ThreadedCamera(
            src=config.VIDEO_SOURCE,
            width=getattr(config, 'CAMERA_WIDTH', None),
            height=getattr(config, 'CAMERA_HEIGHT', None)
        ).start()
        print(f"\n[main] Threaded camera capture ENABLED")
    else:
        src = int(config.VIDEO_SOURCE) if isinstance(config.VIDEO_SOURCE, str) and config.VIDEO_SOURCE.isdigit() else config.VIDEO_SOURCE
        cap = cv2.VideoCapture(src)
        if getattr(config, 'CAMERA_WIDTH', None) and getattr(config, 'CAMERA_HEIGHT', None):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

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
    print(f"[main] Frame skip    : {getattr(config, 'INFERENCE_SKIP_FRAMES', 1)}")
    print(f"[main] Inference size: {getattr(config, 'YOLO_IMGSZ', 640)}")
    if admin_url:
        print(f"[main] Admin panel   : {admin_url}")
    print()
    print("[main] Controls:")
    print("         Q = Quit")
    print("         R = Reset tracker (clear all worker IDs)")
    print("         Z = Show / hide zone overlays")
    print("         S = Save screenshot")
    print("         K = Toggle skeleton drawing")
    print("         V = Toggle vehicle detection (saves CPU)")
    print()

    # 1h. Start Async Inference Pipeline if enabled
    async_pipeline = None
    if getattr(config, 'ASYNC_INFERENCE', True):
        async_pipeline = AsyncInferencePipeline(
            pose_model=pose_model,
            vehicle_model=vehicle_model,
            worker_tracker=worker_tracker,
            alert_manager=alert_manager
        ).start()
        print("[main] Async non-blocking AI inference pipeline ENABLED (30-60 FPS UI)")

    # ═══════════════════════════════════════════
    #  STEP 2: STATE VARIABLES
    # ═══════════════════════════════════════════

    frame_count         = 0
    show_zones          = True
    fps                 = 0.0
    ai_fps              = 0.0
    total_ai_ms         = 0.0
    prev_time           = time.time()
    detections          = []
    vehicles            = []
    tracked_workers     = []
    workers_with_status = []
    num_warnings        = 0
    num_dangers         = 0

    TERMINAL_LOG_INTERVAL = 30

    # ═══════════════════════════════════════════
    #  STEP 3: MAIN PROCESSING LOOP
    # ═══════════════════════════════════════════

    while True:
        # ─────────────────────────────────────
        #  3a. Capture a frame
        # ─────────────────────────────────────
        ret, frame = cap.read()

        if not ret or frame is None:
            print("\n[main] Video stream ended or camera disconnected.")
            break

        # Resize frame if capture didn't respect configured target resolution
        cam_w = getattr(config, 'CAMERA_WIDTH', None)
        cam_h = getattr(config, 'CAMERA_HEIGHT', None)
        if cam_w and cam_h:
            if frame.shape[1] != cam_w or frame.shape[0] != cam_h:
                frame = cv2.resize(frame, (cam_w, cam_h), interpolation=cv2.INTER_LINEAR)

        frame_count += 1

        # ─────────────────────────────────────
        #  3b. Smoothed FPS calculation
        # ─────────────────────────────────────
        current_time = time.time()
        elapsed = current_time - prev_time
        instant_fps = 1.0 / elapsed if elapsed > 0 else 0.0
        fps = (fps * 0.85 + instant_fps * 0.15) if fps > 0 else instant_fps
        prev_time = current_time

        # ─────────────────────────────────────
        #  3c-3e. AI Inference (Async / Sync)
        # ─────────────────────────────────────
        if async_pipeline is not None:
            # Feed current frame to the AI worker without blocking
            async_pipeline.push_frame(frame)

            # Retrieve latest AI detection & tracking results instantaneously
            (tracked_workers,
             workers_with_status,
             vehicles,
             num_warnings,
             num_dangers,
             ai_fps,
             total_ai_ms) = async_pipeline.get_results()

            should_log = (frame_count % TERMINAL_LOG_INTERVAL == 0)
        else:
            skip_interval = max(1, getattr(config, 'INFERENCE_SKIP_FRAMES', 1))
            run_inference = (frame_count == 1) or (frame_count % skip_interval == 0)

            if run_inference:
                detections = detector.detect_persons(pose_model, frame)
                if config.DETECT_VEHICLES:
                    vehicles = detector.detect_vehicles(vehicle_model, frame)
                    detections = detector.detect_persons_in_vehicles(detections, vehicles)
                else:
                    vehicles = []

                tracked_workers = worker_tracker.update(detections)

                workers_with_status = []
                num_warnings = 0
                num_dangers  = 0

                should_log = (frame_count % TERMINAL_LOG_INTERVAL == 0)

                for worker in tracked_workers:
                    centroid = worker["centroid"]
                    wid      = worker["worker_id"]
                    edge_pts = worker.get("edge_points", [centroid])

                    status = zones.get_worker_status_multi(edge_pts)

                    worker_data = {
                        "worker_id": wid,
                        "status":    status,
                        "centroid":  centroid,
                        "bbox":      worker["bbox"],
                        "confidence": worker["confidence"],
                        "edge_points": edge_pts,
                        "in_vehicle": worker.get("in_vehicle", False),
                        "vehicle_info": worker.get("vehicle_info"),
                    }
                    workers_with_status.append(worker_data)

                    if status == zones.STATUS_WARNING:
                        num_warnings += 1
                        if should_log:
                            alert_manager.warning_alert(wid)
                        state.add_alert(wid, "WARNING",
                                        f"Worker {wid} approaching danger zone")
                    elif status == zones.STATUS_DANGER:
                        num_dangers += 1
                        alert_manager.danger_alert(wid)
                        state.add_alert(wid, "DANGER",
                                        f"Worker {wid} entered restricted zone!")

            total_ai_ms = detector.last_pose_inference_ms + detector.last_vehicle_inference_ms
            ai_fps = 0.0

        # ─────────────────────────────────────
        #  3f. Draw zone overlays (if enabled)
        # ─────────────────────────────────────
        if show_zones:
            frame = zones.draw_zones(frame)

        # ─────────────────────────────────────
        #  3g. Draw worker annotations
        # ─────────────────────────────────────
        for w_item in workers_with_status:
            worker_dict = {
                "bbox": w_item["bbox"],
                "centroid": w_item["centroid"],
                "worker_id": w_item["worker_id"],
                "confidence": w_item["confidence"],
                "in_vehicle": w_item.get("in_vehicle", False),
                "vehicle_info": w_item.get("vehicle_info"),
                "edge_points": w_item.get("edge_points", []),
                "keypoints": next((tw.get("keypoints", []) for tw in tracked_workers if tw["worker_id"] == w_item["worker_id"]), []),
            }
            frame = draw_worker(frame, worker_dict, w_item["status"])

        # ─────────────────────────────────────
        #  3h. Draw vehicle bounding boxes
        # ─────────────────────────────────────
        frame = detector.draw_vehicles(frame, vehicles)

        # ─────────────────────────────────────
        #  3i. Draw alert banners (top-right)
        # ─────────────────────────────────────
        frame = alert_manager.draw_alert_banner(frame, workers_with_status)

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
            inference_ms = total_ai_ms,
            ai_fps       = ai_fps,
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
            print(f"\n[Frame {frame_count:>6d}] DisplayFPS={fps:.1f}  AIFPS={ai_fps:.1f}  "
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
            if async_pipeline:
                async_pipeline.stop()
            break

        elif key == ord('r') or key == ord('R'):
            if async_pipeline:
                async_pipeline.reset_tracker()
            else:
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

        elif key == ord('v') or key == ord('V'):
            config.DETECT_VEHICLES = not config.DETECT_VEHICLES
            state_str = "ENABLED" if config.DETECT_VEHICLES else "DISABLED"
            print(f"\n[main] Vehicle detection toggled: {state_str}")

    # ═══════════════════════════════════════════
    #  STEP 4: CLEANUP
    # ═══════════════════════════════════════════
    if async_pipeline:
        async_pipeline.stop()
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
