"""
zones.py — Polygon-Based Safety Zone Module
============================================
Responsibilities:
  1. Draw translucent WARNING and DANGER polygons on the video frame.
  2. Check whether a given centroid point falls inside a polygon.
  3. Classify each worker as SAFE, WARNING, or DANGER.

Key OpenCV function used:
  cv2.pointPolygonTest(contour, point, measureDist)
    - contour    : numpy array of polygon vertices
    - point      : (x, y) tuple to test
    - measureDist: False → returns +1 (inside), 0 (on edge), -1 (outside)

Priority rule:
  DANGER is checked FIRST.  If a centroid is inside the danger zone
  it is classified as DANGER regardless of the warning zone.
  Only if it is NOT in the danger zone do we check the warning zone.

How polygon coordinates map to image pixels:
  ┌─────────────────────────────────────────────────────┐
  │ (0,0)                                    (1919,0)   │
  │   ┌──── WARNING ZONE ────┐                         │
  │   │  ┌── DANGER ZONE ──┐ │                         │
  │   │  │                  │ │                         │
  │   │  │   (machine /     │ │                         │
  │   │  │    hazard area)  │ │                         │
  │   │  └──────────────────┘ │                         │
  │   └───────────────────────┘                         │
  │                                          (1919,1079)│
  └─────────────────────────────────────────────────────┘
  x increases left → right   (columns)
  y increases top  → bottom  (rows)
  All coordinates are in pixels for the given resolution.

This module does NOT handle alerts — that is alerts.py.
"""

import cv2
import numpy as np
import config


# ──────────────────────────────────────────────
#  STATUS CONSTANTS
# ──────────────────────────────────────────────
# These strings are returned by get_worker_status() and used
# by other modules (alerts.py, main.py) for decisions and display.

STATUS_SAFE    = "SAFE"
STATUS_WARNING = "WARNING"
STATUS_DANGER  = "DANGER"


# ──────────────────────────────────────────────
#  POINT-IN-POLYGON CHECK
# ──────────────────────────────────────────────

def is_inside_zone(centroid, zone_polygon):
    """
    Check whether a centroid point lies inside a polygon.

    Uses cv2.pointPolygonTest which implements the ray-casting
    algorithm internally — a standard computational geometry method
    for point-in-polygon tests.

    Args:
        centroid     : tuple (cx, cy) — the point to test.
        zone_polygon : numpy.ndarray  — polygon vertices, shape (N, 2).

    Returns:
        bool — True if the point is inside or on the edge of the polygon.

    How it works:
        pointPolygonTest returns:
          +1  → point is INSIDE the polygon
           0  → point is ON the polygon edge
          -1  → point is OUTSIDE the polygon
        We treat both +1 and 0 as "inside" for safety purposes.
    """
    # pointPolygonTest expects the point as a tuple of floats.
    result = cv2.pointPolygonTest(
        contour=zone_polygon,
        pt=(float(centroid[0]), float(centroid[1])),
        measureDist=False          # we only need inside/outside, not distance
    )
    # result >= 0 means the point is inside or on the boundary.
    return result >= 0


# ──────────────────────────────────────────────
#  WORKER STATUS CLASSIFICATION
# ──────────────────────────────────────────────

def get_worker_status(centroid, warning_zone=None, danger_zone=None):
    """
    Determine whether a worker is SAFE, in WARNING, or in DANGER.

    Priority: DANGER is checked first.
      1. If centroid is inside danger_zone  → "DANGER"
      2. If centroid is inside warning_zone → "WARNING"
      3. Otherwise                          → "SAFE"

    Args:
        centroid     : tuple (cx, cy) — worker's centroid position.
        warning_zone : numpy.ndarray  — warning polygon (default: from config).
        danger_zone  : numpy.ndarray  — danger polygon  (default: from config).

    Returns:
        str — one of: "SAFE", "WARNING", "DANGER"
    """
    # Use config defaults if no zones are passed in.
    if warning_zone is None:
        warning_zone = config.WARNING_ZONE
    if danger_zone is None:
        danger_zone = config.DANGER_ZONE

    # ── Check DANGER first (highest priority) ──
    if is_inside_zone(centroid, danger_zone):
        return STATUS_DANGER

    # ── Then check WARNING ──
    if is_inside_zone(centroid, warning_zone):
        return STATUS_WARNING

    # ── Otherwise the worker is SAFE ──
    return STATUS_SAFE


# ──────────────────────────────────────────────
#  STATUS → COLOUR MAPPING
# ──────────────────────────────────────────────

def get_status_color(status):
    """
    Return the BGR colour associated with a status string.

    Args:
        status : str — "SAFE", "WARNING", or "DANGER"

    Returns:
        tuple (B, G, R) — colour from config.py
    """
    if status == STATUS_DANGER:
        return config.COLOR_DANGER
    elif status == STATUS_WARNING:
        return config.COLOR_WARNING
    else:
        return config.COLOR_SAFE


# ──────────────────────────────────────────────
#  ZONE DRAWING
# ──────────────────────────────────────────────

def draw_zones(frame, warning_zone=None, danger_zone=None):
    """
    Draw semi-transparent WARNING and DANGER polygons on the frame.

    The zones are rendered as translucent filled polygons with a
    solid border, so the camera feed is still visible underneath.

    Args:
        frame        : numpy.ndarray — BGR frame to draw on (modified in place).
        warning_zone : numpy.ndarray — warning polygon (default: from config).
        danger_zone  : numpy.ndarray — danger polygon  (default: from config).

    Returns:
        frame : numpy.ndarray — the annotated frame.

    Transparency technique:
        1. Create a copy of the frame (the "overlay").
        2. Draw filled polygons on the overlay.
        3. Blend the overlay with the original frame using
           cv2.addWeighted() at the configured alpha level.
        4. Draw solid polygon borders on top.
    """
    if warning_zone is None:
        warning_zone = config.WARNING_ZONE
    if danger_zone is None:
        danger_zone = config.DANGER_ZONE

    alpha = config.ZONE_OVERLAY_ALPHA

    # ── Step 1: Create an overlay copy for transparent fill ──
    overlay = frame.copy()

    # ── Step 2: Draw filled polygons on the overlay ──
    # fillPoly expects a list of arrays, each with shape (N, 1, 2).
    cv2.fillPoly(overlay, [warning_zone.reshape(-1, 1, 2)], config.COLOR_WARNING)
    cv2.fillPoly(overlay, [danger_zone.reshape(-1, 1, 2)],  config.COLOR_DANGER)

    # ── Step 3: Blend overlay with original frame ──
    # result = alpha * overlay + (1 - alpha) * frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ── Step 4: Draw solid polygon borders on top ──
    # polylines draws the outline; isClosed=True connects last→first vertex.
    cv2.polylines(frame, [warning_zone.reshape(-1, 1, 2)],
                  isClosed=True, color=config.COLOR_WARNING, thickness=2)
    cv2.polylines(frame, [danger_zone.reshape(-1, 1, 2)],
                  isClosed=True, color=config.COLOR_DANGER, thickness=2)

    # ── Step 5: Add text labels to each zone ──
    # Place labels at the first vertex of each polygon.
    _draw_zone_label(frame, "WARNING ZONE", warning_zone[0], config.COLOR_WARNING)
    _draw_zone_label(frame, "DANGER ZONE",  danger_zone[0],  config.COLOR_DANGER)

    return frame


def _draw_zone_label(frame, text, position, color):
    """
    Draw a zone label with a filled background rectangle at the
    given position.

    Args:
        frame    : numpy.ndarray — frame to draw on.
        text     : str           — label text (e.g. "DANGER ZONE").
        position : array-like    — [x, y] pixel coordinate for the label.
        color    : tuple         — BGR colour for background.
    """
    x, y = int(position[0]), int(position[1])

    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )
    # Background rectangle
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, cv2.FILLED)
    # White text
    cv2.putText(frame, text, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# ──────────────────────────────────────────────
#  SELF-TEST  (run:  python zones.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import time
    import os
    import detector
    import tracker as tracker_module

    print("=" * 55)
    print("  zones.py — Self-Test  (Detector + Tracker + Zones)")
    print("=" * 55)

    # Load YOLO model
    model = detector.load_model()

    # Create tracker
    trk = tracker_module.CentroidTracker()

    # Open video source
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {config.VIDEO_SOURCE}")
        exit(1)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[zones] Video opened: {config.VIDEO_SOURCE}")
    print(f"[zones] Resolution : {frame_w} x {frame_h}")
    print(f"[zones] WARNING zone vertices: {config.WARNING_ZONE.tolist()}")
    print(f"[zones] DANGER  zone vertices: {config.DANGER_ZONE.tolist()}")
    print("[zones] Press 'q' to quit, 's' to save screenshot.\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[zones] End of video stream.")
            break

        frame_count += 1
        start_time = time.time()

        # Step 1: Detect persons
        detections = detector.detect_persons(model, frame)

        # Step 2: Track workers
        tracked_workers = trk.update(detections)

        # Step 3: Draw zones FIRST (so they appear behind workers)
        frame = draw_zones(frame)

        # Step 4: Classify each worker and draw results
        for worker in tracked_workers:
            x1, y1, x2, y2 = worker["bbox"]
            cx, cy = worker["centroid"]
            wid = worker["worker_id"]
            conf = worker["confidence"]

            # Get zone status for this worker
            status = get_worker_status((cx, cy))
            color = get_status_color(status)

            # Bounding box coloured by status
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BBOX_THICKNESS)

            # Worker ID + status label
            label = f"Worker {wid} [{status}] {conf:.2f}"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, config.FONT_THICKNESS
            )
            cv2.rectangle(frame, (x1, y1 - th - baseline - 6),
                          (x1 + tw + 4, y1), color, cv2.FILLED)
            cv2.putText(frame, label, (x1 + 2, y1 - baseline - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                        (255, 255, 255), config.FONT_THICKNESS)

            # Centroid dot
            cv2.circle(frame, (cx, cy), 5, color, -1)

        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # HUD
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(frame, f"Tracked: {len(tracked_workers)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        # Print to terminal (every 30 frames)
        if frame_count % 30 == 0:
            print(f"Frame {frame_count:>5d} | FPS {fps:5.1f}")
            for w in tracked_workers:
                s = get_worker_status(w["centroid"])
                print(f"  └─ Worker {w['worker_id']}  "
                      f"centroid={w['centroid']}  status={s}")

        cv2.imshow("Zones Self-Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[zones] Quit by user.")
            break
        elif key == ord('s'):
            os.makedirs("screenshots", exist_ok=True)
            fname = f"screenshots/zones_frame_{frame_count}.png"
            cv2.imwrite(fname, frame)
            print(f"[zones] Screenshot saved: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    print("[zones] Self-test finished.")
