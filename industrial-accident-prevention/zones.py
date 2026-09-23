"""
zones.py — Polygon-Based Safety Zone Module (Multi-Point + Admin Persistence)
===============================================================================
Responsibilities:
  1. Draw translucent WARNING and DANGER polygons on the video frame.
  2. Check whether a given point or set of edge points falls inside a polygon.
  3. Classify each worker as SAFE, WARNING, or DANGER using MULTI-POINT
     edge detection (any body part entering a zone triggers the alert).
  4. Load/save zone configurations from/to JSON (for admin dashboard).

Key difference from v1:
  - v1 used a SINGLE centroid point for zone checks.
  - v2 uses MULTIPLE edge keypoints (head, shoulders, elbows, wrists,
    hips, knees, ankles) — if ANY of these points enters a zone, the
    worker is classified into that zone.

Key OpenCV function used:
  cv2.pointPolygonTest(contour, point, measureDist)
    - contour    : numpy array of polygon vertices
    - point      : (x, y) tuple to test
    - measureDist: False → returns +1 (inside), 0 (on edge), -1 (outside)

Priority rule:
  DANGER is checked FIRST.  If ANY edge point is inside the danger zone
  it is classified as DANGER regardless of the warning zone.

This module does NOT handle alerts — that is alerts.py.
"""

import cv2
import numpy as np
import json
import os
import time
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
#  ZONE STORAGE (loaded from file or config defaults)
# ──────────────────────────────────────────────
# These module-level lists hold the currently active zones.
# They can be overridden by loading from zones_config.json.

_active_zones = []     # list of dicts: { "id", "type", "name", "points" (np.array) }
_zones_loaded = False  # flag to avoid loading repeatedly


def _ensure_zones_loaded():
    """Load zones from file or config defaults if not yet loaded."""
    global _zones_loaded
    if not _zones_loaded:
        load_zones()
        _zones_loaded = True


def load_zones():
    """
    Load zone configurations. Tries zones_config.json first;
    if that doesn't exist, uses the defaults from config.py.

    This function populates the module-level _active_zones list.
    """
    global _active_zones

    if os.path.exists(config.ZONES_CONFIG_FILE):
        try:
            with open(config.ZONES_CONFIG_FILE, 'r') as f:
                data = json.load(f)

            _active_zones = []
            for zone_data in data.get("zones", []):
                _active_zones.append({
                    "id": zone_data.get("id", f"zone_{len(_active_zones)+1}"),
                    "type": zone_data.get("type", "DANGER"),
                    "name": zone_data.get("name", "Unnamed Zone"),
                    "points": np.array(zone_data["points"], dtype=np.int32),
                    "created_at": zone_data.get("created_at", ""),
                })

            if _active_zones:
                print(f"[zones] Loaded {len(_active_zones)} zones from {config.ZONES_CONFIG_FILE}")
                return
            else:
                print(f"[zones] {config.ZONES_CONFIG_FILE} has no zones. Falling back to config.py defaults.")

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"[zones] Error loading {config.ZONES_CONFIG_FILE}: {e}")
            print("[zones] Falling back to config.py defaults.")

    # Fall back to config.py defaults
    _active_zones = [
        {
            "id": "default_warning",
            "type": "WARNING",
            "name": "Default Warning Zone",
            "points": config.WARNING_ZONE,
            "created_at": "",
        },
        {
            "id": "default_danger",
            "type": "DANGER",
            "name": "Default Danger Zone",
            "points": config.DANGER_ZONE,
            "created_at": "",
        },
    ]
    print("[zones] Using default zones from config.py")


def save_zones_to_file(zones_list=None):
    """
    Save zone configurations to zones_config.json.

    Args:
        zones_list : list[dict] or None — zones to save.
                     If None, saves the current _active_zones.

    Each zone dict should have: id, type, name, points (list of [x,y]).
    """
    global _active_zones

    if zones_list is not None:
        # Convert and update active zones
        _active_zones = []
        for z in zones_list:
            points = z.get("points", [])
            if isinstance(points, np.ndarray):
                points_list = points.tolist()
            else:
                points_list = points

            _active_zones.append({
                "id": z.get("id", f"zone_{len(_active_zones)+1}"),
                "type": z.get("type", "DANGER"),
                "name": z.get("name", "Unnamed Zone"),
                "points": np.array(points_list, dtype=np.int32),
                "created_at": z.get("created_at",
                                    time.strftime("%Y-%m-%dT%H:%M:%S")),
            })

    # Build JSON-serialisable data
    data = {
        "zones": []
    }
    for z in _active_zones:
        pts = z["points"]
        if isinstance(pts, np.ndarray):
            pts = pts.tolist()
        data["zones"].append({
            "id": z["id"],
            "type": z["type"],
            "name": z["name"],
            "points": pts,
            "created_at": z.get("created_at", ""),
        })

    with open(config.ZONES_CONFIG_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[zones] Saved {len(_active_zones)} zones to {config.ZONES_CONFIG_FILE}")


def get_active_zones():
    """Return the list of currently active zone dicts."""
    _ensure_zones_loaded()
    return _active_zones


def get_zones_by_type(zone_type):
    """
    Return all active zones of a given type ("WARNING" or "DANGER").

    Args:
        zone_type : str — "WARNING" or "DANGER"

    Returns:
        list of zone dicts matching the type.
    """
    _ensure_zones_loaded()
    return [z for z in _active_zones if z["type"] == zone_type]


# ──────────────────────────────────────────────
#  POINT-IN-POLYGON CHECK
# ──────────────────────────────────────────────

def is_inside_zone(point, zone_polygon):
    """
    Check whether a point lies inside a polygon.

    Uses cv2.pointPolygonTest which implements the ray-casting
    algorithm internally.

    Args:
        point        : tuple (x, y) — the point to test.
        zone_polygon : numpy.ndarray — polygon vertices, shape (N, 2).

    Returns:
        bool — True if the point is inside or on the edge of the polygon.
    """
    result = cv2.pointPolygonTest(
        contour=zone_polygon,
        pt=(float(point[0]), float(point[1])),
        measureDist=False
    )
    return result >= 0


# ──────────────────────────────────────────────
#  SINGLE-POINT STATUS (legacy compatibility)
# ──────────────────────────────────────────────

def get_worker_status(centroid, warning_zone=None, danger_zone=None):
    """
    Determine whether a worker is SAFE, in WARNING, or in DANGER
    based on a SINGLE centroid point (legacy method).

    Priority: DANGER is checked first.

    Args:
        centroid     : tuple (cx, cy)
        warning_zone : numpy.ndarray or None (uses active zones)
        danger_zone  : numpy.ndarray or None (uses active zones)

    Returns:
        str — "SAFE", "WARNING", or "DANGER"
    """
    _ensure_zones_loaded()

    # If explicit zones are passed, use them (legacy behaviour)
    if warning_zone is not None or danger_zone is not None:
        if warning_zone is None:
            warning_zone = config.WARNING_ZONE
        if danger_zone is None:
            danger_zone = config.DANGER_ZONE

        if is_inside_zone(centroid, danger_zone):
            return STATUS_DANGER
        if is_inside_zone(centroid, warning_zone):
            return STATUS_WARNING
        return STATUS_SAFE

    # Use active zones (from file or defaults)
    # Check DANGER zones first
    for zone in _active_zones:
        if zone["type"] == "DANGER" and is_inside_zone(centroid, zone["points"]):
            return STATUS_DANGER

    # Then WARNING zones
    for zone in _active_zones:
        if zone["type"] == "WARNING" and is_inside_zone(centroid, zone["points"]):
            return STATUS_WARNING

    return STATUS_SAFE


# ──────────────────────────────────────────────
#  MULTI-POINT STATUS (new — uses edge keypoints)
# ──────────────────────────────────────────────

def get_worker_status_multi(edge_points):
    """
    Determine worker status using MULTIPLE body edge keypoints.

    If ANY edge point falls inside a DANGER zone → DANGER.
    If ANY edge point falls inside a WARNING zone → WARNING.
    Otherwise → SAFE.

    This is more accurate than single-centroid checking because
    even if a worker's center is outside the zone, their hand or
    foot extending into the zone will trigger the alert.

    Args:
        edge_points : list of (x, y) tuples — body edge coordinates
                      (head, wrists, elbows, ankles, etc.)

    Returns:
        str — "SAFE", "WARNING", or "DANGER"
    """
    _ensure_zones_loaded()

    if not edge_points:
        return STATUS_SAFE

    # Check ALL danger zones first (highest priority)
    for zone in _active_zones:
        if zone["type"] == "DANGER":
            for point in edge_points:
                if is_inside_zone(point, zone["points"]):
                    return STATUS_DANGER

    # Then check ALL warning zones
    for zone in _active_zones:
        if zone["type"] == "WARNING":
            for point in edge_points:
                if is_inside_zone(point, zone["points"]):
                    return STATUS_WARNING

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

def draw_zones(frame, zones_list=None):
    """
    Draw semi-transparent zone polygons on the frame.

    Uses the active zones (loaded from file or config defaults)
    unless a custom zones_list is provided.

    Args:
        frame      : numpy.ndarray — BGR frame to draw on.
        zones_list : list[dict] or None — custom zones to draw.

    Returns:
        frame : numpy.ndarray — the annotated frame.
    """
    _ensure_zones_loaded()

    if zones_list is None:
        zones_list = _active_zones

    alpha = config.ZONE_OVERLAY_ALPHA

    if not zones_list:
        return frame

    # Step 1 & 2: Bounding-box restricted overlay for fast transparent fill
    all_pts = np.vstack([z["points"] for z in zones_list])
    min_x = max(0, int(np.min(all_pts[:, 0])))
    max_x = min(frame.shape[1], int(np.max(all_pts[:, 0])))
    min_y = max(0, int(np.min(all_pts[:, 1])))
    max_y = min(frame.shape[0], int(np.max(all_pts[:, 1])))

    if max_x > min_x and max_y > min_y:
        roi = frame[min_y:max_y, min_x:max_x]
        overlay_roi = roi.copy()

        for zone in zones_list:
            zone_type = zone["type"]
            points = zone["points"] - [min_x, min_y]

            if zone_type == "DANGER":
                color = config.COLOR_DANGER
            elif zone_type == "WARNING":
                color = config.COLOR_WARNING
            else:
                color = config.COLOR_SAFE

            cv2.fillPoly(overlay_roi, [points.reshape(-1, 1, 2)], color)

        # Blend only the ROI back into the frame
        cv2.addWeighted(overlay_roi, alpha, roi, 1 - alpha, 0, roi)

    # Step 4: Draw solid polygon borders + labels on top
    for zone in zones_list:
        zone_type = zone["type"]
        points = zone["points"]
        name = zone.get("name", zone_type)

        if zone_type == "DANGER":
            color = config.COLOR_DANGER
        elif zone_type == "WARNING":
            color = config.COLOR_WARNING
        else:
            color = config.COLOR_SAFE

        cv2.polylines(frame, [points.reshape(-1, 1, 2)],
                      isClosed=True, color=color, thickness=2)

        # Zone label at first vertex
        label_text = f"{zone_type}: {name}"
        _draw_zone_label(frame, label_text, points[0], color)

    return frame


def _draw_zone_label(frame, text, position, color):
    """
    Draw a zone label with a filled background rectangle.

    Args:
        frame    : numpy.ndarray — frame to draw on.
        text     : str           — label text.
        position : array-like    — [x, y] pixel coordinate.
        color    : tuple         — BGR colour for background.
    """
    x, y = int(position[0]), int(position[1])

    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 6, y), color, cv2.FILLED)
    cv2.putText(frame, text, (x + 3, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# ──────────────────────────────────────────────
#  SELF-TEST  (run:  python zones.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import time as _time
    import os
    import detector
    import tracker as tracker_module

    print("=" * 55)
    print("  zones.py — Self-Test  (Multi-Point Zone Detection)")
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

    # Show active zones
    load_zones()
    for z in _active_zones:
        print(f"[zones] Active zone: {z['type']} - {z['name']} "
              f"({len(z['points'])} vertices)")

    print("[zones] Press 'q' to quit, 's' to save screenshot.\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[zones] End of video stream.")
            break

        frame_count += 1
        start_time = _time.time()

        # Step 1: Detect persons with keypoints
        detections = detector.detect_persons(model, frame)

        # Step 2: Track workers
        tracked_workers = trk.update(detections)

        # Step 3: Draw zones
        frame = draw_zones(frame)

        # Step 4: Classify each worker using multi-point detection
        for worker in tracked_workers:
            x1, y1, x2, y2 = worker["bbox"]
            cx, cy = worker["centroid"]
            wid = worker["worker_id"]
            conf = worker["confidence"]
            edge_pts = worker.get("edge_points", [(cx, cy)])

            # Multi-point zone status
            status = get_worker_status_multi(edge_pts)
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

            # Draw skeleton
            keypoints = worker.get("keypoints", [])
            if keypoints:
                detector.draw_skeleton(frame, keypoints, color)

            # Centroid dot
            cv2.circle(frame, (cx, cy), 5, color, -1)

            # Edge point dots
            for (ex, ey) in edge_pts:
                cv2.circle(frame, (ex, ey), 3, (255, 255, 0), -1)

        elapsed = _time.time() - start_time
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
                edge_pts = w.get("edge_points", [(w["centroid"][0], w["centroid"][1])])
                s = get_worker_status_multi(edge_pts)
                print(f"  └─ Worker {w['worker_id']}  "
                      f"centroid={w['centroid']}  "
                      f"edges={len(edge_pts)}  status={s}")

        cv2.imshow("Zones Self-Test (Multi-Point)", frame)

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
