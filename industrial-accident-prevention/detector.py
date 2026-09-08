"""
detector.py — Person & Vehicle Detection Module (Pose + Standard)
==================================================================
Responsibilities:
  1. Load TWO YOLO models at startup:
       - yolov8n-pose.pt  → person detection with 17-keypoint skeleton
       - yolov8n.pt       → vehicle detection (car, bus, truck)
  2. Run pose inference on each frame to get person detections
     with full skeleton keypoints.
  3. Run standard inference for vehicle detection.
  4. Determine which persons are inside vehicles using bounding
     box overlap (IoU-style).
  5. Return enriched detection dicts containing:
       - bounding box, confidence, centroid (as before)
       - keypoints: list of 17 (x, y, conf) tuples
       - edge_points: filtered keypoints used for zone checks
       - in_vehicle: bool flag + vehicle info
  6. Provide drawing helpers for skeleton, vehicles, and labels.

This module does NOT handle tracking or zones — those are
separate modules (tracker.py, zones.py).
"""

import cv2
import numpy as np
from ultralytics import YOLO
import config


# ──────────────────────────────────────────────
#  MODEL LOADING
# ──────────────────────────────────────────────

def load_model():
    """
    Load the YOLO pose model for person detection with keypoints.

    Returns:
        model : ultralytics.YOLO — the loaded YOLO pose model object.

    Notes:
        - On the FIRST run, Ultralytics auto-downloads the weights
          file (yolov8n-pose.pt ≈ 6 MB) into the working directory.
        - The model object should be loaded ONCE and reused.
    """
    model = YOLO(config.YOLO_POSE_MODEL_PATH)
    print(f"[detector] YOLO pose model loaded: {config.YOLO_POSE_MODEL_PATH}")
    return model


def load_vehicle_model():
    """
    Load the standard YOLO model for vehicle detection.

    Returns:
        model : ultralytics.YOLO — the loaded YOLO model object.

    This is separate from the pose model because yolov8-pose only
    detects persons. We need the standard model to detect vehicles
    (car, bus, truck classes).
    """
    model = YOLO(config.YOLO_VEHICLE_MODEL_PATH)
    print(f"[detector] YOLO vehicle model loaded: {config.YOLO_VEHICLE_MODEL_PATH}")
    return model


# ──────────────────────────────────────────────
#  PERSON DETECTION WITH KEYPOINTS
# ──────────────────────────────────────────────

def detect_persons(model, frame):
    """
    Run YOLO pose inference on a single BGR frame and return
    'person' detections with full skeleton keypoints.

    Args:
        model : ultralytics.YOLO  — the pre-loaded YOLO pose model.
        frame : numpy.ndarray     — a BGR image from OpenCV.

    Returns:
        detections : list[dict]
            Each dict contains:
                "bbox"       : (x1, y1, x2, y2) — top-left & bottom-right
                "confidence" : float             — YOLO confidence score
                "class_id"   : int               — COCO class ID (0 = person)
                "centroid"   : (cx, cy)          — center point of bounding box
                "keypoints"  : list of 17 tuples (x, y, kp_conf)
                "edge_points": list of (x, y) for body-edge keypoints
                               with confidence above threshold
                "in_vehicle" : False             — default, updated later
                "vehicle_info": None             — default, updated later
    """
    # Run YOLO pose inference
    results = model(frame, conf=config.YOLO_CONFIDENCE_THRESHOLD, verbose=False)

    detections = []

    for result in results:
        # Check if keypoints are available (pose model)
        has_keypoints = hasattr(result, 'keypoints') and result.keypoints is not None

        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])

            # Filter: keep ONLY person detections
            if class_id != config.PERSON_CLASS_ID:
                continue

            # Bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Confidence score
            confidence = round(float(box.conf[0]), 4)

            # Centroid (geometric center of bounding box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Extract keypoints if available
            keypoints = []
            edge_points = []

            if has_keypoints and i < len(result.keypoints):
                kp_data = result.keypoints[i]

                # kp_data.xy has shape [1, 17, 2] — the (x, y) coords
                # kp_data.conf has shape [1, 17] — the confidence scores
                if kp_data.xy is not None and kp_data.conf is not None:
                    xy = kp_data.xy[0].cpu().numpy()      # shape: (17, 2)
                    confs = kp_data.conf[0].cpu().numpy()  # shape: (17,)

                    for kp_idx in range(17):
                        kp_x = float(xy[kp_idx][0])
                        kp_y = float(xy[kp_idx][1])
                        kp_conf = float(confs[kp_idx])
                        keypoints.append((kp_x, kp_y, kp_conf))

                        # Collect edge points above confidence threshold
                        if (kp_idx in config.EDGE_KEYPOINT_INDICES and
                                kp_conf >= config.KEYPOINT_CONFIDENCE_THRESHOLD and
                                kp_x > 0 and kp_y > 0):
                            edge_points.append((int(kp_x), int(kp_y)))

            # If no valid edge points, fall back to centroid
            if not edge_points:
                edge_points = [(cx, cy)]

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": confidence,
                "class_id": class_id,
                "centroid": (cx, cy),
                "keypoints": keypoints,
                "edge_points": edge_points,
                "in_vehicle": False,
                "vehicle_info": None,
            })

    return detections


# ──────────────────────────────────────────────
#  VEHICLE DETECTION
# ──────────────────────────────────────────────

def detect_vehicles(model, frame):
    """
    Run standard YOLO inference to detect vehicles.

    Args:
        model : ultralytics.YOLO — the standard YOLO model.
        frame : numpy.ndarray    — a BGR image from OpenCV.

    Returns:
        vehicles : list[dict]
            Each dict contains:
                "bbox"       : (x1, y1, x2, y2)
                "confidence" : float
                "class_id"   : int (2=car, 5=bus, 7=truck)
                "class_name" : str ("Car", "Bus", "Truck")
                "centroid"   : (cx, cy)
    """
    if not config.DETECT_VEHICLES:
        return []

    results = model(frame, conf=config.YOLO_CONFIDENCE_THRESHOLD, verbose=False)

    vehicles = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])

        if class_id not in config.VEHICLE_CLASS_IDS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        confidence = round(float(box.conf[0]), 4)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        class_name = config.VEHICLE_CLASS_NAMES.get(class_id, f"Vehicle-{class_id}")

        vehicles.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": confidence,
            "class_id": class_id,
            "class_name": class_name,
            "centroid": (cx, cy),
        })

    return vehicles


# ──────────────────────────────────────────────
#  PERSON-IN-VEHICLE DETECTION
# ──────────────────────────────────────────────

def _compute_overlap_ratio(person_bbox, vehicle_bbox):
    """
    Compute what fraction of the person's bounding box area overlaps
    with the vehicle's bounding box.

    Args:
        person_bbox  : (x1, y1, x2, y2)
        vehicle_bbox : (x1, y1, x2, y2)

    Returns:
        float — overlap ratio (0.0 = no overlap, 1.0 = fully inside)
    """
    px1, py1, px2, py2 = person_bbox
    vx1, vy1, vx2, vy2 = vehicle_bbox

    # Intersection coordinates
    ix1 = max(px1, vx1)
    iy1 = max(py1, vy1)
    ix2 = min(px2, vx2)
    iy2 = min(py2, vy2)

    # No overlap
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    person_area = (px2 - px1) * (py2 - py1)

    if person_area <= 0:
        return 0.0

    return intersection_area / person_area


def detect_persons_in_vehicles(person_detections, vehicle_detections):
    """
    Cross-reference person and vehicle detections to determine which
    persons are inside/on vehicles.

    A person is considered "in a vehicle" if their bounding box overlaps
    with a vehicle bounding box by at least VEHICLE_OVERLAP_THRESHOLD.

    Args:
        person_detections  : list[dict] — output of detect_persons()
        vehicle_detections : list[dict] — output of detect_vehicles()

    Returns:
        person_detections : list[dict] — same list, with in_vehicle
                            and vehicle_info fields updated in-place.
    """
    if not vehicle_detections:
        return person_detections

    for person in person_detections:
        best_overlap = 0.0
        best_vehicle = None

        for vehicle in vehicle_detections:
            overlap = _compute_overlap_ratio(person["bbox"], vehicle["bbox"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_vehicle = vehicle

        if best_overlap >= config.VEHICLE_OVERLAP_THRESHOLD and best_vehicle is not None:
            person["in_vehicle"] = True
            person["vehicle_info"] = {
                "class_name": best_vehicle["class_name"],
                "class_id": best_vehicle["class_id"],
                "vehicle_bbox": best_vehicle["bbox"],
                "overlap": round(best_overlap, 3),
            }

    return person_detections


# ──────────────────────────────────────────────
#  DRAWING HELPERS
# ──────────────────────────────────────────────

def draw_skeleton(frame, keypoints, color=None):
    """
    Draw the skeleton (keypoint joints + connecting lines) on the frame.

    Args:
        frame     : numpy.ndarray — BGR frame to draw on.
        keypoints : list of 17 (x, y, conf) tuples.
        color     : tuple or None — override colour; uses config defaults if None.

    Returns:
        frame : numpy.ndarray — annotated frame.
    """
    if not keypoints or not config.SKELETON_DRAW:
        return frame

    line_color = color or config.SKELETON_LINE_COLOR
    joint_color = color or config.SKELETON_JOINT_COLOR

    # Draw skeleton connection lines
    for (idx_a, idx_b) in config.SKELETON_CONNECTIONS:
        if idx_a >= len(keypoints) or idx_b >= len(keypoints):
            continue

        xa, ya, ca = keypoints[idx_a]
        xb, yb, cb = keypoints[idx_b]

        # Only draw if both keypoints are confident enough
        if (ca >= config.KEYPOINT_CONFIDENCE_THRESHOLD and
                cb >= config.KEYPOINT_CONFIDENCE_THRESHOLD and
                xa > 0 and ya > 0 and xb > 0 and yb > 0):
            cv2.line(frame,
                     (int(xa), int(ya)),
                     (int(xb), int(yb)),
                     line_color,
                     config.SKELETON_LINE_THICKNESS)

    # Draw keypoint joints as circles
    for kp_idx, (kx, ky, kc) in enumerate(keypoints):
        if (kc >= config.KEYPOINT_CONFIDENCE_THRESHOLD and
                kx > 0 and ky > 0):
            # Edge keypoints get a larger, brighter dot
            if kp_idx in config.EDGE_KEYPOINT_INDICES:
                cv2.circle(frame, (int(kx), int(ky)),
                           config.SKELETON_JOINT_RADIUS + 1,
                           (0, 255, 255), -1)  # Yellow for edges
            else:
                cv2.circle(frame, (int(kx), int(ky)),
                           config.SKELETON_JOINT_RADIUS,
                           joint_color, -1)

    return frame


def draw_detections(frame, detections):
    """
    Draw bounding boxes, skeletons, centroid dots, and confidence labels
    onto the frame for every detection.

    Args:
        frame      : numpy.ndarray — the BGR frame to draw on.
        detections : list[dict]    — output from detect_persons().

    Returns:
        frame : numpy.ndarray — the same frame, now annotated.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        cx, cy = det["centroid"]

        # Green bounding box around each person
        cv2.rectangle(
            frame,
            (x1, y1), (x2, y2),
            config.COLOR_SAFE,
            config.BBOX_THICKNESS,
        )

        # Label: "Person 0.87" or "Driver 0.87" above the box
        label_prefix = "Driver" if det.get("in_vehicle") else "Person"
        label = f"{label_prefix} {conf:.2f}"

        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE, config.FONT_THICKNESS
        )
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 6),
            (x1 + text_w + 4, y1),
            config.COLOR_SAFE,
            cv2.FILLED,
        )
        cv2.putText(
            frame, label,
            (x1 + 2, y1 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            (255, 255, 255),
            config.FONT_THICKNESS,
        )

        # Draw skeleton if keypoints are available
        if det.get("keypoints"):
            draw_skeleton(frame, det["keypoints"])

        # Red filled circle at the centroid
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # Draw edge points as small cyan dots
        for (ex, ey) in det.get("edge_points", []):
            cv2.circle(frame, (ex, ey), 3, (255, 255, 0), -1)

    return frame


def draw_vehicles(frame, vehicles):
    """
    Draw bounding boxes and labels for detected vehicles.

    Args:
        frame    : numpy.ndarray — BGR frame to draw on.
        vehicles : list[dict]   — output from detect_vehicles().

    Returns:
        frame : numpy.ndarray — annotated frame.
    """
    for veh in vehicles:
        x1, y1, x2, y2 = veh["bbox"]
        conf = veh["confidence"]
        name = veh["class_name"]

        # Cyan bounding box for vehicles
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      config.COLOR_VEHICLE, config.BBOX_THICKNESS)

        label = f"{name} {conf:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE, config.FONT_THICKNESS
        )
        cv2.rectangle(frame,
                      (x1, y1 - th - baseline - 6),
                      (x1 + tw + 4, y1),
                      config.COLOR_VEHICLE, cv2.FILLED)
        cv2.putText(frame, label,
                    (x1 + 2, y1 - baseline - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    config.FONT_SCALE, (0, 0, 0),
                    config.FONT_THICKNESS)

    return frame


# ──────────────────────────────────────────────
#  SELF-TEST  (run:  python detector.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick standalone test for the detection module.

    What it does:
      1. Loads the YOLO pose model + vehicle model.
      2. Opens the video source defined in config.py.
      3. Runs person detection (with keypoints) on every frame.
      4. Runs vehicle detection and cross-references with persons.
      5. Draws bounding boxes, skeletons, centroids, and vehicles.
      6. Shows FPS and detection counts on screen.
      7. Prints each detection's details to the terminal.

    Controls:
      q — quit
      s — save a screenshot to the screenshots/ folder
    """
    import time
    import os

    print("=" * 55)
    print("  detector.py — Self-Test (Pose + Vehicle)")
    print("=" * 55)

    # Step 1: Load models
    pose_model = load_model()
    vehicle_model = load_vehicle_model()

    # Step 2: Open video source
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {config.VIDEO_SOURCE}")
        print("        If using webcam, check that it is connected.")
        print("        If using a file, check that the path is correct.")
        exit(1)

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[detector] Video opened: {config.VIDEO_SOURCE}")
    print(f"[detector] Resolution : {frame_width} x {frame_height}")
    print("[detector] Press 'q' to quit, 's' to save screenshot.\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[detector] End of video stream or camera disconnected.")
            break

        frame_count += 1
        start_time = time.time()

        # Detect persons with keypoints
        detections = detect_persons(pose_model, frame)

        # Detect vehicles
        vehicles = detect_vehicles(vehicle_model, frame)

        # Cross-reference: mark persons inside vehicles
        detections = detect_persons_in_vehicles(detections, vehicles)

        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # Print detection details to terminal (every 30th frame)
        if frame_count % 30 == 0 or len(detections) > 0:
            print(f"Frame {frame_count:>5d} | FPS {fps:5.1f} | "
                  f"Persons: {len(detections)} | Vehicles: {len(vehicles)}")
            for i, det in enumerate(detections):
                vehicle_tag = " [IN VEHICLE]" if det["in_vehicle"] else ""
                kp_count = sum(1 for kp in det["keypoints"]
                               if kp[2] >= config.KEYPOINT_CONFIDENCE_THRESHOLD) \
                    if det["keypoints"] else 0
                print(f"  └─ #{i+1}  bbox={det['bbox']}  "
                      f"conf={det['confidence']:.4f}  "
                      f"centroid={det['centroid']}  "
                      f"keypoints={kp_count}/17  "
                      f"edges={len(det['edge_points'])}"
                      f"{vehicle_tag}")

        # Draw detections onto the frame
        frame = draw_detections(frame, detections)
        frame = draw_vehicles(frame, vehicles)

        # HUD: FPS counter (top-left)
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2
        )

        # HUD: Counts
        cv2.putText(
            frame, f"Persons: {len(detections)}  Vehicles: {len(vehicles)}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
        )

        # Show frame
        cv2.imshow("Detector Self-Test", frame)

        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[detector] Quit by user.")
            break
        elif key == ord('s'):
            os.makedirs("screenshots", exist_ok=True)
            filename = f"screenshots/detector_frame_{frame_count}.png"
            cv2.imwrite(filename, frame)
            print(f"[detector] Screenshot saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("[detector] Self-test finished.")
