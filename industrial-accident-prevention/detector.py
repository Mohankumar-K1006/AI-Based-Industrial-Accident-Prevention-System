"""
detector.py — Person Detection Module
======================================
Responsibilities:
  1. Load the YOLO model once at startup.
  2. Run inference on each video frame.
  3. Filter results to keep only "person" detections above the
     confidence threshold defined in config.py.
  4. Return a clean list of detections, each containing:
       - bounding box  (x1, y1, x2, y2)
       - confidence     (float 0–1)
       - class_id       (int, always 0 for "person" in COCO)
       - centroid       (cx, cy)
  5. Provide a draw_detections() helper so main.py can render
     bounding boxes, centroids, and labels in one call.

This module does NOT handle tracking or zones — those are
separate modules (tracker.py, zones.py).
"""

import cv2
from ultralytics import YOLO
import config


# ──────────────────────────────────────────────
#  MODEL LOADING
# ──────────────────────────────────────────────

def load_model():
    """
    Load the YOLO model specified in config.py.

    Returns:
        model : ultralytics.YOLO  — the loaded YOLO model object.

    Notes:
        - On the FIRST run, Ultralytics auto-downloads the weights
          file (e.g. yolov8n.pt ≈ 6 MB) into the working directory.
        - After downloading, the model is cached locally.
        - The model object should be loaded ONCE and reused for
          every frame to avoid repeated disk I/O.
    """
    model = YOLO(config.YOLO_MODEL_PATH)
    print(f"[detector] YOLO model loaded: {config.YOLO_MODEL_PATH}")
    return model


# ──────────────────────────────────────────────
#  PERSON DETECTION
# ──────────────────────────────────────────────

def detect_persons(model, frame):
    """
    Run YOLO inference on a single BGR frame and return only
    'person' detections that meet the confidence threshold.

    Args:
        model : ultralytics.YOLO  — the pre-loaded YOLO model.
        frame : numpy.ndarray     — a BGR image from OpenCV.

    Returns:
        detections : list[dict]
            Each dict contains:
                "bbox"       : (x1, y1, x2, y2) — top-left & bottom-right pixel coords
                "confidence" : float              — YOLO confidence score (0.0 – 1.0)
                "class_id"   : int                — COCO class ID (0 = person)
                "centroid"   : (cx, cy)           — center point of bounding box

    How the centroid is calculated:
        cx = (x1 + x2) // 2   → horizontal center
        cy = (y1 + y2) // 2   → vertical center
        This gives the pixel coordinate at the geometric center of
        the bounding box, which later modules use for zone checks.
    """
    # ── Run YOLO inference ──
    # conf  : only return detections above this threshold
    # verbose=False : suppress per-frame console output from YOLO
    results = model(frame, conf=config.YOLO_CONFIDENCE_THRESHOLD, verbose=False)

    detections = []

    # `results` is a list (one entry per image). We sent one frame,
    # so we access results[0].  Each `box` object contains:
    #   box.xyxy  → tensor of shape [1, 4] with [x1, y1, x2, y2]
    #   box.conf  → tensor of shape [1]    with confidence
    #   box.cls   → tensor of shape [1]    with class ID
    for box in results[0].boxes:

        # ── Extract class ID ──
        class_id = int(box.cls[0])

        # ── Filter: keep ONLY person detections (class 0 in COCO) ──
        if class_id != config.PERSON_CLASS_ID:
            continue

        # ── Bounding box coordinates (convert tensor → Python ints) ──
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # ── Confidence score (convert tensor → Python float) ──
        confidence = round(float(box.conf[0]), 4)

        # ── Centroid calculation ──
        # The centroid is the geometric center of the bounding box.
        #   cx = midpoint along x-axis
        #   cy = midpoint along y-axis
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        detections.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": confidence,
            "class_id": class_id,
            "centroid": (cx, cy),
        })

    return detections


# ──────────────────────────────────────────────
#  DRAWING HELPER (used by main.py)
# ──────────────────────────────────────────────

def draw_detections(frame, detections):
    """
    Draw bounding boxes, centroid dots, and confidence labels
    onto the frame for every detection.

    Args:
        frame      : numpy.ndarray — the BGR frame to draw on (modified in place).
        detections : list[dict]    — output from detect_persons().

    Returns:
        frame : numpy.ndarray — the same frame, now annotated.

    This is a convenience function so that main.py can render
    detector results with a single call:
        frame = draw_detections(frame, detections)
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        conf = det["confidence"]
        cx, cy = det["centroid"]

        # ── Green bounding box around each person ──
        cv2.rectangle(
            frame,
            (x1, y1), (x2, y2),
            config.COLOR_SAFE,             # green by default
            config.BBOX_THICKNESS,
        )

        # ── Label: "Person 0.87" above the box ──
        label = f"Person {conf:.2f}"

        # Calculate text size so we can draw a filled background rectangle.
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE, config.FONT_THICKNESS
        )
        # Background rectangle for the label (improves readability).
        cv2.rectangle(
            frame,
            (x1, y1 - text_h - baseline - 6),
            (x1 + text_w + 4, y1),
            config.COLOR_SAFE,
            cv2.FILLED,
        )
        # White text on green background.
        cv2.putText(
            frame, label,
            (x1 + 2, y1 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            (255, 255, 255),               # white text
            config.FONT_THICKNESS,
        )

        # ── Red filled circle at the centroid ──
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    return frame


# ──────────────────────────────────────────────
#  SELF-TEST  (run:  python detector.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    """
    Quick standalone test for the detection module.

    What it does:
      1. Loads the YOLO model.
      2. Opens the video source defined in config.py.
      3. Runs person detection on every frame.
      4. Draws bounding boxes, centroids, and confidence.
      5. Shows FPS and person count on screen.
      6. Prints each detection's details to the terminal.

    Controls:
      q — quit
      s — save a screenshot to the screenshots/ folder
    """
    import time
    import os

    print("=" * 55)
    print("  detector.py — Self-Test")
    print("=" * 55)

    # Step 1: Load YOLO model
    model = load_model()

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

        # ── Detect persons ──
        detections = detect_persons(model, frame)

        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # ── Print detection details to terminal (every 30th frame) ──
        if frame_count % 30 == 0 or len(detections) > 0:
            print(f"Frame {frame_count:>5d} | FPS {fps:5.1f} | "
                  f"Persons detected: {len(detections)}")
            for i, det in enumerate(detections):
                print(f"  └─ #{i+1}  bbox={det['bbox']}  "
                      f"conf={det['confidence']:.4f}  "
                      f"centroid={det['centroid']}")

        # ── Draw detections onto the frame ──
        frame = draw_detections(frame, detections)

        # ── HUD: FPS counter (top-left) ──
        cv2.putText(
            frame, f"FPS: {fps:.1f}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2
        )

        # ── HUD: Person count (below FPS) ──
        cv2.putText(
            frame, f"Persons: {len(detections)}", (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2
        )

        # ── Show frame ──
        cv2.imshow("Detector Self-Test", frame)

        # ── Keyboard controls ──
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
