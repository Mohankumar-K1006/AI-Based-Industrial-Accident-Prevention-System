"""
config.py — Central Configuration File
=======================================
All tuneable parameters for the accident prevention system live here.
Edit this file to change video source, zone coordinates, thresholds, etc.
No other file should contain hardcoded magic numbers.
"""

import numpy as np

# ──────────────────────────────────────────────
#  VIDEO SOURCE
# ──────────────────────────────────────────────
# Use 0 for the default webcam, or provide a path to a video file.
# Examples:
#   VIDEO_SOURCE = 0                        # webcam
#   VIDEO_SOURCE = "videos/factory.mp4"     # local video file
VIDEO_SOURCE = 0

# ──────────────────────────────────────────────
#  YOLO MODEL SETTINGS
# ──────────────────────────────────────────────
# Path to the YOLO pose model weights file for skeleton/keypoint detection.
# The first run will auto-download "yolov8n-pose.pt" (~6 MB) if not found.
YOLO_POSE_MODEL_PATH = "yolov8n-pose.pt"

# Path to the standard YOLO model for vehicle detection.
# Used alongside the pose model to detect cars, trucks, buses, forklifts.
YOLO_VEHICLE_MODEL_PATH = "yolov8n.pt"

# Legacy alias — kept for backward compatibility with self-test scripts.
YOLO_MODEL_PATH = YOLO_POSE_MODEL_PATH

# Minimum confidence score to accept a detection (0.0 – 1.0).
YOLO_CONFIDENCE_THRESHOLD = 0.45

# COCO class ID for "person" (do not change unless using a custom model).
PERSON_CLASS_ID = 0

# ──────────────────────────────────────────────
#  PERFORMANCE & FPS OPTIMIZATION SETTINGS
# ──────────────────────────────────────────────
# YOLO inference input resolution (smaller = dramatically higher FPS on CPU).
# 320 gives ~2x speedup compared to standard 640 while maintaining accurate detection.
YOLO_IMG_SIZE = 320

# Run YOLO pose detection every N frames (skipped frames reuse tracked workers).
# Set to 1 for detection on every frame, or 2 for ~2x FPS boost.
DETECTION_INTERVAL = 2

# Vehicle detection interval in frames.
# Running every 5 frames saves ~160ms on 80% of frames while keeping vehicle tags fresh.
VEHICLE_DETECTION_INTERVAL = 5

# Automatic dark-frame detection threshold (0-255).
# If average frame brightness is below this value, an on-screen warning is shown
# alerting the user that the camera is blocked or the privacy shutter is closed.
CAMERA_DARK_THRESHOLD = 3.0


# ──────────────────────────────────────────────
#  SKELETON / KEYPOINT SETTINGS
# ──────────────────────────────────────────────
# Toggle skeleton visualisation on/off.
SKELETON_DRAW = True

# Minimum confidence for a keypoint to be considered valid (0.0 – 1.0).
# Keypoints below this threshold are ignored in zone checks and drawing.
KEYPOINT_CONFIDENCE_THRESHOLD = 0.3

# COCO Pose keypoint names (17 total):
#   0  Nose              5  Left Shoulder    10 Right Wrist
#   1  Left Eye          6  Right Shoulder   11 Left Hip
#   2  Right Eye         7  Left Elbow       12 Right Hip
#   3  Left Ear          8  Right Elbow      13 Left Knee
#   4  Right Ear         9  Left Wrist       14 Right Knee
#                                             15 Left Ankle
#                                             16 Right Ankle

# Indices of keypoints considered "body edges" for zone boundary checks.
# If ANY of these points enters a zone, the worker is classified into that zone.
EDGE_KEYPOINT_INDICES = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Skeleton connections for drawing (pairs of keypoint indices).
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),        # Nose → Eyes
    (1, 3), (2, 4),        # Eyes → Ears
    (5, 6),                # Shoulder → Shoulder
    (5, 7), (7, 9),        # Left Shoulder → Elbow → Wrist
    (6, 8), (8, 10),       # Right Shoulder → Elbow → Wrist
    (5, 11), (6, 12),      # Shoulders → Hips
    (11, 12),              # Hip → Hip
    (11, 13), (13, 15),    # Left Hip → Knee → Ankle
    (12, 14), (14, 16),    # Right Hip → Knee → Ankle
]

# Colours for skeleton drawing (BGR)
SKELETON_LINE_COLOR = (255, 200, 50)    # Light blue
SKELETON_JOINT_COLOR = (0, 255, 128)    # Green-yellow
SKELETON_LINE_THICKNESS = 2
SKELETON_JOINT_RADIUS = 4

# ──────────────────────────────────────────────
#  VEHICLE DETECTION SETTINGS
# ──────────────────────────────────────────────
# Enable/disable vehicle detection.
DETECT_VEHICLES = True

# COCO class IDs for vehicles of interest.
#   2 = car,  5 = bus,  7 = truck
VEHICLE_CLASS_IDS = [2, 5, 7]

# COCO class names for display (indexed by class ID).
VEHICLE_CLASS_NAMES = {2: "Car", 5: "Bus", 7: "Truck"}

# Minimum bounding-box overlap ratio to consider a person "inside" a vehicle.
# Uses the fraction of the person bbox area that overlaps with the vehicle bbox.
VEHICLE_OVERLAP_THRESHOLD = 0.4

# Colour for vehicle bounding boxes (BGR).
COLOR_VEHICLE = (255, 200, 0)    # Cyan

# ──────────────────────────────────────────────
#  TRACKER SETTINGS
# ──────────────────────────────────────────────
# Maximum pixel distance between centroids in consecutive frames
# for them to be considered the same worker.
MAX_TRACKING_DISTANCE = 80

# Number of consecutive frames a worker can be missing before
# their ID is removed from the tracker.
MAX_FRAMES_MISSING = 30

# ──────────────────────────────────────────────
#  SAFETY ZONES (polygon vertices)
# ──────────────────────────────────────────────
# Each zone is a list of (x, y) points forming a closed polygon.
# Coordinates are in pixels and depend on your camera resolution.
#
# *** IMPORTANT ***
# You MUST adjust these coordinates to match your video/camera feed.
# Use the zone calibration helper (zones.py --calibrate) to pick points.
#
# The WARNING zone is the outer boundary — workers get a caution alert.
# The DANGER zone is the inner/critical area — workers get a loud alarm.
#
# NOTE: These are DEFAULT zones. If zones_config.json exists (created by
# the admin dashboard), those zones will OVERRIDE these defaults.

WARNING_ZONE = np.array([
    [200, 150],
    [500, 150],
    [500, 400],
    [200, 400],
], dtype=np.int32)

DANGER_ZONE = np.array([
    [280, 200],
    [420, 200],
    [420, 350],
    [280, 350],
], dtype=np.int32)

# Path to the JSON file where admin-defined zones are persisted.
ZONES_CONFIG_FILE = "zones_config.json"

# ──────────────────────────────────────────────
#  ZONE DISPLAY COLOURS  (BGR format for OpenCV)
# ──────────────────────────────────────────────
COLOR_SAFE    = (0, 200, 0)      # Green
COLOR_WARNING = (0, 200, 255)    # Orange / Yellow
COLOR_DANGER  = (0, 0, 255)      # Red

# Overlay transparency for zone polygons (0.0 = invisible, 1.0 = opaque).
ZONE_OVERLAY_ALPHA = 0.25

# ──────────────────────────────────────────────
#  BOUNDING BOX & TEXT DISPLAY
# ──────────────────────────────────────────────
BBOX_THICKNESS = 2
FONT_SCALE     = 0.55
FONT_THICKNESS = 2

# ──────────────────────────────────────────────
#  ALERT SETTINGS
# ──────────────────────────────────────────────
# Frequency (Hz) and duration (ms) for the audible beep on Windows.
ALERT_BEEP_FREQUENCY = 1000      # Hz
ALERT_BEEP_DURATION  = 200       # milliseconds

# Minimum gap (seconds) between consecutive audible alerts so the
# beep does not fire on every single frame.
ALERT_COOLDOWN = 2.0

# ──────────────────────────────────────────────
#  DISPLAY WINDOW
# ──────────────────────────────────────────────
WINDOW_NAME = "AI Accident Prevention System"

# ──────────────────────────────────────────────
#  ADMIN DASHBOARD SETTINGS (Flask)
# ──────────────────────────────────────────────
ADMIN_HOST = "0.0.0.0"          # Accessible from any device on the network
ADMIN_PORT = 5000
ADMIN_SECRET_KEY = os.getenv("ADMIN_SECRET_KEY", "change-this-secret-before-deployment")
ADMIN_ENABLED = True            # Set to False to disable the web dashboard
