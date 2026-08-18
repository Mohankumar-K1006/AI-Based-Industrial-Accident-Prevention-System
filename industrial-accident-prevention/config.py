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
# Path to the YOLO model weights file.
# The first run will auto-download "yolov8n.pt" (~6 MB) if not found.
YOLO_MODEL_PATH = "yolov8n.pt"

# Minimum confidence score to accept a detection (0.0 – 1.0).
YOLO_CONFIDENCE_THRESHOLD = 0.45

# COCO class ID for "person" (do not change unless using a custom model).
PERSON_CLASS_ID = 0

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
