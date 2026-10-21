# AI-Based Industrial Accident Prevention System

**Real-Time Danger Zone Detection and Worker Tracking**

## Overview

A computer vision system that detects workers using YOLO, tracks them across
video frames, maps their positions to configurable safety zones, and generates
warnings when workers enter hazardous areas.

## Technology Stack

| Component        | Technology            |
|------------------|-----------------------|
| Language         | Python 3.9+           |
| Object Detection | Ultralytics YOLOv8    |
| Video Processing | OpenCV                |
| Computation      | NumPy                 |

## Project Structure

```
industrial-accident-prevention/
├── main.py            # Application entry point — captures video, orchestrates pipeline
├── detector.py        # YOLO-based person detection module
├── tracker.py         # Simple centroid-based worker ID tracker
├── zones.py           # Polygon zone definitions and point-in-zone logic
├── alerts.py          # Audible and visual alert generation
├── config.py          # All configurable parameters (edit this!)
├── requirements.txt   # Python dependencies
├── models/            # YOLO weight files (auto-downloaded on first run)
├── videos/            # Input test videos
├── screenshots/       # Saved screenshots for documentation
├── output/            # Recorded output videos
└── README.md          # This file
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Place a test video in videos/ and update VIDEO_SOURCE in config.py

# 3. Run the system
python main.py
```

## Controls

| Key | Action             |
|-----|--------------------|
| `q` | Quit the application |
| `s` | Save a screenshot    |

## Configuration

Edit `config.py` to change:
- Video source (webcam index or file path)
- YOLO model and confidence threshold
- Warning / Danger zone polygon coordinates
- Alert frequency and cooldown
- Display colours and font sizes

## Module Responsibilities

| Module         | Responsibility |
|----------------|----------------|
| `config.py`    | Stores all tuneable parameters in one place |
| `detector.py`  | Loads YOLO model, runs inference, returns person detections |
| `tracker.py`   | Assigns persistent IDs to workers across consecutive frames |
| `zones.py`     | Checks if a worker centroid is inside warning/danger polygons |
| `alerts.py`    | Plays audible beeps and draws on-screen warning messages |
| `main.py`      | Captures video frames and connects all modules together |

## Face Recognition — Employee Identification

The system can identify **which employee** entered a danger zone using face recognition.

### Enrollment
```bash
streamlit run enroll_dashboard.py
```
Upload 2–3 clear photos per employee. Encodings are stored in `face_encodings.pkl`; metadata in `employees.json`.

### Runtime
When a DANGER entry is detected, the person's bounding box is cropped and matched against all enrolled encodings in a **background thread** (never blocks the video loop). If matched, the employee receives an emergency email at their personal address. The zone manager **also** still receives their usual alert.

### Testing (no camera needed)
```bash
python test_face_match.py                    # verifies zero-enrollment safety
python test_face_match.py path/to/photo.jpg  # checks match against enrolled data
```

---

## ⚠️ Privacy & Consent Notice

> **This section is mandatory reading for reviewers and examiners.**

1. **Informed consent is required.** Storing biometric face data (even as numeric
   encodings, not raw photos) requires each employee's explicit, written consent
   before enrollment.
2. **Encodings, not photos.** At runtime the system matches against 128-dimensional
   numeric vectors (`face_encodings.pkl`), not raw images. Uploaded photos are used
   only during enrollment to compute these vectors and are **not** stored on disk
   by the enrollment tool.
3. **Prototype limitations.** This implementation lacks access controls (anyone with
   filesystem access can read the encoding file), encryption at rest, a data
   retention/deletion policy, and audit logging for who accessed the data. A
   production deployment **must** add these safeguards before handling real employee
   biometrics.
4. **Applicable regulations.** Depending on jurisdiction, biometric data may fall
   under GDPR (EU), the IT Act (India), BIPA (Illinois, USA), or equivalent
   legislation. Consult legal counsel before deployment.

---

## License

Academic / Educational use.
