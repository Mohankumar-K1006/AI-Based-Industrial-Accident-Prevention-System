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

## License

Academic / Educational use.
