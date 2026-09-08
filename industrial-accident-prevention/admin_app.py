"""
admin_app.py — Flask Admin Dashboard Server
=============================================
Provides a web-based admin panel for the Industrial Accident
Prevention System with:
  1. Live CCTV video stream (MJPEG via /video_feed)
  2. Real-time worker status API (/api/workers)
  3. Danger zone management API (/api/zones — CRUD)
  4. Alert log API (/api/logs)
  5. System stats API (/api/stats)

The Flask server runs in a BACKGROUND THREAD alongside the main
OpenCV processing loop. It reads data from shared_state.py (the
thread-safe singleton) and never touches the camera directly.

Routes:
  GET  /             — Main dashboard HTML page
  GET  /video_feed   — MJPEG video stream (img tag src)
  GET  /api/workers  — Current worker statuses (JSON)
  GET  /api/zones    — Current zone configurations (JSON)
  POST /api/zones    — Save new zone configuration (JSON)
  DELETE /api/zones/<id> — Delete a specific zone
  GET  /api/logs     — Recent alert history (JSON)
  GET  /api/stats    — System statistics (JSON)
  GET  /api/workers/history — Worker movement history (JSON)
"""

import cv2
import json
import time
import threading
import numpy as np
from flask import Flask, Response, render_template, request, jsonify

import config
import zones
from shared_state import state


# ──────────────────────────────────────────────
#  FLASK APP SETUP
# ──────────────────────────────────────────────

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')
app.secret_key = config.ADMIN_SECRET_KEY


# ──────────────────────────────────────────────
#  MAIN DASHBOARD PAGE
# ──────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the main admin dashboard HTML page."""
    return render_template('admin.html')


# ──────────────────────────────────────────────
#  MJPEG VIDEO STREAM
# ──────────────────────────────────────────────

def _generate_mjpeg():
    """
    Generator function that yields MJPEG frames for streaming.

    Reads the latest frame from shared_state, encodes it as JPEG,
    and yields it in the multipart/x-mixed-replace format that
    browsers can display in an <img> tag.
    """
    while True:
        frame = state.get_frame()

        if frame is not None:
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       frame_bytes + b'\r\n')

        # ~15 FPS for the stream (enough for dashboard monitoring)
        time.sleep(0.066)


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream endpoint for the live CCTV feed."""
    return Response(
        _generate_mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


# ──────────────────────────────────────────────
#  API: WORKER STATUS
# ──────────────────────────────────────────────

@app.route('/api/workers')
def api_workers():
    """
    Return current tracked workers with their statuses.

    Response JSON:
    {
      "workers": [
        {
          "worker_id": 1,
          "status": "SAFE",
          "centroid": [350, 240],
          "confidence": 0.87,
          "in_vehicle": false,
          "vehicle_info": null,
          "edge_points_count": 13,
          "time_in_zone": "00:02:15"
        }
      ],
      "count": 1,
      "timestamp": 1694100000.0
    }
    """
    workers_raw = state.get_workers()
    history = state.get_worker_history()
    current_time = time.time()

    workers = []
    for w in workers_raw:
        wid = w.get("worker_id")
        status = w.get("status", "SAFE")

        # Calculate time in current zone
        time_in_zone = ""
        if wid in history:
            entered = history[wid].get("entered_at", current_time)
            duration = current_time - entered
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            time_in_zone = f"{minutes:02d}:{seconds:02d}"

        workers.append({
            "worker_id": wid,
            "status": status,
            "centroid": list(w.get("centroid", (0, 0))),
            "bbox": list(w.get("bbox", (0, 0, 0, 0))),
            "confidence": w.get("confidence", 0),
            "in_vehicle": w.get("in_vehicle", False),
            "vehicle_info": w.get("vehicle_info"),
            "edge_points_count": len(w.get("edge_points", [])),
            "time_in_zone": time_in_zone,
        })

    return jsonify({
        "workers": workers,
        "count": len(workers),
        "timestamp": current_time,
    })


# ──────────────────────────────────────────────
#  API: WORKER HISTORY
# ──────────────────────────────────────────────

@app.route('/api/workers/history')
def api_worker_history():
    """Return movement history for all tracked workers."""
    history = state.get_worker_history()

    # Convert to JSON-serializable format
    result = {}
    for wid, data in history.items():
        result[str(wid)] = {
            "current_status": data.get("status", "SAFE"),
            "entered_at": data.get("entered_at", 0),
            "transitions": data.get("history", []),
        }

    return jsonify({
        "history": result,
        "timestamp": time.time(),
    })


# ──────────────────────────────────────────────
#  API: ZONE MANAGEMENT
# ──────────────────────────────────────────────

@app.route('/api/zones', methods=['GET'])
def api_get_zones():
    """
    Return current zone configurations.

    Response JSON:
    {
      "zones": [
        {
          "id": "zone_1",
          "type": "DANGER",
          "name": "Machine Area",
          "points": [[280,200],[420,200],[420,350],[280,350]],
          "created_at": "2026-09-07T21:00:00"
        }
      ]
    }
    """
    active_zones = zones.get_active_zones()

    zones_data = []
    for z in active_zones:
        pts = z["points"]
        if isinstance(pts, np.ndarray):
            pts = pts.tolist()
        zones_data.append({
            "id": z["id"],
            "type": z["type"],
            "name": z.get("name", ""),
            "points": pts,
            "created_at": z.get("created_at", ""),
        })

    return jsonify({"zones": zones_data})


@app.route('/api/zones', methods=['POST'])
def api_save_zones():
    """
    Save new zone configuration from the admin's zone drawing tool.

    Request JSON:
    {
      "zones": [
        {
          "id": "zone_1",
          "type": "DANGER",
          "name": "Machine Area",
          "points": [[280,200],[420,200],[420,350],[280,350]]
        }
      ]
    }
    """
    try:
        data = request.get_json()
        if not data or "zones" not in data:
            return jsonify({"error": "Missing 'zones' field"}), 400

        new_zones = data["zones"]

        # Validate each zone
        for z in new_zones:
            if "points" not in z or len(z["points"]) < 3:
                return jsonify({
                    "error": f"Zone '{z.get('id', '?')}' must have at least 3 points"
                }), 400

            if z.get("type") not in ("WARNING", "DANGER"):
                return jsonify({
                    "error": f"Zone type must be 'WARNING' or 'DANGER'"
                }), 400

        # Save to file and reload
        zones.save_zones_to_file(new_zones)
        zones.load_zones()

        return jsonify({
            "status": "ok",
            "message": f"Saved {len(new_zones)} zones",
            "zones_count": len(new_zones),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/zones/<zone_id>', methods=['DELETE'])
def api_delete_zone(zone_id):
    """
    Delete a specific zone by ID.

    After deletion, the remaining zones are saved to file.
    """
    active_zones = zones.get_active_zones()

    # Find and remove the zone
    updated_zones = [z for z in active_zones if z.get("id") != zone_id]

    if len(updated_zones) == len(active_zones):
        return jsonify({"error": f"Zone '{zone_id}' not found"}), 404

    # Save updated zones
    zones.save_zones_to_file(updated_zones)
    zones.load_zones()

    return jsonify({
        "status": "ok",
        "message": f"Deleted zone '{zone_id}'",
        "remaining": len(updated_zones),
    })


# ──────────────────────────────────────────────
#  API: ALERT LOG
# ──────────────────────────────────────────────

@app.route('/api/logs')
def api_logs():
    """Return recent alert log entries."""
    limit = request.args.get('limit', 50, type=int)
    alerts = state.get_alerts(limit=limit)

    return jsonify({
        "alerts": alerts,
        "count": len(alerts),
        "timestamp": time.time(),
    })


# ──────────────────────────────────────────────
#  API: SYSTEM STATS
# ──────────────────────────────────────────────

@app.route('/api/stats')
def api_stats():
    """Return system statistics."""
    stats = state.get_stats()
    return jsonify(stats)


# ──────────────────────────────────────────────
#  SERVER STARTUP (background thread)
# ──────────────────────────────────────────────

def start_admin_server():
    """
    Start the Flask admin server in a background daemon thread.

    This is called by main.py at startup. The server runs on
    ADMIN_HOST:ADMIN_PORT from config.py (default: 0.0.0.0:5000).
    """
    def _run_server():
        # Suppress Flask's default startup banner in production
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.WARNING)

        print(f"\n[admin] Dashboard starting at "
              f"http://localhost:{config.ADMIN_PORT}")
        print(f"[admin] Network access at "
              f"http://0.0.0.0:{config.ADMIN_PORT}\n")

        app.run(
            host=config.ADMIN_HOST,
            port=config.ADMIN_PORT,
            debug=False,
            threaded=True,
            use_reloader=False,  # important: don't restart in bg thread
        )

    server_thread = threading.Thread(target=_run_server, daemon=True)
    server_thread.start()
    return server_thread


# ──────────────────────────────────────────────
#  STANDALONE ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  admin_app.py — Standalone Admin Dashboard")
    print("=" * 55)
    print()
    print("NOTE: For full functionality, run main.py instead.")
    print("      This standalone mode shows the dashboard UI")
    print("      but the video feed and worker data require")
    print("      the CV pipeline running in main.py.")
    print()

    app.run(
        host=config.ADMIN_HOST,
        port=config.ADMIN_PORT,
        debug=True,
    )
