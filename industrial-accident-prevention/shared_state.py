"""
shared_state.py — Thread-Safe Shared State
============================================
Provides a centralised, thread-safe state container shared between:
  - The OpenCV processing loop (main.py) — WRITES frame data, worker info
  - The Flask admin dashboard (admin_app.py) — READS frame data, worker info

Uses threading.Lock to prevent race conditions when the CV loop writes
data while the Flask server reads it for MJPEG streaming and API responses.

State contents:
  - current_frame  : latest BGR frame from the CV pipeline
  - workers        : list of current tracked workers with status
  - vehicles       : list of detected vehicles
  - alert_log      : rolling list of recent alerts (last 200)
  - zones          : current zone configurations
  - system_stats   : FPS, frame count, uptime
"""

import threading
import time
import copy


class SharedState:
    """
    Thread-safe shared state between CV pipeline and Flask dashboard.

    All public methods acquire the lock before reading/writing,
    ensuring safe concurrent access from multiple threads.
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Latest video frame (numpy array, BGR)
        self._current_frame = None

        # Current tracked workers with status
        # list of dicts: { worker_id, centroid, bbox, status, confidence,
        #                   edge_points, keypoints, in_vehicle, vehicle_info }
        self._workers = []

        # Detected vehicles
        self._vehicles = []

        # Alert log — rolling list of recent alerts
        # list of dicts: { timestamp, worker_id, status, message }
        self._alert_log = []
        self._max_alerts = 200

        # System stats
        self._fps = 0.0
        self._frame_count = 0
        self._start_time = time.time()

        # Worker history — tracks each worker's zone entry/exit times
        # { worker_id: { "status": str, "entered_at": float,
        #                "history": [ {"status": str, "at": float}, ... ] } }
        self._worker_history = {}

    # ──────────────────────────────────────────
    #  FRAME DATA (written by CV loop, read by Flask)
    # ──────────────────────────────────────────

    def set_frame(self, frame):
        """Store the latest frame from the CV pipeline."""
        with self._lock:
            self._current_frame = frame

    def get_frame(self):
        """Get a copy of the latest frame. Returns None if no frame yet."""
        with self._lock:
            if self._current_frame is not None:
                return self._current_frame.copy()
            return None

    # ──────────────────────────────────────────
    #  WORKER DATA
    # ──────────────────────────────────────────

    def set_workers(self, workers_with_status):
        """
        Update the current worker list.

        Args:
            workers_with_status : list[dict] — each dict must have:
                worker_id, centroid, bbox, status, confidence,
                edge_points, in_vehicle, vehicle_info
        """
        with self._lock:
            self._workers = [dict(w) for w in workers_with_status]

            # Update worker history
            current_time = time.time()
            for w in workers_with_status:
                wid = w["worker_id"]
                status = w["status"]

                if wid not in self._worker_history:
                    self._worker_history[wid] = {
                        "status": status,
                        "entered_at": current_time,
                        "history": [],
                    }
                else:
                    prev_status = self._worker_history[wid]["status"]
                    if status != prev_status:
                        # Status changed — record in history
                        self._worker_history[wid]["history"].append({
                            "from_status": prev_status,
                            "to_status": status,
                            "at": current_time,
                        })
                        self._worker_history[wid]["status"] = status
                        self._worker_history[wid]["entered_at"] = current_time

                        # Keep only last 50 history entries per worker
                        if len(self._worker_history[wid]["history"]) > 50:
                            self._worker_history[wid]["history"] = \
                                self._worker_history[wid]["history"][-50:]

    def get_workers(self):
        """Get a copy of the current worker list."""
        with self._lock:
            return copy.deepcopy(self._workers)

    def get_worker_history(self, worker_id=None):
        """
        Get worker movement history.

        Args:
            worker_id : int or None — specific worker, or all workers.

        Returns:
            dict or list of dicts with history data.
        """
        with self._lock:
            if worker_id is not None:
                return copy.deepcopy(
                    self._worker_history.get(worker_id, {})
                )
            return copy.deepcopy(self._worker_history)

    # ──────────────────────────────────────────
    #  VEHICLE DATA
    # ──────────────────────────────────────────

    def set_vehicles(self, vehicles):
        """Store the current vehicle detections."""
        with self._lock:
            self._vehicles = copy.deepcopy(vehicles)

    def get_vehicles(self):
        """Get a copy of current vehicle detections."""
        with self._lock:
            return copy.deepcopy(self._vehicles)

    # ──────────────────────────────────────────
    #  ALERT LOG
    # ──────────────────────────────────────────

    def __init_alert_dedup(self):
        """Lazy-init the alert dedup tracker."""
        if not hasattr(self, '_alert_last_logged'):
            self._alert_last_logged = {}  # "worker_id::status" -> timestamp
            self._alert_dedup_secs = 10.0  # min seconds between same alert

    def add_alert(self, worker_id, status, message, employee_id=None, display_name=None):
        """
        Add an alert entry to the log with per-worker de-duplication.

        Prevents the same worker+status from flooding the log every frame.
        Each unique worker+status combo is logged at most once every 10 seconds.

        Args:
            worker_id    : int
            status       : str — "WARNING" or "DANGER"
            message      : str — alert description
            employee_id  : str or None — matched employee ID
            display_name : str or None — matched employee name
        """
        with self._lock:
            self.__init_alert_dedup()

            # De-duplicate: skip if same worker+status was logged recently
            dedup_key = f"{worker_id}::{status}"
            now = time.time()
            last_logged = self._alert_last_logged.get(dedup_key, 0.0)
            if now - last_logged < self._alert_dedup_secs:
                # Update display_name if a better one became available.
                # This handles the case where a worker initially enters
                # a zone as "Worker N" (unidentified) and face recognition
                # later resolves their real name.
                # Retroactively upgrade "Unknown Person" → enrolled name when
                # face recognition resolves the identity after the alert was posted.
                if employee_id and display_name:
                    for alert in reversed(self._alert_log):
                        if alert["worker_id"] == worker_id and alert["status"] == status:
                            old_eid = alert.get("employee_id")
                            if not old_eid:
                                # Upgrade from Unknown Person / Worker N to real name
                                alert["display_name"] = display_name
                                alert["employee_id"]  = employee_id
                                alert["message"]      = message
                            break
                return

            self._alert_last_logged[dedup_key] = now

            self._alert_log.append({
                "timestamp": now,
                "time_str": time.strftime("%H:%M:%S"),
                "worker_id": worker_id,
                "status": status,
                "message": message,
                "employee_id": employee_id,
                "display_name": display_name,
            })
            # Trim to max size
            if len(self._alert_log) > self._max_alerts:
                self._alert_log = self._alert_log[-self._max_alerts:]

    def get_alerts(self, limit=50):
        """Get the most recent alerts."""
        with self._lock:
            return copy.deepcopy(self._alert_log[-limit:])

    # ──────────────────────────────────────────
    #  SYSTEM STATS
    # ──────────────────────────────────────────

    def set_stats(self, fps, frame_count):
        """Update system statistics."""
        with self._lock:
            self._fps = fps
            self._frame_count = frame_count

    def get_stats(self):
        """Get current system statistics."""
        with self._lock:
            uptime = time.time() - self._start_time
            return {
                "fps": round(self._fps, 1),
                "frame_count": self._frame_count,
                "uptime_seconds": round(uptime, 1),
                "uptime_str": self._format_uptime(uptime),
                "total_workers_tracked": len(self._worker_history),
                "active_workers": len(self._workers),
                "total_alerts": len(self._alert_log),
            }

    @staticmethod
    def _format_uptime(seconds):
        """Format seconds into HH:MM:SS string."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# ──────────────────────────────────────────────
#  GLOBAL SINGLETON INSTANCE
# ──────────────────────────────────────────────
# This is the single shared state instance used by both
# main.py (writer) and admin_app.py (reader).

state = SharedState()
