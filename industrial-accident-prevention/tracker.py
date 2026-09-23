"""
tracker.py — Centroid-Based Worker Tracker (with Keypoints)
============================================================
Responsibilities:
  1. Accept a list of person detections (from detector.py) each frame.
  2. Assign a stable Worker ID to each person across consecutive frames.
  3. Match current-frame detections to previous-frame tracks using
     Euclidean distance between centroids.
  4. Create new IDs for workers who appear for the first time.
  5. Remove IDs for workers who disappear for too many frames.
  6. Store and forward keypoints + edge_points alongside centroid/bbox.
  7. Return an enriched list with worker_id added to each detection.

Changes from v1:
  - Each track now stores "keypoints" and "edge_points" alongside
    the existing centroid/bbox/confidence.
  - _build_output() includes these new fields so that main.py and
    zones.py can use multi-point zone checking.
  - Tracking STILL uses centroid distance (proven, fast approach).
  - Also stores "in_vehicle" and "vehicle_info" for vehicle awareness.

Algorithm overview (simple centroid tracker):
  ┌─────────────────────────────────────────────────────────────────┐
  │  Frame N-1 tracks:  {ID:1 → (100,200),  ID:2 → (300,400)}     │
  │  Frame N detections:       [(105,198),          (298,405)]     │
  │                                                                 │
  │  Step 1: Compute Euclidean distance from EVERY old centroid     │
  │          to EVERY new centroid → build a distance table.        │
  │  Step 2: Sort all pairs by distance (smallest first).           │
  │  Step 3: Greedily match the closest pair, remove both from the  │
  │          candidate pools, repeat.                               │
  │  Step 4: Unmatched new detections → assign fresh IDs.           │
  │  Step 5: Unmatched old tracks → increment "missing" counter;    │
  │          if missing too long → delete the track.                │
  └─────────────────────────────────────────────────────────────────┘

This module does NOT use DeepSORT or any external tracking library.
All matching is done with basic Euclidean geometry + a greedy sort.
"""

import math
import config


class CentroidTracker:
    """
    A simple centroid-based multi-object tracker.

    Attributes:
        next_worker_id (int):
            The ID that will be assigned to the next NEW worker.
            Starts at 1 and increments forever (never reused).

        tracks (dict):
            Currently active tracks.
            Key   = worker_id  (int)
            Value = dict with:
                "centroid"     : (cx, cy) — last known centroid
                "bbox"         : (x1, y1, x2, y2) — last known bounding box
                "confidence"   : float — last known confidence score
                "keypoints"    : list of 17 (x, y, conf) tuples — skeleton
                "edge_points"  : list of (x, y) — body edge coordinates
                "in_vehicle"   : bool — whether person is inside a vehicle
                "vehicle_info" : dict or None — vehicle details
                "missing"      : int — consecutive frames not matched
    """

    def __init__(self):
        """Initialise the tracker with no active tracks."""
        self.next_worker_id = 1      # first worker gets ID 1
        self.tracks = {}             # empty — no workers yet

    # ──────────────────────────────────────────
    #  MAIN UPDATE METHOD (called every frame)
    # ──────────────────────────────────────────

    def update(self, detections):
        """
        Match new detections to existing tracks and return tracked workers.

        Args:
            detections : list[dict]
                Output of detector.detect_persons(). Each dict has:
                    "bbox", "confidence", "class_id", "centroid",
                    "keypoints", "edge_points", "in_vehicle", "vehicle_info"

        Returns:
            tracked_workers : list[dict]
                Each dict contains:
                    "worker_id"    : int
                    "bbox"         : (x1,y1,x2,y2)
                    "centroid"     : (cx, cy)
                    "confidence"   : float
                    "keypoints"    : list of (x,y,conf) tuples
                    "edge_points"  : list of (x,y) tuples
                    "in_vehicle"   : bool
                    "vehicle_info" : dict or None
        """
        # ─────────────────────────────────────
        #  CASE 1: No existing tracks at all.
        #  → Register every detection as a new worker.
        # ─────────────────────────────────────
        if len(self.tracks) == 0:
            for det in detections:
                self._register_new_worker(det)
            return self._build_output()

        # ─────────────────────────────────────
        #  CASE 2: No new detections this frame.
        #  → All existing tracks become "missing" for one more frame.
        # ─────────────────────────────────────
        if len(detections) == 0:
            self._mark_all_missing()
            return self._build_output()

        # ─────────────────────────────────────
        #  CASE 3: Both old tracks and new detections exist.
        #  → Run the centroid matching algorithm.
        # ─────────────────────────────────────

        # Step 3a: Build distance pairs
        track_ids = list(self.tracks.keys())
        distance_pairs = []

        for tid in track_ids:
            old_centroid = self.tracks[tid]["centroid"]
            for d_idx, det in enumerate(detections):
                new_centroid = det["centroid"]
                dist = self._euclidean(old_centroid, new_centroid)
                distance_pairs.append((dist, tid, d_idx))

        # Step 3b: Sort by distance
        distance_pairs.sort(key=lambda x: x[0])

        # Step 3c: Greedy matching
        matched_track_ids = set()
        matched_det_indices = set()

        for dist, tid, d_idx in distance_pairs:
            if tid in matched_track_ids or d_idx in matched_det_indices:
                continue

            # Adaptive distance threshold: allow larger distance for larger worker bboxes
            track_bbox = self.tracks[tid]["bbox"]
            bw = track_bbox[2] - track_bbox[0]
            bh = track_bbox[3] - track_bbox[1]
            adaptive_limit = max(config.MAX_TRACKING_DISTANCE, int(0.75 * max(bw, bh)))
            if dist > adaptive_limit:
                continue

            # Match found! Smooth coordinates using Exponential Moving Average
            det = detections[d_idx]
            alpha = getattr(config, 'TRACKER_SMOOTHING_ALPHA', 0.70)
            ob = self.tracks[tid]["bbox"]
            nb = det["bbox"]
            sb = (
                int(alpha * nb[0] + (1 - alpha) * ob[0]),
                int(alpha * nb[1] + (1 - alpha) * ob[1]),
                int(alpha * nb[2] + (1 - alpha) * ob[2]),
                int(alpha * nb[3] + (1 - alpha) * ob[3]),
            )
            oc = self.tracks[tid]["centroid"]
            nc = det["centroid"]
            sc = (
                int(alpha * nc[0] + (1 - alpha) * oc[0]),
                int(alpha * nc[1] + (1 - alpha) * oc[1]),
            )

            self.tracks[tid]["centroid"]     = sc
            self.tracks[tid]["bbox"]         = sb
            self.tracks[tid]["confidence"]   = det["confidence"]
            self.tracks[tid]["keypoints"]    = det.get("keypoints", [])
            self.tracks[tid]["edge_points"]  = det.get("edge_points", [])
            self.tracks[tid]["in_vehicle"]   = det.get("in_vehicle", False)
            self.tracks[tid]["vehicle_info"] = det.get("vehicle_info", None)
            self.tracks[tid]["missing"]      = 0

            matched_track_ids.add(tid)
            matched_det_indices.add(d_idx)

        # Step 3d: Handle unmatched detections → new workers.
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_indices:
                self._register_new_worker(det)

        # Step 3e: Handle unmatched tracks → increment missing counter.
        for tid in track_ids:
            if tid not in matched_track_ids:
                self.tracks[tid]["missing"] += 1

        # Step 3f: Remove tracks that have been missing too long.
        self._remove_stale_tracks()

        return self._build_output()

    # ──────────────────────────────────────────
    #  PRIVATE HELPER METHODS
    # ──────────────────────────────────────────

    def _register_new_worker(self, detection):
        """Create a new track entry with a fresh worker ID."""
        self.tracks[self.next_worker_id] = {
            "centroid":     detection["centroid"],
            "bbox":         detection["bbox"],
            "confidence":   detection["confidence"],
            "keypoints":    detection.get("keypoints", []),
            "edge_points":  detection.get("edge_points", []),
            "in_vehicle":   detection.get("in_vehicle", False),
            "vehicle_info": detection.get("vehicle_info", None),
            "missing":      0,
        }
        self.next_worker_id += 1

    def _mark_all_missing(self):
        """Increment the missing counter for every active track."""
        for tid in list(self.tracks.keys()):
            self.tracks[tid]["missing"] += 1
        self._remove_stale_tracks()

    def _remove_stale_tracks(self):
        """
        Delete any track whose missing counter exceeds
        config.MAX_FRAMES_MISSING.
        """
        stale_ids = [
            tid for tid, info in self.tracks.items()
            if info["missing"] > config.MAX_FRAMES_MISSING
        ]
        for tid in stale_ids:
            del self.tracks[tid]

    def _build_output(self):
        """
        Convert internal track dict into the clean list format
        that main.py expects.

        Returns:
            list[dict] with keys: worker_id, bbox, centroid, confidence,
            keypoints, edge_points, in_vehicle, vehicle_info
            Sorted by worker_id for consistent display order.
        """
        output = []
        grace = getattr(config, 'TRACKER_GRACE_FRAMES', 3)
        for tid, info in self.tracks.items():
            if info["missing"] <= grace:
                output.append({
                    "worker_id":    tid,
                    "bbox":         info["bbox"],
                    "centroid":     info["centroid"],
                    "confidence":   info["confidence"],
                    "keypoints":    info["keypoints"],
                    "edge_points":  info["edge_points"],
                    "in_vehicle":   info["in_vehicle"],
                    "vehicle_info": info["vehicle_info"],
                    "missing":      info["missing"],
                })
        output.sort(key=lambda x: x["worker_id"])
        return output

    @staticmethod
    def _euclidean(point_a, point_b):
        """
        Compute the Euclidean distance between two 2D points.

        Formula:  d = √((x₂ - x₁)² + (y₂ - y₁)²)
        """
        return math.sqrt(
            (point_a[0] - point_b[0]) ** 2 +
            (point_a[1] - point_b[1]) ** 2
        )


# ──────────────────────────────────────────────
#  SELF-TEST  (run:  python tracker.py)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import cv2
    import time
    import detector

    print("=" * 55)
    print("  tracker.py — Self-Test  (Detector + Tracker)")
    print("=" * 55)

    # Load YOLO model
    model = detector.load_model()

    # Create tracker instance
    tracker = CentroidTracker()

    # Open video source
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {config.VIDEO_SOURCE}")
        exit(1)

    print(f"[tracker] Video opened: {config.VIDEO_SOURCE}")
    print(f"[tracker] MAX_TRACKING_DISTANCE = {config.MAX_TRACKING_DISTANCE} px")
    print(f"[tracker] MAX_FRAMES_MISSING    = {config.MAX_FRAMES_MISSING} frames")
    print("[tracker] Press 'q' to quit.\n")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[tracker] End of video stream.")
            break

        frame_count += 1
        start_time = time.time()

        # Step 1: Detect persons with keypoints
        detections = detector.detect_persons(model, frame)

        # Step 2: Update tracker with new detections
        tracked_workers = tracker.update(detections)

        elapsed = time.time() - start_time
        fps = 1.0 / elapsed if elapsed > 0 else 0.0

        # Step 3: Draw results
        for worker in tracked_workers:
            x1, y1, x2, y2 = worker["bbox"]
            cx, cy = worker["centroid"]
            wid = worker["worker_id"]
            conf = worker["confidence"]

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), config.COLOR_SAFE, config.BBOX_THICKNESS)

            # Worker ID label with background
            label = f"Worker {wid} ({conf:.2f})"
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX,
                config.FONT_SCALE, config.FONT_THICKNESS
            )
            cv2.rectangle(frame, (x1, y1 - th - baseline - 6),
                          (x1 + tw + 4, y1), config.COLOR_SAFE, cv2.FILLED)
            cv2.putText(frame, label, (x1 + 2, y1 - baseline - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE,
                        (255, 255, 255), config.FONT_THICKNESS)

            # Draw skeleton
            keypoints = worker.get("keypoints", [])
            if keypoints:
                detector.draw_skeleton(frame, keypoints)

            # Centroid dot
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # HUD
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(frame, f"Tracked: {len(tracked_workers)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
        cv2.putText(frame, f"Total IDs issued: {tracker.next_worker_id - 1}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Print to terminal (every 30 frames)
        if frame_count % 30 == 0:
            print(f"Frame {frame_count:>5d} | FPS {fps:5.1f} | "
                  f"Visible: {len(tracked_workers)} | "
                  f"Active tracks: {len(tracker.tracks)}")
            for w in tracked_workers:
                vehicle_tag = " [IN VEHICLE]" if w.get("in_vehicle") else ""
                print(f"  └─ Worker {w['worker_id']}  "
                      f"centroid={w['centroid']}  "
                      f"edges={len(w.get('edge_points', []))}  "
                      f"conf={w['confidence']:.4f}"
                      f"{vehicle_tag}")

        cv2.imshow("Tracker Self-Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[tracker] Quit by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[tracker] Self-test finished.")
