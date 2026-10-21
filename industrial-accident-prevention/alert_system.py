"""
alert_system.py — Multi-channel alert system for industrial safety.
ADVISORY ONLY: not a certified safety interlock.
"""

import collections
import csv
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ──────────────────────────────────────────────
#  CONFIG BLOCK — tune everything here
# ──────────────────────────────────────────────
N_ENTER = 3          # consecutive frames before state raises
N_EXIT  = 6          # consecutive frames before state lowers

WARN_BEEP_INTERVAL   = 1.5   # s between warning beeps
DANGER_BEEP_INTERVAL = 0.3   # s between danger beeps
SIREN_TRIGGER_SECS   = 2.0   # danger duration before siren starts

WARN_FREQ     = 700    # Hz
DANGER_FREQ   = 1400   # Hz
SIREN_LO      = 900    # Hz  siren low pitch
SIREN_HI      = 1700   # Hz  siren high pitch
SIREN_STEP_MS = 40     # ms per siren frequency step
BEEP_DURATION_MS = 120
VOLUME = 0.6           # 0.0-1.0

VOICE_COOLDOWN    = 5.0   # s between voice phrases
NOTIFY_COOLDOWN   = 10.0  # s between desktop notifications

FLASH_HZ = 3           # border flash rate in DANGER

AUDIO  = True
VOICE  = True
VISUAL = True
NOTIFY = False  # Avoid Windows Shell_NotifyIconW balloon tip errors
LOG    = True

# Anchor all paths to this file's directory so every process agrees
_ROOT    = Path(__file__).resolve().parent
LOG_DIR  = str(_ROOT / "alerts")
CLIP_DIR = str(_ROOT / "clips")
LOG_CSV  = str(_ROOT / "alerts" / "alerts_log.csv")

CLIP_PRE_SECS  = 5   # seconds of pre-event buffer
CLIP_POST_SECS = 5   # seconds of live frames after DANGER entry

# ──────────────────────────────────────────────
#  OPTIONAL LIBRARY IMPORTS (fail-safe)
# ──────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.pre_init(44100, -16, 2, 512)  # stereo
    pygame.mixer.init()
    _HAS_PYGAME = True
except Exception as e:
    _HAS_PYGAME = False
    print(f"[alert_system] pygame unavailable ({e}); audio disabled.")

try:
    import pyttsx3
    _HAS_TTS = True
except Exception as e:
    _HAS_TTS = False
    print(f"[alert_system] pyttsx3 unavailable ({e}); voice disabled.")

try:
    from plyer import notification as _plyer_notify
    _HAS_PLYER = True
except Exception as e:
    _HAS_PLYER = False
    print(f"[alert_system] plyer unavailable ({e}); notifications disabled.")


# ──────────────────────────────────────────────
#  AUDIO HELPERS
# ──────────────────────────────────────────────

def _make_tone(freq_hz, duration_ms, volume=VOLUME):
    """Synthesise a sine tone, return pygame Sound or None."""
    if not _HAS_PYGAME:
        return None
    sr = 44100
    n  = int(sr * duration_ms / 1000)
    t  = np.linspace(0, duration_ms / 1000, n, endpoint=False)
    wave = (np.sin(2 * np.pi * freq_hz * t) * 32767 * volume).astype(np.int16)
    stereo = np.column_stack([wave, wave])  # make stereo (n, 2)
    return pygame.sndarray.make_sound(stereo)


_warn_beep   = None
_danger_beep = None


def _init_sounds():
    global _warn_beep, _danger_beep
    _warn_beep   = _make_tone(WARN_FREQ,   BEEP_DURATION_MS)
    _danger_beep = _make_tone(DANGER_FREQ, BEEP_DURATION_MS)


def _play_siren_thread(stop_event):
    """Sweep siren between SIREN_LO and SIREN_HI until stop_event is set."""
    if not _HAS_PYGAME:
        return
    sr = 44100
    step_samples = int(sr * SIREN_STEP_MS / 1000)
    direction = 1
    freq = float(SIREN_LO)
    while not stop_event.is_set():
        n = step_samples
        t = np.linspace(0, SIREN_STEP_MS / 1000, n, endpoint=False)
        wave = (np.sin(2 * np.pi * freq * t) * 32767 * VOLUME).astype(np.int16)
        stereo = np.column_stack([wave, wave])
        sound = pygame.sndarray.make_sound(stereo)
        sound.play()
        time.sleep(SIREN_STEP_MS / 1000)
        freq += direction * 30
        if freq >= SIREN_HI:
            direction = -1
        elif freq <= SIREN_LO:
            direction = 1


# ──────────────────────────────────────────────
#  VOICE HELPER
# ──────────────────────────────────────────────

_voice_lock = threading.Lock()
_voice_busy = False


def _speak(text):
    global _voice_busy
    with _voice_lock:
        if _voice_busy:
            return
        _voice_busy = True

    def _run():
        global _voice_busy
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass
        finally:
            _voice_busy = False

    threading.Thread(target=_run, daemon=True).start()


# ──────────────────────────────────────────────
#  LOG HELPER
# ──────────────────────────────────────────────

def _ensure_log():
    os.makedirs(LOG_DIR, exist_ok=True)
    if not os.path.exists(LOG_CSV):
        with open(LOG_CSV, "w", newline="") as f:
            csv.writer(f).writerow(
                ["timestamp", "prev_state", "new_state", "frame", "image", "zone", "clip_filename"])


def _log_event(prev, new, frame_no, img_path, zone="default", clip_path=""):
    _ensure_log()
    with open(LOG_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [datetime.now().isoformat(), prev, new, frame_no, img_path, zone, clip_path])
        f.flush()  # ensure dashboard sees row immediately


# ──────────────────────────────────────────────
#  CLIP RECORDER — background MP4 writer
# ──────────────────────────────────────────────

class ClipRecorder:
    """Background MP4 writer; pre-event buffer + post-event frames."""

    _SENTINEL = None  # signals worker to stop
    _CLIP_W   = 640   # evidence clip width (downscaled for speed)
    _Q_MAX    = 150   # bound queue; drop oldest if full

    def __init__(self, fps: float = 30.0):
        self._fps   = max(fps, 1.0)
        maxlen      = int(self._fps * CLIP_PRE_SECS)
        self._buf   = collections.deque(maxlen=maxlen)  # stores downscaled frames
        self._q: queue.Queue = queue.Queue(maxsize=self._Q_MAX)
        self._worker: threading.Thread | None = None
        self._recording = False
        self._lock = threading.Lock()

    @staticmethod
    def _downscale(frame: np.ndarray) -> np.ndarray:
        """Resize to _CLIP_W wide, preserve aspect ratio."""
        h, w = frame.shape[:2]
        if w <= ClipRecorder._CLIP_W:
            return frame
        scale = ClipRecorder._CLIP_W / w
        return cv2.resize(frame, (ClipRecorder._CLIP_W, int(h * scale)),
                          interpolation=cv2.INTER_LINEAR)

    def push_frame(self, frame: np.ndarray):
        """Called every frame on main thread — keep this as cheap as possible."""
        small = self._downscale(frame)  # downscale once
        self._buf.append(small)         # deque.append is O(1), no copy needed
        if self._recording:
            if self._q.full():
                try:
                    self._q.get_nowait()  # drop oldest, never block main thread
                except queue.Empty:
                    pass
            try:
                self._q.put_nowait(small)
            except queue.Full:
                pass  # still full after drop — skip frame

    def start_clip(self, clip_path: str, h: int, w: int) -> bool:
        """Start a new clip; returns False if one is already in progress."""
        with self._lock:
            if self._recording:
                return False
            self._recording = True

        os.makedirs(CLIP_DIR, exist_ok=True)
        pre_frames = list(self._buf)  # snapshot (already downscaled)
        post_count = int(self._fps * CLIP_POST_SECS)
        # derive clip dimensions from downscaled frames
        if pre_frames:
            ch, cw = pre_frames[0].shape[:2]
        else:
            scale  = self._CLIP_W / max(w, 1)
            cw, ch = self._CLIP_W, int(h * scale)

        self._worker = threading.Thread(
            target=self._write_clip,
            args=(clip_path, ch, cw, pre_frames, post_count),
            daemon=True)
        self._worker.start()
        return True

    def _write_clip(self, path: str, h: int, w: int, pre: list, post_count: int):
        writer = self._open_writer(path, h, w)
        received = 0
        try:
            for f in pre:
                if f.shape[:2] == (h, w):
                    writer.write(f)
                else:
                    writer.write(cv2.resize(f, (w, h)))
            while received < post_count:
                frame = self._q.get(timeout=5.0)
                if frame is self._SENTINEL:
                    break
                if frame.shape[:2] == (h, w):
                    writer.write(frame)
                else:
                    writer.write(cv2.resize(frame, (w, h)))
                received += 1
        except queue.Empty:
            pass
        except Exception as e:
            print(f"[alert_system] Error writing clip: {e}")
        finally:
            writer.release()
            with self._lock:
                self._recording = False
            if os.path.exists(path) and os.path.getsize(path) > 0:
                print(f"[alert_system] Incident clip saved: {path} ({received} post frames)")

    @staticmethod
    def _open_writer(path: str, h: int, w: int) -> cv2.VideoWriter:
        for fourcc_str in ("mp4v", "XVID", "MJPG", "avc1"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
            vw = cv2.VideoWriter(path, fourcc, 20.0, (w, h))
            if vw.isOpened():
                return vw
        return cv2.VideoWriter(path, 0, 20.0, (w, h))

    def shutdown(self):
        """Signal writer and wait."""
        if self._recording:
            self._q.put(self._SENTINEL)
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=10)


# ──────────────────────────────────────────────
#  ALERT MANAGER
# ──────────────────────────────────────────────

class AlertManager:
    """
    Enhanced AlertManager with debounce, audio, voice, visual, notifications, logging.
    Call per frame: frame = alert_manager.update(highest_state, frame, frame_no, workers)
    Legacy methods warning_alert / danger_alert / draw_alert_banner still work.
    """

    def __init__(self, fps: float = 30.0, zone: str = "default"):
        _init_sounds()
        _ensure_log()

        self._confirmed  = "SAFE"
        self._candidate  = "SAFE"
        self._cand_count = 0
        self._exit_count = 0

        self._danger_start  = None
        self._siren_stop    = None
        self._siren_thread  = None

        self._last_beep_time   = 0.0
        self._last_voice_time  = 0.0
        self._last_notify_time = 0.0
        self._beep_playing     = False  # legacy compat
        self._last_beep_time_w = {}     # legacy per-worker

        self._muted  = False
        self._zone   = zone
        self._clip   = ClipRecorder(fps=fps)  # evidence clip recorder

    # ── mute toggle ──
    def toggle_mute(self):
        self._muted = not self._muted
        print(f"[alert_system] Audio/Voice {'MUTED' if self._muted else 'UNMUTED'}")
        if self._muted:
            self._stop_siren()
            if _HAS_PYGAME:
                pygame.mixer.stop()

    # ── main per-frame entry point ──
    def update(self, highest_state, frame, frame_no=0, workers_with_status=None):
        """Debounce, fire all channels, draw visuals. Returns annotated frame."""
        prev = self._confirmed
        self._confirmed = self._debounce(highest_state)

        # feed every raw frame into the rolling buffer
        self._clip.push_frame(frame)

        if self._confirmed != "SAFE":
            self._fire_audio(self._confirmed)
            self._fire_voice(self._confirmed)
            self._fire_notify(self._confirmed)

        if self._confirmed == "DANGER":
            if prev != "DANGER":
                self._danger_start = time.time()
                self._on_enter_danger(frame, frame_no, prev, workers_with_status or [])
            else:
                self._check_repeat_danger_notify(frame, frame_no, workers_with_status or [])
        elif self._confirmed != "DANGER":
            self._danger_start = None
            self._stop_siren()

        if (self._confirmed == "DANGER" and self._danger_start and
                time.time() - self._danger_start >= SIREN_TRIGGER_SECS):
            self._start_siren()

        if VISUAL:
            frame = self._draw_visual(frame, self._confirmed, workers_with_status or [])

        return frame

    # ── legacy hooks (no-op; update() now owns audio/visual) ──
    def warning_alert(self, worker_id):
        pass

    def danger_alert(self, worker_id):
        pass

    def draw_alert_banner(self, frame, workers_with_status):
        """Legacy call from main.py — compute highest state and draw visual."""
        if not workers_with_status:
            highest = "SAFE"
        else:
            order = {"DANGER": 2, "WARNING": 1, "SAFE": 0}
            highest = max((w.get("status", "SAFE") for w in workers_with_status),
                          key=lambda s: order.get(s, 0))
        if VISUAL:
            frame = self._draw_visual(frame, highest, workers_with_status)
        return frame

    def shutdown(self):
        """Stop sound and threads cleanly."""
        self._stop_siren()
        self._clip.shutdown()
        if _HAS_PYGAME:
            try:
                pygame.mixer.stop()
                pygame.mixer.quit()
            except Exception:
                pass

    # ── debounce state machine ──
    def _debounce(self, raw):
        order = {"SAFE": 0, "WARNING": 1, "DANGER": 2}
        cur_level = order.get(self._confirmed, 0)
        raw_level = order.get(raw, 0)

        if raw == self._confirmed:
            self._cand_count = 0
            self._exit_count = 0
            return self._confirmed

        if raw_level > cur_level:
            if raw == self._candidate:
                self._cand_count += 1
            else:
                self._candidate  = raw
                self._cand_count = 1
            if self._cand_count >= N_ENTER:
                self._cand_count = 0
                return raw
            return self._confirmed

        # de-escalation: step down one level
        if raw == "SAFE":
            self._exit_count += 1
            if self._exit_count >= N_EXIT:
                self._exit_count = 0
                levels = ["SAFE", "WARNING", "DANGER"]
                return levels[max(0, cur_level - 1)]
        else:
            self._exit_count = 0
        return self._confirmed

    # ── audio ──
    def _fire_audio(self, state):
        if not AUDIO or not _HAS_PYGAME or self._muted:
            return
        now = time.time()
        interval = WARN_BEEP_INTERVAL if state == "WARNING" else DANGER_BEEP_INTERVAL
        if now - self._last_beep_time < interval:
            return
        self._last_beep_time = now
        sound = _warn_beep if state == "WARNING" else _danger_beep
        if sound:
            sound.play()

    def _start_siren(self):
        if not _HAS_PYGAME or self._muted:
            return
        if self._siren_thread and self._siren_thread.is_alive():
            return
        self._stop_siren()
        self._siren_stop   = threading.Event()
        self._siren_thread = threading.Thread(
            target=_play_siren_thread, args=(self._siren_stop,), daemon=True)
        self._siren_thread.start()

    def _stop_siren(self):
        if self._siren_stop:
            self._siren_stop.set()
        self._siren_stop   = None
        self._siren_thread = None

    # ── voice ──
    def _fire_voice(self, state):
        if not VOICE or not _HAS_TTS or self._muted:
            return
        if time.time() - self._last_voice_time < VOICE_COOLDOWN:
            return
        self._last_voice_time = time.time()
        text = ("Warning. Approaching danger zone."
                if state == "WARNING" else "Danger. Leave the restricted area.")
        _speak(text)

    # ── notification ──
    def _fire_notify(self, state):
        if not NOTIFY or not _HAS_PLYER or state != "DANGER":
            return
        if time.time() - self._last_notify_time < NOTIFY_COOLDOWN:
            return
        self._last_notify_time = time.time()
        def _do_notify():
            try:
                _plyer_notify.notify(
                    title="DANGER - Restricted Area",
                    message="Worker entered a restricted zone!",
                    timeout=5,
                )
            except Exception:
                pass
        threading.Thread(target=_do_notify, daemon=True).start()

    def _on_enter_danger(self, frame, frame_no, prev, workers=None):
        if not LOG:
            return
        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        target_zone = self._zone
        track_id = "0"
        if workers:
            for wk in workers:
                if wk.get("status") == "DANGER":
                    if wk.get("zone_id"):
                        target_zone = wk["zone_id"]
                    track_id = str(wk.get("worker_id", wk.get("id", wk.get("track_id", "0"))))
                    break

        safe_zone = str(target_zone).replace(" ", "_")
        img_name  = f"{ts}_{safe_zone}_worker{track_id}.jpg"
        img_path  = os.path.join(LOG_DIR, img_name)
        # draw bounding boxes on a copy so the main frame is not mutated
        save_frame = frame.copy()
        if workers:
            for wk in workers:
                if wk.get("status") == "DANGER" and "bbox" in wk:
                    x1, y1, x2, y2 = wk["bbox"]
                    cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"DANGER: Worker {wk.get('worker_id', track_id)}"
                    cv2.putText(save_frame, label, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        ok = cv2.imwrite(img_path, save_frame)
        if not ok or not os.path.exists(img_path):
            print(f"[alert_system] ERROR: imwrite failed for {img_path}")
            img_path = ""
        # start background clip recording
        h, w      = frame.shape[:2]
        clip_path = os.path.join(CLIP_DIR, f"{ts}_{safe_zone}.mp4")
        started = self._clip.start_clip(clip_path, h, w)
        if started:
            print(f"[alert_system] Started recording incident clip: {clip_path}")
        else:
            print(f"[alert_system] Clip recording already in progress")

        _log_event(prev, "DANGER", frame_no, img_path, target_zone, clip_path)
        # notify the zone's manager via email + SMS
        try:
            from notifier import notify_manager
            notify_manager(target_zone, img_path)
        except Exception as exc:
            print(f"[alert_system] notify_manager error: {exc}")

        # face recognition — identify which employee (background thread)
        try:
            from face_recognizer import recognize_and_notify
            if workers:
                for wk in workers:
                    if wk.get("status") == "DANGER" and "bbox" in wk:
                        recognize_and_notify(frame, tuple(wk["bbox"]), target_zone, img_path)
        except Exception as exc:
            print(f"[alert_system] face recognition error: {exc}")

    def _check_repeat_danger_notify(self, frame, frame_no, workers):
        """If worker remains continuously in DANGER, periodically trigger notification with a fresh snapshot.
        notify_manager() enforces the 10-minute cooldown."""
        now = time.time()
        if now - getattr(self, "_last_repeat_check", 0.0) < 5.0:
            return
        self._last_repeat_check = now

        target_zone = self._zone
        track_id = "0"
        if workers:
            for wk in workers:
                if wk.get("status") == "DANGER":
                    if wk.get("zone_id"):
                        target_zone = wk["zone_id"]
                    track_id = str(wk.get("worker_id", wk.get("id", "0")))
                    break

        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_zone = str(target_zone).replace(" ", "_")
        img_name = f"repeat_{ts}_{safe_zone}_worker{track_id}.jpg"
        img_path = os.path.join(LOG_DIR, img_name)
        save_frame = frame.copy()
        if workers:
            for wk in workers:
                if wk.get("status") == "DANGER" and "bbox" in wk:
                    x1, y1, x2, y2 = wk["bbox"]
                    cv2.rectangle(save_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    label = f"DANGER: Worker {wk.get('worker_id', track_id)}"
                    cv2.putText(save_frame, label, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        try:
            cv2.imwrite(img_path, save_frame)
        except Exception:
            img_path = ""

        try:
            from notifier import notify_manager
            notify_manager(target_zone, img_path)
        except Exception:
            pass

        # face recognition on repeat danger
        try:
            from face_recognizer import recognize_and_notify
            if workers:
                for wk in workers:
                    if wk.get("status") == "DANGER" and "bbox" in wk:
                        recognize_and_notify(frame, tuple(wk["bbox"]), target_zone, img_path)
        except Exception:
            pass

    # ── visual overlays ──
    def _draw_visual(self, frame, state, workers):
        h, w = frame.shape[:2]
        flash_on = int(time.time() * FLASH_HZ * 2) % 2 == 0

        if state == "SAFE":
            cv2.putText(frame, "SAFE", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 0), 2)

        elif state == "WARNING":
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 165, 255), 6)
            self._draw_banner(frame, "WARNING - Approaching danger zone",
                              (0, 165, 255), w, h)

        elif state == "DANGER":
            if flash_on:
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 8)
            for wk in workers:
                if wk.get("status") == "DANGER":
                    x1, y1, x2, y2 = wk["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            self._draw_banner(frame, "DANGER - Restricted area",
                              (0, 0, 255), w, h)

        return frame

    @staticmethod
    def _draw_banner(frame, text, color, frame_w, frame_h):
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        bx = frame_w // 2 - tw // 2
        by = frame_h - 50
        cv2.rectangle(frame, (bx - 8, by - th - 8),
                      (bx + tw + 8, by + base + 4), (20, 20, 20), cv2.FILLED)
        cv2.putText(frame, text, (bx, by),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)
