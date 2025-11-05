# /always-winning-plinko/scripts/ball_tracker.py
#!/usr/bin/env python3
"""
BallTracker class
Encapsulates Picamera2 capture + fast orange-ball detection.

Public API:
  bt = BallTracker(...)      # configure
  bt.start()                 # starts camera + capture thread
  bt.run_preview()           # runs GUI preview loop (blocking) - returns when stopped
  bt.stop()                  # stops capture and camera (safe to call multiple times)

Detection results:
  - prints detected pixel coords, normalized coords, radius, and smoothed coords
  - draws circle + smoothed center on preview
"""

import threading
import time
import cv2
import numpy as np
from picamera2 import Picamera2

class BallTracker:
    def __init__(self,
                 cap_width=640, cap_height=480,
                 proc_w=320, proc_h=240,
                 process_every=1,
                 smooth_alpha=0.35,
                 hsv_range=(5,140,120,18,255,255),
                 show_mask=False,
                 use_trackbars=False):
        # capture & processing sizes
        self.cap_width = int(cap_width)
        self.cap_height = int(cap_height)
        self.proc_w = int(proc_w)
        self.proc_h = int(proc_h)

        self.process_every = max(1, int(process_every))
        self.smooth_alpha = float(smooth_alpha)

        # HSV defaults (Hlow, Slow, Vlow, Hhigh, Shigh, Vhigh)
        self.hsv_default = tuple(hsv_range)

        self.show_mask = bool(show_mask)
        self.use_trackbars = bool(use_trackbars)

        # picamera2 objects
        self.picam2 = None

        # capture thread + buffer
        self._latest_frame = None  # raw RGB numpy array from picamera2
        self._frame_lock = threading.Lock()
        self._capture_thread = None
        self._capture_thread_stop = threading.Event()

        # running flags
        self._started = False

        # smoothing state
        self._smoothed_center = None

    # -------------------------
    # Trackbar helpers (optional)
    # -------------------------
    @staticmethod
    def _nothing(x): pass

    def create_trackbars(self, win_name="Trackbars"):
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        hl, sl, vl, hh, sh, vh = self.hsv_default
        cv2.createTrackbar("H low", win_name, hl, 179, self._nothing)
        cv2.createTrackbar("S low", win_name, sl, 255, self._nothing)
        cv2.createTrackbar("V low", win_name, vl, 255, self._nothing)
        cv2.createTrackbar("H high", win_name, hh, 179, self._nothing)
        cv2.createTrackbar("S high", win_name, sh, 255, self._nothing)
        cv2.createTrackbar("V high", win_name, vh, 255, self._nothing)
        self._trackbar_window = win_name

    def read_trackbars(self, win_name="Trackbars"):
        hl = cv2.getTrackbarPos("H low", win_name)
        sl = cv2.getTrackbarPos("S low", win_name)
        vl = cv2.getTrackbarPos("V low", win_name)
        hh = cv2.getTrackbarPos("H high", win_name)
        sh = cv2.getTrackbarPos("S high", win_name)
        vh = cv2.getTrackbarPos("V high", win_name)
        low = np.array([hl, sl, vl], dtype=np.uint8)
        high = np.array([hh, sh, vh], dtype=np.uint8)
        return low, high

    # -------------------------
    # Camera / capture thread
    # -------------------------
    def _capture_loop(self):
        """Background thread: read frames repeatedly and keep only the latest."""
        while not self._capture_thread_stop.is_set():
            try:
                arr = self.picam2.capture_array()
            except Exception:
                # brief sleep then continue if capture fails
                time.sleep(0.001)
                continue
            with self._frame_lock:
                # store raw RGB frame
                self._latest_frame = arr
            # tiny yield to avoid busy-loop
            time.sleep(0.001)

    def start(self):
        if self._started:
            return
        # init camera
        self.picam2 = Picamera2()
        # prefer video config for streaming
        try:
            cfg = self.picam2.create_video_configuration({"size": (self.cap_width, self.cap_height)})
        except Exception:
            cfg = self.picam2.create_preview_configuration({"size": (self.cap_width, self.cap_height)})
        self.picam2.configure(cfg)
        self.picam2.start()
        # start capture thread
        self._capture_thread_stop.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        # optional trackbars
        if self.use_trackbars:
            try:
                self.create_trackbars()
            except Exception:
                self.use_trackbars = False
        # warm up
        time.sleep(1.0)
        self._started = True

    def stop(self):
        if not self._started:
            return
        # stop capture thread
        self._capture_thread_stop.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
        # stop camera
        try:
            self.picam2.stop()
        except Exception:
            pass
        self._started = False

    # -------------------------
    # Detection routine
    # -------------------------
    def detect_on_frame(self, frame_bgr):
        """
        Detect largest orange-ish object in BGR frame (full-size).
        We actually process a downscaled copy for speed and then map back.
        Returns: (center_pixel_tuple or None, radius_pixels or 0, mask_proc)
        """
        # downscale for processing
        proc = cv2.resize(frame_bgr, (self.proc_w, self.proc_h), interpolation=cv2.INTER_LINEAR)
        proc_blur = cv2.GaussianBlur(proc, (5,5), 0)
        hsv = cv2.cvtColor(proc_blur, cv2.COLOR_BGR2HSV)

        # read thresholds
        if self.use_trackbars:
            lower, upper = self.read_trackbars(self._trackbar_window)
        else:
            lower = np.array(self.hsv_default[0:3], dtype=np.uint8)
            upper = np.array(self.hsv_default[3:6], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)

        # light morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # contours on small mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, 0, mask

        c = max(contours, key=cv2.contourArea)
        ((x_proc, y_proc), r_proc) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx_proc = M["m10"] / M["m00"]
            cy_proc = M["m01"] / M["m00"]
        else:
            cx_proc = x_proc
            cy_proc = y_proc

        # ignore tiny noise
        if r_proc <= 4:
            return None, 0, mask

        # map back to full frame coords
        scale_x = frame_bgr.shape[1] / float(self.proc_w)
        scale_y = frame_bgr.shape[0] / float(self.proc_h)
        cx_full = int(cx_proc * scale_x)
        cy_full = int(cy_proc * scale_y)
        r_full = int(r_proc * (scale_x + scale_y) / 2.0)

        return (cx_full, cy_full), r_full, mask

    # -------------------------
    # Helper: run preview loop (blocking)
    # -------------------------
    def run_preview(self,
                    window_name="Ball Detection - Preview",
                    show_mask=False,
                    process_every=1):
        """
        Blocking preview loop.
        - show_mask: show the mask window scaled up (extra cost)
        - process_every: process every Nth frame (>=1)
        """
        if not self._started:
            raise RuntimeError("BallTracker.start() must be called before run_preview()")

        frame_count = 0
        last_time = None
        fps_ema = None
        fps_alpha = 0.2
        smoothed_center = None

        print("Preview running. Press 'q' in window or Ctrl+C to stop.")
        try:
            while True:
                # fetch latest frame
                with self._frame_lock:
                    frame_rgb = None if (self._latest_frame is None) else self._latest_frame.copy()

                if frame_rgb is None:
                    time.sleep(0.005)
                    continue

                frame_count += 1
                # prepare full-size BGR frame for display
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                # decide whether to process this frame
                if (frame_count % max(1, process_every)) == 0:
                    center, radius, mask_proc = self.detect_on_frame(frame_bgr)
                    if center is not None:
                        # draw detection
                        cv2.circle(frame_bgr, center, max(2, radius), (0,255,255), 2)
                        cv2.circle(frame_bgr, center, 3, (0,0,255), -1)
                        # normalized coords
                        norm_x = center[0] / frame_bgr.shape[1]
                        norm_y = center[1] / frame_bgr.shape[0]
                        # smoothing (EMA)
                        if smoothed_center is None:
                            smoothed_center = (norm_x, norm_y)
                        else:
                            smoothed_center = (self.smooth_alpha * norm_x + (1-self.smooth_alpha) * smoothed_center[0],
                                               self.smooth_alpha * norm_y + (1-self.smooth_alpha) * smoothed_center[1])
                        # print detection (this is the place to hook serial)
                        print(f"Ball px=({center[0]},{center[1]}) r={radius} | norm=({norm_x:.3f},{norm_y:.3f}) | smooth=({smoothed_center[0]:.3f},{smoothed_center[1]:.3f})")

                # overlay smoothed center if available
                if smoothed_center is not None:
                    sx = int(smoothed_center[0] * frame_bgr.shape[1])
                    sy = int(smoothed_center[1] * frame_bgr.shape[0])
                    cv2.circle(frame_bgr, (sx, sy), 6, (0,255,0), 2)
                    cv2.putText(frame_bgr, f"S({sx},{sy})", (sx+8, sy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

                # fps calculation (EMA)
                now = time.time()
                if last_time is None:
                    last_time = now
                    inst_fps = 0.0
                else:
                    inst_fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0.0
                last_time = now
                if fps_ema is None:
                    fps_ema = inst_fps
                else:
                    fps_ema = fps_alpha * inst_fps + (1-fps_alpha) * fps_ema

                cv2.putText(frame_bgr, f"FPS: {fps_ema:.1f}", (10,24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
                cv2.putText(frame_bgr, f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]}", (10,52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

                cv2.imshow(window_name, frame_bgr)
                if show_mask and 'mask_proc' in locals():
                    mask_up = cv2.resize(mask_proc, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Mask", mask_up)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                # toggle trackbars with 't' if available
                elif key == ord('t') and self.use_trackbars:
                    # user toggles the trackbar window
                    if cv2.getWindowProperty("Trackbars", 0) >= 0:
                        cv2.destroyWindow("Trackbars")
                        self.use_trackbars = False
                    else:
                        self.create_trackbars()
                        self.use_trackbars = True

            # end while
        except KeyboardInterrupt:
            pass
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # end run_preview
