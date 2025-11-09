#!/usr/bin/env python3
"""
src/scripts/board_detection.py

BoardDetector class that handles camera capture and interactive board calibration.

Public API (match pattern from BallTracker):
  bd = BoardDetector(...)
  bd.start()                # starts camera + capture thread (non-blocking)
  bd.run_preview(...)       # blocking GUI loop (calls internal detection & UI)
  bd.stop()                 # stops capture thread and camera (safe to call multiple times)

Notes:
 - Saves captures to `captures/`.
 - If `calib.npz` present (mtx, dist), frames are undistorted before processing.
 - Mouse controls: left-click = add corner, right-click or Backspace = undo last corner.
 - Keys: r=reset, s=save image to captures/, h=save homography, q/ESC=quit.
"""
import os
import time
import threading
import math
import argparse
from collections import deque

import cv2
import numpy as np

# Try Picamera2 (preferred on modern Pi). Fall back to OpenCV capture.
try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except Exception:
    PICAMERA2_AVAILABLE = False

# ----------------- Defaults -----------------
BOARD_WIDTH_IN = 12.375
BOARD_HEIGHT_IN = 12.375
CAP_WIDTH = 1280
CAP_HEIGHT = 720
REPROJ_ERR_THRESH_IN = 0.25
TRACK_RECOMPUTE_EVERY = 10
LK_TRACK_PARAMS = dict(winSize=(21,21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
# --------------------------------------------

class BoardDetector:
    def __init__(self,
                 board_w_in=BOARD_WIDTH_IN,
                 board_h_in=BOARD_HEIGHT_IN,
                 cap_width=CAP_WIDTH,
                 cap_height=CAP_HEIGHT,
                 enable_tracking=True):
        self.board_w_in = float(board_w_in)
        self.board_h_in = float(board_h_in)
        self.cap_w = int(cap_width)
        self.cap_h = int(cap_height)
        self.enable_tracking = bool(enable_tracking)

        # capture objects / thread
        self._thread = None
        self._thread_stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame = None
        self._started = False

        # camera holders (if picamera2 used)
        self._picam2 = None
        self._cap_cv = None  # fallback cv2.VideoCapture

        # UI / detection state
        self.corners_selected = []   # list of [x,y]
        self.selecting_corners = True
        self.H = None
        self.H_inv = None
        self.tracking_enabled = False
        self.tracking_points = None
        self.prev_gray = None
        self.track_frame_count = 0
        self.mouse_pos = (0,0)

        # calibration
        self.calib_K = None
        self.calib_D = None
        if os.path.exists("calib.npz"):
            try:
                d = np.load("calib.npz")
                self.calib_K = d["mtx"]
                self.calib_D = d["dist"]
                print("Loaded calib.npz (K,D).")
            except Exception as e:
                print("Failed reading calib.npz:", e)

        os.makedirs("captures", exist_ok=True)

    # ---------------- Capture -----------------
    def _capture_loop_picam2(self):
        """Background capture loop for Picamera2 (stores latest frame as BGR)."""
        while not self._thread_stop.is_set():
            try:
                arr = self._picam2.capture_array()
            except Exception:
                time.sleep(0.002)
                continue
            # convert to BGR if RGB
            try:
                frame = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            except Exception:
                frame = arr
            with self._frame_lock:
                self._latest_frame = frame
            time.sleep(0.001)

    def _capture_loop_cv(self):
        while not self._thread_stop.is_set():
            ret, frame = self._cap_cv.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue
            with self._frame_lock:
                self._latest_frame = frame
            time.sleep(0.001)

    def start(self, device_index=0):
        """Start camera and capture thread."""
        if self._started:
            return
        # try picamera2
        if PICAMERA2_AVAILABLE:
            try:
                self._picam2 = Picamera2()
                try:
                    cfg = self._picam2.create_video_configuration({"size": (self.cap_w, self.cap_h)})
                except Exception:
                    cfg = self._picam2.create_preview_configuration({"size": (self.cap_w, self.cap_h)})
                self._picam2.configure(cfg)
                self._picam2.start()
                # start thread
                self._thread_stop.clear()
                self._thread = threading.Thread(target=self._capture_loop_picam2, daemon=True)
                self._thread.start()
                self._started = True
                print("BoardDetector: using Picamera2.")
                return
            except Exception as e:
                print("BoardDetector: Picamera2 start failed:", e)
                self._picam2 = None

        # fallback to OpenCV VideoCapture
        self._cap_cv = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
        if not self._cap_cv.isOpened():
            # try default index
            self._cap_cv.release()
            self._cap_cv = cv2.VideoCapture(device_index)
        if not self._cap_cv.isOpened():
            raise RuntimeError("BoardDetector: failed to open any camera.")
        self._cap_cv.set(cv2.CAP_PROP_FRAME_WIDTH, self.cap_w)
        self._cap_cv.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cap_h)
        self._thread_stop.clear()
        self._thread = threading.Thread(target=self._capture_loop_cv, daemon=True)
        self._thread.start()
        self._started = True
        print("BoardDetector: using OpenCV VideoCapture.")

    def stop(self):
        """Stop camera and capture thread."""
        if not self._started:
            return
        self._thread_stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            if self._picam2 is not None:
                self._picam2.stop()
                self._picam2 = None
        except Exception:
            pass
        try:
            if self._cap_cv is not None:
                self._cap_cv.release()
                self._cap_cv = None
        except Exception:
            pass
        self._started = False

    # ---------------- Utilities ----------------
    def _order_corners(self, pts):
        """Order arbitrary 4 points into [BL, BR, TR, TL]"""
        pts = np.array(pts, dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return [bl.tolist(), br.tolist(), tr.tolist(), tl.tolist()]

    def compute_homography_from_corners(self, corners):
        """Expect 4 selected corners (any order). Returns (H, H_inv, reproj_err_in_inches)."""
        if len(corners) != 4:
            return None, None, None
        ordered = self._order_corners(corners)
        src = np.array(ordered, dtype=np.float32)
        dst = np.array([[0.0, 0.0],
                        [self.board_w_in, 0.0],
                        [self.board_w_in, self.board_h_in],
                        [0.0, self.board_h_in]], dtype=np.float32)
        try:
            H = cv2.getPerspectiveTransform(src, dst)
            H_inv = cv2.getPerspectiveTransform(dst, src)
        except Exception:
            H, _ = cv2.findHomography(src, dst, 0)
            if H is None:
                return None, None, None
            H_inv = np.linalg.inv(H)
        src_pts = src.reshape(-1,1,2).astype(np.float32)
        proj = cv2.perspectiveTransform(src_pts, H).reshape(-1,2)
        err = np.mean(np.linalg.norm(proj - dst, axis=1))
        return H, H_inv, float(err)

    def pixel_to_board(self, pixel_xy):
        if self.H is None:
            return None
        p = np.array([[pixel_xy]], dtype=np.float32)
        b = cv2.perspectiveTransform(p, self.H)[0,0]
        return float(b[0]), float(b[1])

    def board_to_pixel(self, board_xy):
        if self.H_inv is None:
            return None
        p = np.array([[board_xy]], dtype=np.float32)
        q = cv2.perspectiveTransform(p, self.H_inv)[0,0]
        return int(round(q[0])), int(round(q[1]))

    def save_homography(self, path="board_homography.npz"):
        if self.H is None:
            print("No homography to save.")
            return
        np.savez(path, H=self.H)
        print("Saved", path)

    def save_capture(self, frame):
        ts = int(time.time())
        path = os.path.join("captures", f"board_capture_{ts}.jpg")
        cv2.imwrite(path, frame)
        print("Saved", path)

    # ---------------- UI & interaction ----------------
    def _mouse_cb(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and self.selecting_corners:
            self.corners_selected.append([int(x), int(y)])
            print("Added corner", len(self.corners_selected), ":", (x,y))
            if len(self.corners_selected) == 4:
                self.selecting_corners = False
                print("4 corners selected -> will compute homography.")
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.corners_selected:
                removed = self.corners_selected.pop()
                print("Undo last corner:", removed)
                self.selecting_corners = True
                self._disable_tracking()

    def _disable_tracking(self):
        self.tracking_enabled = False
        self.tracking_points = None
        self.prev_gray = None
        self.H = None
        self.H_inv = None

    def _draw_grid(self, frame, H_inv, spacing_in=1.0, color=(0,180,0)):
        if H_inv is None:
            return frame
        nx = int(math.floor(self.board_w_in / spacing_in)) + 1
        ny = int(math.floor(self.board_h_in / spacing_in)) + 1
        pts_board = []
        for i in range(nx):
            for j in range(ny):
                pts_board.append([i * spacing_in, j * spacing_in])
        pts_board = np.array(pts_board, dtype=np.float32).reshape(-1,1,2)
        img_pts = cv2.perspectiveTransform(pts_board, H_inv).reshape(-1,2)
        for i in range(nx):
            line_pts = []
            for j in range(ny):
                p = img_pts[i*ny + j]
                line_pts.append((int(round(p[0])), int(round(p[1]))))
            for k in range(len(line_pts)-1):
                cv2.line(frame, line_pts[k], line_pts[k+1], color, 1)
        return frame

    def _draw_overlay(self, frame):
        disp = frame
        h,w = frame.shape[:2]
        # corners
        for i,c in enumerate(self.corners_selected):
            x,y = int(c[0]), int(c[1])
            next_idx = len(self.corners_selected) if self.selecting_corners else -1
            col = (0,255,255) if (self.selecting_corners and i==next_idx) else (0,255,0)
            cv2.circle(disp, (x,y), 7, col, -1)
            cv2.putText(disp, f"{i+1}", (x+8,y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        if self.H is not None and self.H_inv is not None and len(self.corners_selected) == 4:
            pts = np.array(self.corners_selected, dtype=np.int32).reshape(-1,2)
            cv2.polylines(disp, [pts], True, (0,255,0), 2)
            bl = tuple(map(int, self.corners_selected[0]))
            br = tuple(map(int, self.corners_selected[1]))
            tl = tuple(map(int, self.corners_selected[3]))
            mid_bottom = ((bl[0]+br[0])//2, (bl[1]+br[1])//2)
            mid_left = ((bl[0]+tl[0])//2, (bl[1]+tl[1])//2)
            cv2.arrowedLine(disp, bl, mid_bottom, (0,0,255), 2, tipLength=0.2)
            cv2.putText(disp, "X+", (mid_bottom[0]+6, mid_bottom[1]-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.arrowedLine(disp, bl, mid_left, (0,255,0), 2, tipLength=0.2)
            cv2.putText(disp, "Y+", (mid_left[0]+6, mid_left[1]+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            disp = self._draw_grid(disp, self.H_inv, spacing_in=1.0)
            cx = int(round(np.mean(pts[:,0])))
            cy = int(round(np.mean(pts[:,1])))
            cv2.circle(disp, (cx,cy), 5, (255,0,255), 2)
            center_board = self.pixel_to_board((cx,cy))
            if center_board is not None:
                cv2.putText(disp, f"Center: {center_board[0]:.2f}in, {center_board[1]:.2f}in", (10, h-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        # mouse probe
        if self.mouse_pos is not None and self.H is not None:
            b = self.pixel_to_board(self.mouse_pos)
            if b is not None:
                mx,my = self.mouse_pos
                cv2.putText(disp, f"Mouse: ({b[0]:.2f}in, {b[1]:.2f}in)", (mx+12, my-12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
        return disp

    # ---------------- Preview (blocking) ---------------
    def run_preview(self, window_name="Board Calibrator", device_index=0, no_tracking=False):
        """Blocking GUI loop. Use start() first (or call start inside if not started)."""
        if not self._started:
            self.start(device_index=device_index)

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, lambda e,x,y,f,p: self._mouse_cb(e,x,y,f,p))

        fps_ema = None
        last_time = time.time()
        while True:
            with self._frame_lock:
                frame = None if (self._latest_frame is None) else self._latest_frame.copy()
            if frame is None:
                time.sleep(0.005)
                continue

            # undistort if calibration available
            if self.calib_K is not None and self.calib_D is not None:
                frame = cv2.undistort(frame, self.calib_K, self.calib_D)

            frame_disp = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # compute homography if 4 corners present and not already set
            if len(self.corners_selected) == 4 and not self.selecting_corners and self.H is None:
                H, H_inv, reproj = self.compute_homography_from_corners(self.corners_selected)
                if H is None:
                    print("Failed to compute homography. Reselect corners.")
                    self.selecting_corners = True
                    self.corners_selected = []
                else:
                    print(f"Computed homography (reproj err {reproj:.3f} in)")
                    if reproj > REPROJ_ERR_THRESH_IN:
                        print(f"Reprojection error {reproj:.3f}in > {REPROJ_ERR_THRESH_IN}in; reselect corners.")
                        self.selecting_corners = True
                        self.corners_selected = []
                    else:
                        self.H = H
                        self.H_inv = H_inv
                        if (not no_tracking) and self.enable_tracking:
                            pts = np.array(self._order_corners(self.corners_selected), dtype=np.float32).reshape(-1,1,2)
                            self.tracking_points = pts.copy()
                            self.prev_gray = gray.copy()
                            self.tracking_enabled = True
                            self.track_frame_count = 0
                            print("Tracking enabled.")

            # tracking via LK optical flow (if enabled)
            if self.tracking_enabled and self.tracking_points is not None and self.prev_gray is not None:
                new_pts, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.tracking_points, None, **LK_TRACK_PARAMS)
                st = st.reshape(-1)
                good = st.sum() if st.size>0 else 0
                if good < 3:
                    print("Tracking lost points -> disabling tracking, please reselect corners.")
                    self._disable_tracking()
                else:
                    self.tracking_points = new_pts
                    tracked = [ [float(p[0][0]), float(p[0][1])] for p in self.tracking_points.reshape(-1,1,2) ]
                    self.corners_selected = [ [int(round(x)), int(round(y))] for x,y in tracked ]
                    self.track_frame_count += 1
                    if self.track_frame_count % TRACK_RECOMPUTE_EVERY == 0:
                        H, H_inv, reproj = self.compute_homography_from_corners(self.corners_selected)
                        if H is None or reproj > REPROJ_ERR_THRESH_IN * 2:
                            print("Tracking recompute failed or error too large -> disabling tracking.")
                            self._disable_tracking()
                        else:
                            self.H = H
                            self.H_inv = H_inv
                    self.prev_gray = gray.copy()

            # overlay
            frame_disp = self._draw_overlay(frame_disp)

            # fps
            now = time.time()
            inst_fps = 1.0/(now-last_time) if (now-last_time)>1e-6 else 0.0
            last_time = now
            if fps_ema is None:
                fps_ema = inst_fps
            else:
                fps_ema = 0.2*inst_fps + 0.8*fps_ema
            cv2.putText(frame_disp, f"FPS: {fps_ema:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # instruction footer
            cv2.putText(frame_disp, "L-click add | R-click/Backspace undo | r reset | s save | h save H | q quit",
                        (10, frame_disp.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            cv2.imshow(window_name, frame_disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord('r'):
                self.corners_selected = []
                self.selecting_corners = True
                self._disable_tracking()
                print("Reset.")
            elif key == ord('s'):
                self.save_capture(frame)
            elif key == ord('h'):
                self.save_homography()
            elif key in (8, 127):
                if self.corners_selected:
                    removed = self.corners_selected.pop()
                    self.selecting_corners = True
                    self._disable_tracking()
                    print("Undo last corner:", removed)
            # continue loop

        cv2.destroyAllWindows()

# End of file
