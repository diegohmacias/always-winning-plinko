#!/usr/bin/env python3
"""
Camera distortion calibration using Picamera2 + OpenCV.
Assumes a checkerboard with CHECKERBOARD inner corners and square size SQUARE_SIZE (in metres).
Captures NUM_REQUIRED valid frames, computes calibration, and saves to OUTPUT_FILE.
"""

import cv2
import numpy as np
import os
import time
from picamera2 import Picamera2

# ---------------- USER CONFIGURATION ----------------
CHECKERBOARD = (9, 6)            # number of inner corners (width, height)
SQUARE_SIZE = 0.030              # size of one square in metres (30 mm)
NUM_REQUIRED = 20                # number of valid frames to capture
OUTPUT_FILE = "camera_calibration_picamera2.npz"
PREVIEW_SIZE = (640, 480)        # size for preview and capture
CAPTURE_DELAY = 0.5              # seconds delay after a successful capture
# ---------------------------------------------------

def capture_calibration_images():
    """Capture valid checkerboard frames from camera, return object & image points."""
    picam2 = Picamera2()
    config = picam2.create_preview_configuration({"size": PREVIEW_SIZE})
    picam2.configure(config)
    picam2.start()
    time.sleep(1.0)  # warm-up

    objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
    objp *= SQUARE_SIZE

    objpoints = []
    imgpoints = []

    if not os.path.exists("calib_images"):
        os.makedirs("calib_images")

    count = 0
    print(f"Collecting {NUM_REQUIRED} valid frames …")
    try:
        while count < NUM_REQUIRED:
            frame_rgb = picam2.capture_array()
            if frame_rgb is None:
                print("Warning: no frame captured")
                continue

            img_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH +
                                                     cv2.CALIB_CB_NORMALIZE_IMAGE +
                                                     cv2.CALIB_CB_FAST_CHECK)
            if ret:
                # refine corner detection
                corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                             30, 0.001))
                objpoints.append(objp)
                imgpoints.append(corners2)

                # draw and save image for review
                cv2.drawChessboardCorners(img_bgr, CHECKERBOARD, corners2, ret)
                fname = os.path.join("calib_images", f"img_{count:02d}.png")
                cv2.imwrite(fname, img_bgr)
                print(f"Saved {fname}")
                count += 1
                time.sleep(CAPTURE_DELAY)
            # show the frame for feedback
            cv2.imshow("Capture (press q to quit)", img_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

    print(f"Collected {count} frames.")
    return objpoints, imgpoints, gray.shape[::-1]

def calibrate_and_save(objpoints, imgpoints, image_size):
    """Calibrate camera and save calibration file."""
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)
    print(f"Calibration RMS error: {ret}")
    np.savez(OUTPUT_FILE, camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs, rvecs=rvecs, tvecs=tvecs)
    print(f"Saved calibration to {OUTPUT_FILE}")
    return camera_matrix, dist_coeffs

def undistort_test(camera_matrix, dist_coeffs, image_path):
    """Optional: undistort a captured image to verify calibration."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                      dist_coeffs, (w,h), 1, (w,h))
    dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)
    cv2.imshow("Original", img)
    cv2.imshow("Undistorted", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    objpoints, imgpoints, image_size = capture_calibration_images()
    camera_matrix, dist_coeffs = calibrate_and_save(objpoints, imgpoints, image_size)
    # Optionally test with first saved image:
    # undistort_test(camera_matrix, dist_coeffs, "calib_images/img_00.png")

if __name__ == "__main__":
    main()
