#!/usr/bin/env python3
"""
Board Coordinate Frame Tool

- Uses your camera preview at 640×480.
- You will click in **order** the four corners of the board:
    1. bottom-left
    2. bottom-right
    3. top-right
    4. top-left
- The board’s real size is known: WIDTH_IN = 12.375 inches, HEIGHT_IN = 12.375 inches.
- After clicking the four corners, the script computes a homography so that any future click maps (x,y) in pixels → (X_in, Y_in) on the board in inches.
- Then the coordinate frame: origin at bottom-left, +X to the right, +Y upward (on the board plane).
- You can then click additional points and it will show board coordinates.
"""

import cv2
import numpy as np
from picamera2 import Picamera2
import time

# --- User parameters ---
PREVIEW_SIZE = (640, 480)      # width, height for preview window
BOARD_WIDTH_IN = 12.5          # inches
BOARD_HEIGHT_IN = 18.75        # inches
# -----------------------------

# Globals (for mouse callback)
clicked_points = []   # list of (x, y) pixel coords for the 4 board corners
homography = None     # 3x3 homography matrix mapping image→board-inches

def mouse_callback(event, x, y, flags, param):
    global clicked_points, homography
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"Corner {len(clicked_points)} clicked at pixel: ({x},{y})")
        else:
            # If homography exists, map this click into board coords
            if homography is not None:
                pt = np.array([[x, y]], dtype=np.float32)
                pts = cv2.perspectiveTransform(np.array([pt]), homography)
                X, Y = pts[0][0]
                print(f"Clicked pixel ({x},{y}) → board coords (inches): X={X:.3f}, Y={Y:.3f}")
            else:
                print("Four corners not yet defined => cannot map additional point.")

def compute_homography_from_board(corners_px):
    """
    Given the four clicked board corners in pixel coords (in order),
    compute a homography that maps from image pixel (x, y) → board-inches (X, Y).
    Order assumed:
      0: bottom-left
      1: bottom-right
      2: top-right
      3: top-left
    """
    # Pixel coordinates
    pts_src = np.array(corners_px, dtype=np.float32)
    # Corresponding board coordinates in inches
    pts_dst = np.array([
        [0.0,             0.0],
        [BOARD_WIDTH_IN,  0.0],
        [BOARD_WIDTH_IN,  BOARD_HEIGHT_IN],
        [0.0,             BOARD_HEIGHT_IN]
    ], dtype=np.float32)
    H, status = cv2.findHomography(pts_src, pts_dst)
    return H

def main():
    global homography
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration({"size": PREVIEW_SIZE})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(0.1)

    window_name = "Board Frame Setup"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Please click the board corners in this order:")
    print("1: bottom-left, 2: bottom-right, 3: top-right, 4: top-left")

    try:
        while True:
            frame_rgb = picam2.capture_array()
            if frame_rgb is None:
                continue
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Draw the clicked points so far
            for idx, (x, y) in enumerate(clicked_points):
                cv2.circle(frame_bgr, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(frame_bgr, str(idx+1), (x+5, y-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # If we have 4 points and no homography yet, compute it
            if len(clicked_points) == 4 and homography is None:
                homography = compute_homography_from_board(clicked_points)
                print("Computed homography matrix:")
                print(homography)
                print("Now click any point to see board coordinates in inches.")

            # Display preview
            cv2.imshow(window_name, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
