#!/usr/bin/env python3
"""
Board coordinate + blue-blob detection + Arduino serial output.

1. Preview at PREVIEW_SIZE. User clicks 4 corners of board in defined order.
2. Detection at smaller DETECT_SIZE for speed.
3. On detection of blue blob: compute normalized x position on board (0-1).
4. Send x_norm via serial to Arduino.
"""

import cv2
import numpy as np
import time
import threading
import queue
import serial  
from picamera2 import Picamera2

# -----------------CONFIGURATION------------------------
PREVIEW_SIZE = (640, 480)
DETECT_SIZE = (160, 120)
BLUE_LOWER = np.array([95, 100, 60])
BLUE_UPPER = np.array([135, 255, 255])
MIN_BLOB_AREA = 50

SERIAL_PORT = '/dev/ttyACM0'     
BAUD_RATE = 9600
SERIAL_TIMEOUT = 0.1            
# -----------------------------------------------------

clicked_points = []

def mouse_click(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Corner clicked: {len(clicked_points)} at image coords {(x, y)}")

def compute_board_transform(corners_img):
    dst = np.array([[0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0]], dtype=np.float32)
    src = np.array(corners_img, dtype=np.float32)
    H, status = cv2.findHomography(src, dst)
    return H

def detect_blue_blob(frame_bgr):
    small = cv2.resize(frame_bgr, DETECT_SIZE, interpolation=cv2.INTER_LINEAR)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        a = cv2.contourArea(c)
        if a < MIN_BLOB_AREA:
            continue
        if a > best_area:
            best_area = a
            best = c
    if best is None:
        return None, None
    x, y, w, h = cv2.boundingRect(best)
    cx = x + w/2
    cy = y + h/2
    # scale up to preview coords
    scale_x = frame_bgr.shape[1] / DETECT_SIZE[0]
    scale_y = frame_bgr.shape[0] / DETECT_SIZE[1]
    centroid = (cx*scale_x, cy*scale_y)
    box = (int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y))
    return box, centroid

def main():
    # Set up serial connection to Arduino
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        ser.flush()
        print(f"Opened serial port {SERIAL_PORT} at {BAUD_RATE} baud.")
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return

    # Set up camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration({"size": PREVIEW_SIZE})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)

    print("Click the board corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left")
    cv2.namedWindow("Preview")
    cv2.setMouseCallback("Preview", mouse_click)

    board_H = None

    # Phase 1: wait for 4 clicks
    while True:
        frame_rgb = picam2.capture_array()
        if frame_rgb is None:
            continue
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        for idx, (x,y) in enumerate(clicked_points):
            cv2.circle(frame_bgr, (x, y), 5, (0,255,0), -1)
            cv2.putText(frame_bgr, f"{idx+1}", (x+5,y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)
        cv2.imshow("Preview", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit before corner selection.")
            picam2.stop()
            cv2.destroyAllWindows()
            ser.close()
            return
        if len(clicked_points) >= 4:
            board_H = compute_board_transform(clicked_points[:4])
            print("Board transform computed.")
            break

    # Phase 2: detection + serial output
    print("Entering blob detection + serial output mode. Press 'q' to quit.")
    # fps helpers for overlay
    prev_time = time.time()
    fps = 0.0
    while True:
        frame_rgb = picam2.capture_array()
        if frame_rgb is None:
            continue
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        box, centroid = detect_blue_blob(frame_bgr)
        if centroid is not None:
            # compute normalized board-frame x
            pt_img = np.array([[centroid]], dtype=np.float32)
            pt_board = cv2.perspectiveTransform(pt_img, board_H)[0][0]
            x_norm = float(pt_board[0])
            # clamp between 0 and 1 just in case
            x_norm = max(0.0, min(1.0, x_norm))
            # send to Arduino
            message = f"{x_norm:.4f}\n"
            ser.write(message.encode('utf-8'))
            # optionally read Arduino response
            if ser.in_waiting > 0:
                response = ser.readline().decode('utf-8').rstrip()
                print(f"Arduino says: {response}")
            # print for debugging
            print(f"Sent to Arduino: {message.strip()}")

            # overlay on preview
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(frame_bgr, (cx, cy), 8, (0,0,255), 2)
            if box:
                bx, by, bw, bh = box
                cv2.rectangle(frame_bgr, (bx,by), (bx+bw, by+bh), (255,0,0),2)

        # calculate fps
        now = time.time()
        dt = now - prev_time
        prev_time = now
        fps = 1.0 / dt if dt > 0 else fps

        # draw resolution and FPS at top-left (outlined for readability)
        text_res = f"Resolution: {frame_bgr.shape[1]}x{frame_bgr.shape[0]}"
        text_fps = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        padding = 10
        (w_res, h_res), _ = cv2.getTextSize(text_res, font, scale, thickness)
        (w_fps, h_fps), _ = cv2.getTextSize(text_fps, font, scale, thickness)
        x_res = padding
        y_res = padding + h_res
        x_fps = padding
        y_fps = y_res + h_fps + 5
        cv2.putText(frame_bgr, text_res, (x_res, y_res), font, scale, (0,0,0), thickness+2)
        cv2.putText(frame_bgr, text_res, (x_res, y_res), font, scale, (255,255,255), thickness)
        cv2.putText(frame_bgr, text_fps, (x_fps, y_fps), font, scale, (0,0,0), thickness+2)
        cv2.putText(frame_bgr, text_fps, (x_fps, y_fps), font, scale, (255,255,255), thickness)

        cv2.imshow("Preview", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    picam2.stop()
    cv2.destroyAllWindows()
    ser.close()

if __name__ == "__main__":
    main()