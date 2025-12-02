#!/usr/bin/env python3
"""
Board coordinate + blue-blob detection + Arduino serial output.

1. Preview at PREVIEW_SIZE. User clicks 4 corners of board in defined order.
2. Send HOME command to Arduino (moves to left corner, 0.0 normalized)
3. Continuously detect blue blob and send normalized x position to Arduino
4. Arduino runs PID continuously to track ball position

Press 'q' to quit
Press 't' to toggle tuning mode (show/hide controls and mask)
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
DETECT_FPS_CAP = 30.0
QUEUE_MAX = 2

# Camera calibration parameters
CAMERA_HEIGHT = 19.5
CAMERA_DISTANCE = 29.25
BALL_DIAMETER = 1.25

# Board dimensions
BOARD_WIDTH_IN = 12.5
BOARD_HEIGHT_IN = 18.75

# Calculate field of view parameters
ESTIMATED_FOV_WIDTH = 20.0
PIXELS_PER_INCH = DETECT_SIZE[0] / ESTIMATED_FOV_WIDTH

# Calculate expected ball size in pixels
EXPECTED_BALL_PIXELS = BALL_DIAMETER * PIXELS_PER_INCH
EXPECTED_BALL_AREA = np.pi * (EXPECTED_BALL_PIXELS / 2) ** 2

# Global variables for trackbar values - TUNED PARAMETERS
hsv_lower = [101, 104, 46]
hsv_upper = [142, 255, 255]
min_area_ratio = 12
max_area_ratio = 104
min_circularity = 30

# Serial configuration
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
SERIAL_TIMEOUT = 0.1

# Morphology kernel
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
MORPH_ITERATIONS = 1

# Global state
clicked_points = []
latest_mask = None
mask_lock = threading.Lock()
tuning_mode = False
last_x_norm_sent = None  # Track last sent position
# -----------------------------------------------------

def nothing(x):
    """Callback for trackbars (does nothing, values read directly)"""
    pass

def create_control_window():
    """Create window with HSV and size threshold trackbars"""
    cv2.namedWindow("Controls")
    
    cv2.createTrackbar("H Low", "Controls", hsv_lower[0], 179, nothing)
    cv2.createTrackbar("S Low", "Controls", hsv_lower[1], 255, nothing)
    cv2.createTrackbar("V Low", "Controls", hsv_lower[2], 255, nothing)
    
    cv2.createTrackbar("H High", "Controls", hsv_upper[0], 179, nothing)
    cv2.createTrackbar("S High", "Controls", hsv_upper[1], 255, nothing)
    cv2.createTrackbar("V High", "Controls", hsv_upper[2], 255, nothing)
    
    cv2.createTrackbar("Min Size %", "Controls", min_area_ratio, 300, nothing)
    cv2.createTrackbar("Max Size %", "Controls", max_area_ratio, 500, nothing)
    
    cv2.createTrackbar("Circularity", "Controls", min_circularity, 100, nothing)

def update_trackbar_values():
    """Read current trackbar values"""
    global hsv_lower, hsv_upper, min_area_ratio, max_area_ratio, min_circularity
    
    try:
        hsv_lower[0] = cv2.getTrackbarPos("H Low", "Controls")
        hsv_lower[1] = cv2.getTrackbarPos("S Low", "Controls")
        hsv_lower[2] = cv2.getTrackbarPos("V Low", "Controls")
        
        hsv_upper[0] = cv2.getTrackbarPos("H High", "Controls")
        hsv_upper[1] = cv2.getTrackbarPos("S High", "Controls")
        hsv_upper[2] = cv2.getTrackbarPos("V High", "Controls")
        
        min_area_ratio = cv2.getTrackbarPos("Min Size %", "Controls")
        max_area_ratio = cv2.getTrackbarPos("Max Size %", "Controls")
        min_circularity = cv2.getTrackbarPos("Circularity", "Controls")
    except:
        pass

def mouse_click(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"Corner clicked: {len(clicked_points)} at image coords {(x, y)}")

def compute_board_transform(corners_img):
    """
    Compute homography from image pixels to normalized board coordinates (0-1).
    Order: bottom-left, bottom-right, top-right, top-left
    """
    dst = np.array([[0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0]], dtype=np.float32)
    src = np.array(corners_img, dtype=np.float32)
    H, status = cv2.findHomography(src, dst)
    return H

def send_home_command(ser):
    """Send HOME command to Arduino and wait for completion"""
    print("\n=== Sending HOME command to Arduino ===")
    print("Catcher will move to LEFT position (0.0 normalized)")
    ser.write(b"HOME\n")
    ser.flush()
    
    # Wait for homing to complete
    timeout = 10.0  # 10 second timeout for homing
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if ser.in_waiting > 0:
            try:
                response = ser.readline().decode('utf-8').rstrip()
                print(f"Arduino: {response}")
                
                if "HOMING_COMPLETE" in response:
                    print("✓ Homing successful! Catcher at LEFT position.")
                    return True
                elif "HOMING_FAILED" in response:
                    print("✗ Homing failed!")
                    return False
                elif "ERROR" in response:
                    print(f"✗ Error during homing: {response}")
                    return False
            except Exception as e:
                print(f"Error reading response: {e}")
        time.sleep(0.1)
    
    print("✗ Homing timeout!")
    return False

def blob_detection_worker(frame_q, result_q, stop_event):
    """Worker thread for blob detection with size filtering."""
    global latest_mask
    last_run = 0.0
    
    while not stop_event.is_set():
        try:
            frame = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        now = time.time()
        if now - last_run < 1.0 / DETECT_FPS_CAP:
            try:
                frame_q.task_done()
            except:
                pass
            continue 
        last_run = now

        small = cv2.resize(frame, DETECT_SIZE, interpolation=cv2.INTER_LINEAR)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        hsv_low = np.array(hsv_lower)
        hsv_high = np.array(hsv_upper)

        mask = cv2.inRange(hsv, hsv_low, hsv_high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=MORPH_ITERATIONS)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=MORPH_ITERATIONS)

        with mask_lock:
            latest_mask = mask.copy()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_area = EXPECTED_BALL_AREA * (min_area_ratio / 100.0)
        max_area = EXPECTED_BALL_AREA * (max_area_ratio / 100.0)

        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            
            if area < min_area or area > max_area:
                continue
            
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            circularity_percent = circularity * 100
            
            if circularity_percent < min_circularity:
                continue
            
            x, y, w, h = cv2.boundingRect(c)
            
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
            else:
                cx = x + w / 2.0
                cy = y + h / 2.0
            
            equiv_diameter_pixels = np.sqrt(4 * area / np.pi)
            equiv_diameter_inches = equiv_diameter_pixels / PIXELS_PER_INCH
            
            blobs.append((cx, cy, x, y, x+w, y+h, area, circularity_percent, equiv_diameter_inches))

        while not result_q.empty():
            try:
                result_q.get_nowait()
            except queue.Empty:
                break

        try:
            result_q.put_nowait((time.time(), blobs))
        except queue.Full:
            pass

        try:
            frame_q.task_done()
        except Exception:
            pass

def main():
    global tuning_mode, last_x_norm_sent
    
    # Set up serial connection to Arduino
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        ser.flush()
        time.sleep(2)  # Wait for Arduino to boot
        print(f"Opened serial port {SERIAL_PORT} at {BAUD_RATE} baud.")
        
        # Wait for Arduino READY signal
        timeout = 5.0
        start_time = time.time()
        while time.time() - start_time < timeout:
            if ser.in_waiting > 0:
                response = ser.readline().decode('utf-8').rstrip()
                print(f"Arduino: {response}")
                if "READY" in response or "WAITING" in response:
                    break
            time.sleep(0.1)
        
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return

    # Set up camera
    picam2 = Picamera2()
    config = picam2.create_preview_configuration({"size": PREVIEW_SIZE})
    picam2.configure(config)
    picam2.start()
    time.sleep(0.5)

    print("\n=== PHASE 1: Board Calibration ===")
    print("Click the board corners in order: Bottom-Left, Bottom-Right, Top-Right, Top-Left")
    cv2.namedWindow("Preview")
    cv2.setMouseCallback("Preview", mouse_click)

    board_H = None

    # Phase 1: Wait for 4 corner clicks
    while True:
        frame_rgb = picam2.capture_array()
        if frame_rgb is None:
            continue
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        for idx, (x, y) in enumerate(clicked_points):
            cv2.circle(frame_bgr, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(frame_bgr, f"{idx+1}", (x+5, y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(frame_bgr, f"Corners: {len(clicked_points)}/4", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
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

    # Phase 2: Home the catcher to LEFT position
    print("\n=== PHASE 2: Homing Catcher ===")
    if not send_home_command(ser):
        print("ERROR: Homing failed. Exiting.")
        picam2.stop()
        cv2.destroyAllWindows()
        ser.close()
        return
    
    print("\n=== PHASE 3: Ball Detection & Tracking ===")
    print("Catcher will continuously track ball position")
    print("Press 'q' to quit")
    print("Press 't' to toggle tuning mode (shows controls and mask)")

    # Create control window (but don't show it initially)
    create_control_window()
    cv2.destroyWindow("Controls")

    # Phase 3: Detection + Continuous Serial Output
    frame_q = queue.Queue(maxsize=QUEUE_MAX)
    result_q = queue.Queue(maxsize=QUEUE_MAX)
    stop_event = threading.Event()
    worker = threading.Thread(target=blob_detection_worker, args=(frame_q, result_q, stop_event), daemon=True)
    worker.start()

    prev_time = time.time()
    fps_preview = 0.0
    latest_blobs = []
    latest_det_time = 0.0
    
    # Initialize with home position
    last_x_norm_sent = 0.0

    try:
        while True:
            if tuning_mode:
                update_trackbar_values()

            frame_rgb = picam2.capture_array()
            if frame_rgb is None:
                continue
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps_preview = 1.0 / dt if dt > 0 else fps_preview

            while frame_q.qsize() > 0:
                try:
                    frame_q.get_nowait()
                    frame_q.task_done()
                except queue.Empty:
                    break

            try: 
                intermediate = cv2.resize(frame_bgr, (max(DETECT_SIZE[0]*2, 320), 
                                                     max(DETECT_SIZE[1]*2, 240)))
                frame_q.put_nowait(intermediate)
            except queue.Full:
                try:
                    frame_q.get_nowait()
                    frame_q.task_done()
                    frame_q.put_nowait(intermediate)
                except:
                    pass

            try:
                latest_det_time, latest_blobs = result_q.get_nowait()
            except queue.Empty:
                pass

            h_preview, w_preview = frame_bgr.shape[:2]
            sx = w_preview / DETECT_SIZE[0]
            sy = h_preview / DETECT_SIZE[1]

            # Determine position to send
            x_norm_to_send = None
            
            if len(latest_blobs) > 0:
                # Ball detected - use its position
                blob = latest_blobs[0]  # Use first (best) detection
                cx, cy, x1, y1, x2, y2, area, circularity, diameter_inches = blob
                
                cx_preview = cx * sx
                cy_preview = cy * sy
                
                pt_img = np.array([[[cx_preview, cy_preview]]], dtype=np.float32)
                pt_board = cv2.perspectiveTransform(pt_img, board_H)[0][0]
                x_norm = float(pt_board[0])
                x_norm = max(0.0, min(1.0, x_norm))
                
                x_norm_to_send = x_norm
                
                # Draw detection
                x1p = int(x1 * sx)
                y1p = int(y1 * sy)
                x2p = int(x2 * sx)
                y2p = int(y2 * sy)
                
                cv2.rectangle(frame_bgr, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
                
                cx_int = int(cx_preview)
                cy_int = int(cy_preview)
                cv2.drawMarker(frame_bgr, (cx_int, cy_int), (0, 0, 255), 
                              cv2.MARKER_CROSS, 10, 2)
                
                info_text = f"x={x_norm:.3f}"
                cv2.putText(frame_bgr, info_text, (x1p, y1p - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                # No ball detected - send last known position or home position
                x_norm_to_send = last_x_norm_sent if last_x_norm_sent is not None else 0.0
            
            # Send position to Arduino (continuously)
            if x_norm_to_send is not None:
                message = f"{x_norm_to_send:.4f}\n"
                try:
                    ser.write(message.encode('utf-8'))
                    ser.flush()
                    last_x_norm_sent = x_norm_to_send
                except Exception as e:
                    print(f"Serial write error: {e}")

            # Draw overlay info
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            y_offset = 20
            
            info_lines = [
                f"FPS: {fps_preview:.1f}",
                f"Detections: {len(latest_blobs)}",
                f"Target x: {last_x_norm_sent:.3f}" if last_x_norm_sent is not None else "Target x: N/A",
                f"Mode: {'TUNING' if tuning_mode else 'RUNNING'}",
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = y_offset + i * 20
                cv2.putText(frame_bgr, line, (10, y_pos), font, scale, (0, 0, 0), thickness+1)
                cv2.putText(frame_bgr, line, (10, y_pos), font, scale, (255, 255, 255), thickness)

            cv2.imshow("Preview", frame_bgr)

            if tuning_mode:
                with mask_lock:
                    if latest_mask is not None:
                        mask_display = cv2.resize(latest_mask, (w_preview, h_preview), 
                                                 interpolation=cv2.INTER_NEAREST)
                        cv2.imshow("Mask", mask_display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('t'):
                tuning_mode = not tuning_mode
                if tuning_mode:
                    create_control_window()
                    print("Tuning mode ENABLED - Controls and mask visible")
                else:
                    try:
                        cv2.destroyWindow("Controls")
                        cv2.destroyWindow("Mask")
                    except:
                        pass
                    print("Tuning mode DISABLED - Normal operation")

    except KeyboardInterrupt:
        pass
    finally:
        print("\nShutting down...")
        stop_event.set()
        worker.join(timeout=1.0)
        picam2.stop()
        cv2.destroyAllWindows()
        ser.close()
        print("Shutdown complete.")

if __name__ == "__main__":
    main()
