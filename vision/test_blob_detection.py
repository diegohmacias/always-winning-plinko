#!/usr/bin/env python3
"""
IMX708: Preview at 640x480, color-based blob detection with size filtering.

- Preview (Picamera2) resolution: PREVIEW_SIZE
- Detection resolution (much smaller): DETECT_SIZE
- Detection: HSV color threshold -> morphology -> contours -> size filter
- Real-world size filtering based on camera calibration
- Interactive HSV and size threshold sliders
- Displays preview, mask, and detection info

Press 'q' to quit
"""

from picamera2 import Picamera2
import cv2
import time
import threading
import queue
import numpy as np

# USER-CONFIGURABLE PARAMETERS
PREVIEW_SIZE = (640, 480)  # width x height for preview
DETECT_SIZE = (160, 120)   # width x height for detection
DETECT_FPS_CAP = 15.0      # maximum detection frequency (Hz)
QUEUE_MAX = 2              # internal frame queue size between preview and detector

# Camera calibration parameters
CAMERA_HEIGHT = 19.5       # inches above the detection plane
CAMERA_DISTANCE = 29.25    # inches horizontal distance from camera to board
BALL_DIAMETER = 1.25       # inches - target ball diameter

# Calculate field of view parameters (these will be refined through calibration)
# For now, we'll use pixel-to-inch ratio that can be adjusted
# Assuming the camera sees approximately 20 inches width at the board distance
ESTIMATED_FOV_WIDTH = 20.0  # inches at the board plane
PIXELS_PER_INCH = DETECT_SIZE[0] / ESTIMATED_FOV_WIDTH  # pixels per inch at detection resolution

# Calculate expected ball size in pixels
EXPECTED_BALL_PIXELS = BALL_DIAMETER * PIXELS_PER_INCH
EXPECTED_BALL_AREA = np.pi * (EXPECTED_BALL_PIXELS / 2) ** 2

# Global variables for trackbar values - TUNED PARAMETERS
hsv_lower = [101, 86, 46]   # Tuned HSV lower bounds
hsv_upper = [142, 255, 255] # Tuned HSV upper bounds
min_area_ratio = 19         # Tuned: 19% of expected ball area
max_area_ratio = 104        # Tuned: 104% of expected ball area
min_circularity = 48        # Tuned: circularity threshold

# Latest mask for display
latest_mask = None
mask_lock = threading.Lock()

# Morphology kernel sizes to clean up mask
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
MORPH_ITERATIONS = 1


def nothing(x):
    """Callback for trackbars (does nothing, values read directly)"""
    pass


def create_control_window():
    """Create window with HSV and size threshold trackbars"""
    cv2.namedWindow("Controls")
    
    # HSV Lower bounds - with tuned defaults
    cv2.createTrackbar("H Low", "Controls", hsv_lower[0], 179, nothing)
    cv2.createTrackbar("S Low", "Controls", hsv_lower[1], 255, nothing)
    cv2.createTrackbar("V Low", "Controls", hsv_lower[2], 255, nothing)
    
    # HSV Upper bounds - with tuned defaults
    cv2.createTrackbar("H High", "Controls", hsv_upper[0], 179, nothing)
    cv2.createTrackbar("S High", "Controls", hsv_upper[1], 255, nothing)
    cv2.createTrackbar("V High", "Controls", hsv_upper[2], 255, nothing)
    
    # Size filtering - with tuned defaults
    cv2.createTrackbar("Min Size %", "Controls", min_area_ratio, 300, nothing)
    cv2.createTrackbar("Max Size %", "Controls", max_area_ratio, 500, nothing)
    
    # Circularity threshold - with tuned default
    cv2.createTrackbar("Circularity", "Controls", min_circularity, 100, nothing)


def update_trackbar_values():
    """Read current trackbar values"""
    global hsv_lower, hsv_upper, min_area_ratio, max_area_ratio, min_circularity
    
    hsv_lower[0] = cv2.getTrackbarPos("H Low", "Controls")
    hsv_lower[1] = cv2.getTrackbarPos("S Low", "Controls")
    hsv_lower[2] = cv2.getTrackbarPos("V Low", "Controls")
    
    hsv_upper[0] = cv2.getTrackbarPos("H High", "Controls")
    hsv_upper[1] = cv2.getTrackbarPos("S High", "Controls")
    hsv_upper[2] = cv2.getTrackbarPos("V High", "Controls")
    
    min_area_ratio = cv2.getTrackbarPos("Min Size %", "Controls")
    max_area_ratio = cv2.getTrackbarPos("Max Size %", "Controls")
    min_circularity = cv2.getTrackbarPos("Circularity", "Controls")


def blob_detection_worker(frame_q, result_q, stop_event):
    """
    Worker thread for blob detection with size filtering.
    - consumes BGR frames from frame_q
    - resizes to DETECT_SIZE
    - converts to HSV and thresholds for blue color
    - applies morphology and finds contours
    - filters by size and circularity
    - returns bounding boxes with additional info through result_q
    """
    global latest_mask
    last_run = 0.0
    
    while not stop_event.is_set():
        try:
            frame = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        now = time.time()
        if now - last_run < 1.0 / DETECT_FPS_CAP:
            continue 
        last_run = now

        # Resize to detection size
        small = cv2.resize(frame, DETECT_SIZE, interpolation=cv2.INTER_LINEAR)

        # Convert to HSV
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Get current HSV thresholds
        hsv_low = np.array(hsv_lower)
        hsv_high = np.array(hsv_upper)

        # Threshold the HSV image
        mask = cv2.inRange(hsv, hsv_low, hsv_high)

        # Morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=MORPH_ITERATIONS)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=MORPH_ITERATIONS)

        # Store mask for display
        with mask_lock:
            latest_mask = mask.copy()

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Calculate size thresholds
        min_area = EXPECTED_BALL_AREA * (min_area_ratio / 100.0)
        max_area = EXPECTED_BALL_AREA * (max_area_ratio / 100.0)

        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            
            # Size filtering
            if area < min_area or area > max_area:
                continue
            
            # Circularity filtering (4*pi*area / perimeter^2)
            # Perfect circle = 1.0, we scale to 0-100
            perimeter = cv2.arcLength(c, True)
            if perimeter == 0:
                continue
            
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            circularity_percent = circularity * 100
            
            if circularity_percent < min_circularity:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(c)
            
            # Calculate equivalent diameter
            equiv_diameter_pixels = np.sqrt(4 * area / np.pi)
            equiv_diameter_inches = equiv_diameter_pixels / PIXELS_PER_INCH
            
            # Store blob info: (x1, y1, x2, y2, area, circularity, diameter_inches)
            blobs.append((x, y, x+w, y+h, area, circularity_percent, equiv_diameter_inches))

        # Put results into queue
        try:
            result_q.put_nowait((time.time(), blobs))
        except queue.Full:
            pass

        try:
            frame_q.task_done()
        except Exception:
            pass


def main():
    # Initialize Picamera2
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": PREVIEW_SIZE})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(2)

    # Create control window with trackbars
    create_control_window()

    # Queues and worker thread
    frame_q = queue.Queue(maxsize=QUEUE_MAX)
    result_q = queue.Queue(maxsize=QUEUE_MAX)
    stop_event = threading.Event()
    worker = threading.Thread(target=blob_detection_worker, args=(frame_q, result_q, stop_event))
    worker.start()

    prev_time = time.time()
    fps_preview = 0.0
    latest_blobs = []
    latest_det_time = 0.0

    try: 
        while True:
            # Update trackbar values
            update_trackbar_values()

            # Capture frame
            frame_rgb = picam2.capture_array()
            if frame_rgb is None:
                continue

            # Convert to BGR
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Compute preview FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps_preview = 1.0 / dt if dt > 0 else fps_preview

            # Send frame to detection worker
            try: 
                if not frame_q.full():
                    intermediate = cv2.resize(frame_bgr, (max(DETECT_SIZE[0]*2, 320), max(DETECT_SIZE[1]*2, 240)))
                    frame_q.put_nowait(intermediate)
            except queue.Full:
                pass

            # Retrieve latest detection results
            try:
                latest_det_time, latest_blobs = result_q.get_nowait()
            except queue.Empty:
                pass

            # Scale detection coordinates to preview resolution
            h_preview, w_preview = frame_bgr.shape[:2]
            sx = w_preview / DETECT_SIZE[0]
            sy = h_preview / DETECT_SIZE[1]

            # Draw detection results
            for blob in latest_blobs:
                x1, y1, x2, y2, area, circularity, diameter_inches = blob
                
                # Scale coordinates
                x1p = int(x1 * sx)
                y1p = int(y1 * sy)
                x2p = int(x2 * sx)
                y2p = int(y2 * sy)
                
                # Draw bounding box
                cv2.rectangle(frame_bgr, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
                
                # Draw center marker
                cx = int((x1p + x2p) / 2)
                cy = int((y1p + y2p) / 2)
                cv2.drawMarker(frame_bgr, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)
                
                # Draw info text
                info_text = f"{diameter_inches:.2f}\" C:{circularity:.0f}%"
                cv2.putText(frame_bgr, info_text, (x1p, y1p - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Draw overlay info
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            y_offset = 20
            
            info_lines = [
                f"FPS: {fps_preview:.1f}",
                f"Resolution: {w_preview}x{h_preview}",
                f"Detections: {len(latest_blobs)}",
                f"Target: {BALL_DIAMETER}\" ball",
                f"Expected area: {EXPECTED_BALL_AREA:.0f} px^2"
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = y_offset + i * 20
                cv2.putText(frame_bgr, line, (10, y_pos), font, scale, (0, 0, 0), thickness+1)
                cv2.putText(frame_bgr, line, (10, y_pos), font, scale, (255, 255, 255), thickness)

            # Show preview window
            cv2.imshow("IMX708 Blob Detection Preview", frame_bgr)

            # Show mask window
            with mask_lock:
                if latest_mask is not None:
                    # Resize mask to match preview for easier viewing
                    mask_display = cv2.resize(latest_mask, (w_preview, h_preview), 
                                             interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Mask", mask_display)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass    
    finally:
        # Shutdown worker cleanly
        stop_event.set()
        worker.join(timeout=1.0)
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
