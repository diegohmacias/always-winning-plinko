#!/usr/bin/env python3
"""
IMX708: Preview at 640x480, color-based blob detection at smaller resolution.

- Preview (Picamera2) resolution: PREVIEW_SIZE
- Detection resolution (much smaller): DETECT_SIZE
- Detection: HSV color threshold -> morphology -> contours -> area filter
- Draw detection boxes (scaled) on preview. SHow preview FPS and resolution.

Press 'q' to quit
"""

from picamera2 import Picamera2
import cv2
import time
import threading
import queue
import numpy as np

# USER-CONFIGURABLE PARAMETERS
PREVIEW_SIZE = (640, 480) # width x height for preview
DETECT_SIZE = (160, 120)  # width x height for detection
DETECT_FPS_CAP = 15.0     # maximum detection frequency (Hz)
QUEUE_MAX = 2             # internal frame queue sizez between preview and detector

# HSV range for "blue" detection
# Format: (H, S, V) with H: 0-179, S: 0-255, V: 0-255
HSV_LOWER = np.array([94, 80, 2])   # lower bound for blue (example)
HSV_UPPER = np.array([126, 255, 255]) # upper bound for blue (example)

# Minimum contour area (in pixels on the DETECT_SIZE image) to be considered a blob
MIN_BLOB_AREA = 50 

# Morphology kernel sizes to clean up mask (remove noise / fill holes)
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
MORPH_ITERATIONS = 1


def blob_detection_worker(frame_q, result_q, stop_event):
    """
    Worker thread for blob detection.
    - consumes BGR frames (intermediate sized) from frame_q,
    - resizes to DETECT_SIZE,
    - converts to HSV and thresholds for blue color,
    - applies morphology and finds contours,
    -returns bounding boxes (in DETECT_SIZE coordinates) through result_q.
    """
    last_run = 0.0
    while not stop_event.is_set():
        try:
            # wait for a frame (timeout so we can check stop_event)
            frame = frame_q.get(timeout=0.1)
        except queue.Empty:
            continue

        now = time.time()
        # throttle detection rate so it doesn't run faster than DETECT_FPS_CAP
        if now - last_run < 1.0 / DETECT_FPS_CAP:
            # drop this frame if we're running too fast
            continue 
        last_run = now

        # resize to the small detection size for processing speed
        small = cv2.resize(frame, DETECT_SIZE, interpolation=cv2.INTER_LINEAR)

        # convert to HSV because HSV separates color (hue) from brightness (value)
        # makes color thresholding more robust under changing light
        # frames passed into the worker are BGR (OpenCV default), so use BGR2HSV
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        # morphological operations to remove small noise and fill holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL, iterations=MORPH_ITERATIONS)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=MORPH_ITERATIONS)

        # find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_BLOB_AREA:
                # skip tiny noise blobs
                continue
            
            x, y, w, h = cv2.boundingRect(c)
            blobs.append((x, y, x+w, y+h))

        # put results into result queue (timestamp + list of blobs)
        try:
            result_q.put_nowait((time.time(), blobs))
        except queue.Full:
            # if results queue is full, drop results (prefer preview responsiveness)
            pass

        # mark this frame processed for the producer if task tracking is used
        try:
            frame_q.task_done()
        except Exception:
            pass

def main():
    # initialize Picamera2 and configure preview size
    picam2 = Picamera2()
    preview_config = picam2.create_preview_configuration(main={"size": PREVIEW_SIZE})
    picam2.configure(preview_config)
    picam2.start()
    time.sleep(2) # let camera settle

    # queues and worker thread for detection
    frame_q = queue.Queue(maxsize=QUEUE_MAX)
    result_q = queue.Queue(maxsize=QUEUE_MAX)
    stop_event = threading.Event()
    worker = threading.Thread(target=blob_detection_worker, args=(frame_q, result_q, stop_event))
    worker.start()

    prev_time = time.time()
    fps = 0.0
    latest_blobs = []
    latest_det_time = 0.0

    try: 
        while True:
            # capture frame (RGB array) from Picamera2
            frame_rgb = picam2.capture_array()
            if frame_rgb is None:
                continue

            # convert BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # compute preview FPS
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps_preview = 1.0 / dt if dt > 0 else fps_preview

            # send a copy (or small resize) to detection worker if queue isn't full
            # resize to an intermediate size to reduce queue memory
            # worker will resize further to DETECT_SIZE for detection
            try: 
                if not frame_q.full():
                    intermediate = cv2.resize(frame_bgr, (max(DETECT_SIZE[0]*2, 320), max(DETECT_SIZE[1]*2, 240)))
                    frame_q.put_nowait(intermediate)
            except queue.Full:
                pass

            # retrieve latest detection results if available
            try:
                latest_det_time, latest_blobs = result_q.get_nowait()
            except queue.Empty:
                pass

            # scale detection blobs (which are in DETECT_SIZE coords) to preview coords
            h_preview, w_preview = frame_bgr.shape[:2]
            sx = w_preview / DETECT_SIZE[0]
            sy = h_preview / DETECT_SIZE[1]

            for (x1, y1, x2, y2) in latest_blobs:
                # scale coordinates to preview resolution
                x1p = int(x1 * sx)
                y1p = int(y1 * sy)
                x2p = int(x2 * sx)
                y2p = int(y2 * sy)
                # draw bounding box and center marker
                cv2.rectangle(frame_bgr, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
                cx = int((x1p + x2p) / 2)
                cy = int((y1p + y2p) / 2)
                cv2.drawMarker(frame_bgr, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 10, 2)

            # draw overlay text: resolution and FPS
            text_res = f"Resolution: {w_preview}x{h_preview}"
            text_fps = f"FPS: {fps_preview:.1f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2
            padding = 10
            (w_res, h_res), _ = cv2.getTextSize(text_res, font, scale, thickness)
            (w_fps, h_fps), _ = cv2.getTextSize(text_fps, font, scale, thickness)
            x_res = w_preview - w_res - padding
            y_res = padding + h_res
            x_fps = w_preview - w_fps - padding
            y_fps = y_res + h_fps + 5

            # black outline then white text for readability
            cv2.putText(frame_bgr, text_res, (x_res, y_res), font, scale, (0, 0, 0), thickness+2)
            cv2.putText(frame_bgr, text_res, (x_res, y_res), font, scale, (255, 255, 255), thickness)
            cv2.putText(frame_bgr, text_fps, (x_fps, y_fps), font, scale, (0, 0, 0), thickness+2)
            cv2.putText(frame_bgr, text_fps, (x_fps, y_fps), font, scale, (255, 255, 255), thickness)

            # show preview window
            cv2.imshow("IMX708 Blob Detection Preview", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass    
    finally:
        # shutdown worker cleanly
        stop_event.set()
        worker.join(timeout=1.0)
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
