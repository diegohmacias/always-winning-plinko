#!/usr/bin/env python3
"""
test_camera.py

Picamera2 test for Arducam IMX708 / Raspberry Pi 
Takes raw image and uses OpenCV to correct image color and saves two .jpg images

To run:
cd /always-winning-plinko/src/tests && python3 test_camera.py
"""

from picamera2 import Picamera2
import cv2
import time
import sys
import os

def main():
    try:
        picam2 = Picamera2()
    except Exception as e:
        print("ERROR: Failed to construct Picamera2:", e)
        sys.exit(2)

    # Print detected camera info (if available)
    try:
        camera_config = picam2.create_preview_configuration({"size": (1280, 720)})
        picam2.configure(camera_config)
        print("Camera configured with requested preview size 1280x720.")
    except Exception as e:
        print("WARNING: could not configure preview with requested settings:", e)

    try:
        picam2.start()

        # allow sensor to auto-adjust (AWB/exposure)
        time.sleep(1.5)

        # capture
        img = picam2.capture_array() # this is RGB
        h, w = img.shape[:2]

        test_dir = os.path.dirname(os.path.abspath(__file__))
        raw_path = os.path.join(test_dir, "capture_raw.jpg")
        out_path = os.path.join(test_dir, "capture.jpg")

        # save the raw array directly (OpenCV will treat it as BGR, so colors will look swapped)
        cv2.imwrite(raw_path, img)
        print(f"Saved raw capture (likely RGB saved as BGR): {raw_path}")

        # convert RGB -> BGR for correct colors in OpenCV and save
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, img_bgr)
        print(f"Saved converted capture (RGB->BGR): {out_path}")

        # print per-channel averages (helpful for diagnosing color casts)
        avg_rgb = img.mean(axis=(0,1)) # [R, G, B]
        print(f"Average pixel values (R, G, B): {avg_rgb}")

        picam2.stop()
        print("Camera test completed successfully.")
    except Exception as e:
        print("ERROR during capture:", e)
        try:
            picam2.stop()
        except Exception:
            pass
        sys.exit(4)

if __name__ == "__main__":
    main()
