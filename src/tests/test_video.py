#!/usr/bin/env python3
"""
test_video.py
Picamera2 live-video test for Arducam IMX708 / Raspberry Pi with smoothed FPS overlay
(Live preview only — no saving)

Usage:
  cd /always-winning-plinko/src/tests
  python3 test_video.py
  # optional args:
  python3 test_video.py --width 640 --height 480 --target-fps 30
"""

import argparse
import os
import time
import sys
import signal
from collections import deque

from picamera2 import Picamera2
import cv2

STOP = False

def signal_handler(sig, frame):
    global STOP
    STOP = True

def main():
    global STOP
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Picamera2 live preview (RGB->BGR) with FPS overlay.")
    parser.add_argument("--width", type=int, default=640, help="Requested frame width (default 640)")
    parser.add_argument("--height", type=int, default=480, help="Requested frame height (default 480)")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Target FPS for display (informational)")
    parser.add_argument("--no-gui", action="store_true", help="Run headless (no preview window)")
    args = parser.parse_args()

    try:
        picam2 = Picamera2()
    except Exception as e:
        print("ERROR: Failed to construct Picamera2:", e)
        sys.exit(2)

    # Use video configuration (better for continuous capture)
    try:
        config = picam2.create_video_configuration({"size": (args.width, args.height)})
        picam2.configure(config)
        print(f"Camera configured (video) -> {args.width}x{args.height}")
    except Exception as e:
        print("WARNING: create_video_configuration failed, trying preview config as fallback:", e)
        try:
            config = picam2.create_preview_configuration({"size": (args.width, args.height)})
            picam2.configure(config)
            print("Fallback preview configuration applied.")
        except Exception as e2:
            print("ERROR: failed to configure camera:", e2)
            sys.exit(3)

    try:
        picam2.start()
    except Exception as e:
        print("ERROR: Failed to start camera:", e)
        sys.exit(4)

    # short warm-up for AWB/exposure
    time.sleep(1.0)

    # We'll compute a smoothed FPS using an exponential moving average to make the displayed FPS stable
    ema_alpha = 0.2   # smoothing factor (0.0..1.0). Higher -> more responsive, Lower -> smoother
    ema_fps = None
    last_time = time.time()

    # Also keep a tiny deque for an alternate rolling-average if you prefer
    rolling_times = deque(maxlen=16)

    print("Starting live preview. Press 'q' in the preview window or Ctrl+C to quit.")
    if args.no_gui:
        print("Running in headless mode (--no-gui). No preview window will be shown.")

    try:
        while not STOP:
            # capture a frame (RGB)
            frame_rgb = picam2.capture_array()
            if frame_rgb is None:
                # brief sleep to avoid busy-loop when no frames
                time.sleep(0.01)
                continue

            # convert to BGR for display in OpenCV
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # compute instantaneous FPS
            now = time.time()
            dt = now - last_time if last_time is not None else 0.0
            last_time = now
            inst_fps = 1.0 / dt if dt > 0 else 0.0

            # update EMA FPS
            if ema_fps is None:
                ema_fps = inst_fps
            else:
                ema_fps = ema_alpha * inst_fps + (1.0 - ema_alpha) * ema_fps

            # update rolling deque
            rolling_times.append(inst_fps)

            # Choose displayed fps value (ema is smoother)
            display_fps = ema_fps if ema_fps is not None else inst_fps

            # Overlay FPS text (top-left)
            text = f"FPS: {display_fps:.1f}  Target: {args.target_fps:.0f}"
            cv2.putText(frame_bgr, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Optionally, show resolution and buffer info
            cv2.putText(frame_bgr, f"{frame_bgr.shape[1]}x{frame_bgr.shape[0]}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)

            # Display frame
            if not args.no_gui:
                try:
                    cv2.imshow("IMX708 Live (q to quit)", frame_bgr)
                except cv2.error:
                    # GUI not available; fallback to headless handling
                    print("No GUI available — switching to headless mode.")
                    args.no_gui = True

            # When headless, optionally throttle loop to approx target fps to avoid full CPU usage
            if args.no_gui:
                # just sleep small amount to aim for target fps
                sleep_time = max(0.0, (1.0 / max(1.0, args.target_fps)) - (time.time() - now))
                if sleep_time > 0:
                    time.sleep(sleep_time)
            else:
                # handle keypress
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("Quit key pressed.")
                    break

        print("Exiting main loop.")

    except KeyboardInterrupt:
        print("Interrupted by user (KeyboardInterrupt).")

    finally:
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        # Print final FPS stats
        if len(rolling_times) > 0:
            avg_roll = sum(rolling_times) / len(rolling_times)
            print(f"Average recent FPS (rolling): {avg_roll:.2f}")
        if ema_fps:
            print(f"Final EMA FPS: {ema_fps:.2f}")

        print("Camera stopped, resources released.")

if __name__ == "__main__":
    main()

