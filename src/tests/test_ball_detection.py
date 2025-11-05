#!/usr/bin/env python3
"""
test_ball_detection.py
Test runner that uses scripts.ball_tracker.BallTracker to run live detection.
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import signal
from src.scripts.ball_tracker import BallTracker

def signal_handler(sig, frame):
    # forward to allow graceful shutdown
    raise KeyboardInterrupt

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # choose sizes: capture size vs processing size
    cap_w, cap_h = 640, 480    # full-size shown in preview
    proc_w, proc_h = 320, 240  # processing size -> faster

    bt = BallTracker(cap_width=cap_w,
                     cap_height=cap_h,
                     proc_w=proc_w,
                     proc_h=proc_h,
                     process_every=1,
                     smooth_alpha=0.35,
                     hsv_range=(5,140,120,18,255,255),
                     show_mask=False,
                     use_trackbars=True)  # helpful to tune initially

    try:
        bt.start()
        # Run blocking preview. Press 'q' to quit; 't' toggles trackbars if available.
        bt.run_preview(window_name="Ball Detection (press q to quit)", show_mask=False, process_every=1)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        bt.stop()
        print("Stopped tracker. Exiting.")

if __name__ == "__main__":
    main()
