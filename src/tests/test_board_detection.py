#!/usr/bin/env python3
"""
src/tests/test_board_detection.py

Test runner for src.scripts.board_detection.BoardDetector.

This test file lives inside src/tests/ so we add the repo root to sys.path
to allow imports like `from src.scripts.board_detection import BoardDetector`.
"""

import sys
import os
import signal

# ---- robustly add repo root and src to sys.path ----
THIS_DIR = os.path.dirname(os.path.abspath(__file__))        # .../repo/src/tests
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))  # go up two -> repo root
SRC_DIR = os.path.abspath(os.path.join(REPO_ROOT, "src"))

# add repo root first (so `import src...` works), then add src as fallback
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
# ----------------------------------------------------

# Now import the class (same pattern as your ball test)
from src.scripts.board_detection import BoardDetector

def signal_handler(sig, frame):
    raise KeyboardInterrupt

def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    cap_w, cap_h = 640, 480

    bd = BoardDetector(board_w_in=12.375,
                       board_h_in=12.375,
                       cap_width=cap_w,
                       cap_height=cap_h,
                       enable_tracking=True)

    try:
        bd.start(device_index=0)
        bd.run_preview(window_name="Board Detection (press q to quit)")
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        bd.stop()
        print("Stopped. Exiting.")

if __name__ == "__main__":
    main()