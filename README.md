# Always Winning Plinko Project

Computer vision system that detects a blue ping-pong ball as it falls down a Plinko board, computes the ball's normalized X position across the board, and sends that value to an Arduino over serial to drive a catcher carriage using closed-loop control.

## Vision System (Raspberry Pi 4 + Arducam IMX708)

### Description
- Computer-vision pipeline that detects a blue ping-pong ball on the Plinko board, computes its normalized X position across the board (0.0 = left edge, 1.0 = right edge), and sends that value to the Arduino over serial as the catcher's setpoint.

### Prerequisites
- Raspberry Pi 4 with Raspberry Pi OS (or equivalent). Follow Arducam model-specific driver/overlay instructions if required.
- Python 3.8+ recommended. Use a virtual environment for installing Python dependencies.

#### Camera configuration
- Add the following to `/boot/config.txt` on the Pi and reboot:
  ```ini
  camera_auto_detect=0
  dtoverlay=imx708
  ```

#### System packages (on the Pi)
- Install the following system packages on the Pi
    ```bash
    sudo apt update
    sudo apt install -y libcamera-apps v4l-utils python3-picamera2 python3-opencv
    ```

#### Python dependencies
- From the project root, create/activate a venv and install dependencies:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```     

### Camera calibration (undistortion)
**NOTE FOR TEAM: VISION CODE DOES NOT CURRENTLY USE THE `.npz` CALIBRATION THIS WILL NEED TO BE DONE SOON AND CHANGES WILL NEED TO BE MADE TO `plinko_vision.py`**

- Run the calibration script to generate `camera_calibration_picamera2.npz`:
    ```bash
    python3 vision/camera_calibration.py
    ```
- Place the resulting `camera_calibration_picamera2.npz` file in the project (alongside the vision scripts) so the vision code can load it to undistort frames. Calibration reduces lens distortion and improves homography accuracy.

### Serial protocol (defaults)
- Port: `/dev/ttyACM0`
- Baud: `9600`
- Message format sent by the main vision script: a single normalized X value followed by newline, for example:
    ```
    0.4213\n
    ```
- Note: `vision/test_serial_communication.py` is a test tool — it was previously sending `x,y\n` pairs; the main `plinko_vision.py` sends a single normalized X value. The test script has been updated to match the main script format.

### How to run (quick)
- Preview camera and click board corners (Top-Left, Top-Right, Bottom-Right, Bottom-Left):
    ```bash
    python3 vision/test_video_feed.py
    ```
- Test blob detection (blue ball):
    ```bash
    python3 vision/test_blob_detection.py
    ```
- Test board coordinate calibration:
    ```bash
    python3 vision/test_board_coordinate.py
    ```
- Test serial comms (sends a dummy normalized x value):
    ```bash
    python3 vision/test_serial_communication.py
    ```
- Full vision + serial output:
    ```bash
    python3 vision/plinko_vision.py
    ```

### Pipeline (high-level)
- Preview at `PREVIEW_SIZE` for manual corner selection (build homography).
- Detection runs at `DETECT_SIZE` for speed (HSV threshold → morphology → contour selection).
- Compute centroid → map to board coordinates via homography → compute normalized X in [0,1].
- Send normalized X to Arduino over serial as a setpoint for catcher control.

### Troubleshooting tips
- No detections: tune HSV ranges `BLUE_LOWER` / `BLUE_UPPER` in `vision/plinko_vision.py` to match your lighting and ball color.
- Distorted frames: run `python3 vision/camera_calibration.py` and ensure `camera_calibration_picamera2.npz` is present.
- Serial errors / port not found: confirm device with `dmesg | grep tty` or `ls /dev/tty*` and ensure the Arduino is connected; match the baud rate used by the Arduino sketch.

## Control System (Arduino UNO R3 + DC Motor with encoder)