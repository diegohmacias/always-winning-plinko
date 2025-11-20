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

### Description
- The Arduino runs a position-control loop for the catcher carriage. The Pi vision system sends a normalized X position (0.0–1.0) over serial; the Arduino converts that to a physical target (inches) and encoder counts, then uses a PID controller to drive a DC motor with encoder feedback to the target position.

### Hardware and pinout
- Typical hardware used in sketches:
    - Arduino Uno R3 (or compatible)
    - DC motor with motor driver (PWM + direction pins)
    - Quadrature encoder on the motor
    - Motor pins used:
        - `ENA` (PWM) = pin `5`
        - `IN1` = pin `7`
        - `IN2` = pin `8`
        - Encoder A = pin `2` (interrupt)
        - Encoder B = pin `3` (interrupt)

### Main sketch behavior
- **Homing:** on power-up the sketch performs a simple homing sequence (drives slowly toward the left hard stop for a fixed time) and then resets the encoder to zero. This establishes a reproducible physical zero position.

- **Serial receive:** listens on `Serial` (9600 baud) for a newline-terminated floating value representing `x_norm` (0.0..1.0). Example received message:
    ```txt
    0.4213\n
    ```

- **Conversion:** `x_norm` is multiplied by an explicitly configured `BOARD_WIDTH_INCHES` to get a target distance, then converted to encoder counts using `COUNTS_PER_INCH`.

- **Safety:** the sketch applies soft limits so targets cannot exceed the calibrated board length.

- **PID control:** a PID loop uses encoder counts as feedback and outputs a signed control value which is applied to the motor driver (mapped to PWM and direction). The sketch uses `PID_v1` with configurable `Kp`, `Ki`, `Kd`.

- **Telemetry:** the sketch prints human-readable debug lines (e.g., `Norm: 0.5 -> Inch: 10 -> Setpoint: 5000`) and occasional position/PWM lines while moving.

### Calibration and tuning
- **Manual calibration sketch:** `arduino/always-winning-plinko-calibration/always-winning-plinko-calibration.ino` is intended for manual measurement

- **Procedure:**
    1. Upload calibration sketch and open Serial Monitor at 9600 baud.
    2. Move the carriage to the physical zero (left edge) and send `z` to reset encoder counts to 0.
    3. Move the carriage to a known distance (e.g., 1.0 in) and type that distance into the serial monitor; the sketch will echo the encoder count for that physical distance.
    4. Use the printed encoder counts to compute `COUNTS_PER_INCH` (counts / inches) and update the main sketch constants.

- **PID tuning:** tune `Kp`, `Ki`, `Kd` in the main sketch for smooth, responsive motion. Start with `Ki = 0`, increase `Kp` until you get reasonable response, then add small `Ki` to eliminate steady-state error, then `Kd` to reduce overshoot.

### Test and debug sketches
- `arduino/alway-winning-plinko-test/alway-winning-plinko-test.ino` is a simple echo test: it reads newline-terminated strings and echoes them back. Use this with `vision/test_serial_communication.py` to verify serial connectivity and that the Pi->Arduino link is functional.

### Serial protocol (Vision <-> Arduino)
- Vision (Pi) sends a single floating value representing normalized board X, terminated with `\n`, e.g. `0.4213\n`.
- Arduino expects the above format (the main sketch parses `toFloat()` from a `readStringUntil('\n')`).
- Arduino prints human-readable status messages which the Pi can read optionally (the vision scripts will read and print Arduino responses if available). The protocol is intentionally simple (text-based) to make debugging over USB-serial easier.

### Safety notes
- The homing routine in the example sketch drives the motor for a fixed delay to reach the hard stop. Ensure the carriage has a physical, safe hard stop and adjust the homing duration to avoid mechanical stress.

### Troubleshooting tips
- If Arduino doesn't respond: ensure both sides use `9600` baud, verify the serial device (e.g. `/dev/ttyACM0`), and test with the echo sketch + `vision/test_serial_communication.py`.
- If the carriage doesn't move to the correct physical location: verify `COUNTS_PER_INCH` from the calibration sketch and ensure `BOARD_WIDTH_INCHES` matches the physical board.
- If movement is unstable: tune PID, check for mechanical binding, and confirm encoder wiring and interrupts are functioning.
