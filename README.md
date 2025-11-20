# Always Winning Plinko Project

Computer vision system that uses blob detection to detect and output a blue ping ball's normalized x position with respect to a Plinko board as it falls down the board. This system also communicates with an Arduino that via serial communication to drive a catcher that sits on a carriage driven by a belt drive via a DC motor + encoder using PID control.

## Vision Setup (Raspberry Pi 4 + Arducam IMX708)

### Hardware used for the project:
- CanaKit Raspberry Pi 4 Kit including:
  - Raspbery Pi 4
  - MicroSD card (pre-loaded with NOOBS)
  - USB-C power adapter and supply
  - Micro HDMI cable
- Bluetooth keyboard + mouse  
- Arducam IMX708 camera module w/ ribbon cable

### Initial OS and remote access:
- Flash / insert the MicroSD with Raspberry PI OS (or use the NOOBS card included in the kit)
- Optionally set up headless access with RealVNC or SSH for remote desktop/control

### Arducam IMX708 Configuration/Setup:
- Before using the Arducam IMX708, add these lines in `/boot/config.txt` on the Pi:
    ```ini
    camera_auto_detect=0
    dtoverlay=imx708
    ```
- Reboot the Pi after making these changes to ensure the overlay is applied

### Package Dependencies
- Install the following packages on the Pi:
    ```bash
    sudo apt update && sudo apt install -y libcamera-apps v4l-utils python3-picamera2 python3-opencv
    ```
- **Note:** Depending on Arducam model, you may require vendor drivers or SDKs. Follow Arducam's model-specific installation guide if additional driver steps are required

### Python Dependencies
- Install the following with pip:
    ```bash
    pip install numpy
    pip install opencv-python
    pip install imutils
    ```

## Running the test scripts
Run a script directly with Python
- Testing video feed
    ```bash
    python3 vision/test_video_feed.py
    ```
- Testing blob detection (specifically using blue ping pong ball)
    ```bash
    python3 vision/test_blob_detection.py
    ```
- Testing board coordinate calibration (uses board dimensions)
    ```bash
    python3 vision/test_board_coordinate.py
    ```
- Testing serial communication with Arduino
    ```bash
    python3 vision/test_serial_communication.py
    ```
- For calibration of camera (TBD)
    ```bash
    python3 vision/camera_calibration.py
    ```
    - Calibration of camera still needs to be done to fix wide angle lens distortion
    - Output of script should be calibration file `camera_calibration_picamera2.npz` along with images placed in `/calib_images`

## Running vision system
- Run the complete vision system of the project using:
    ```bash
    python3 vision/plinko_vision.py
    ```
    - Combines algorithms/logic from the above test scripts for full vision system functionality
    - Follow terminal instructions to calibrate board coordinate 
    - Script outputs ball's normalized x position [0,1] and sends it to Arduino as setpoint for control of catcher