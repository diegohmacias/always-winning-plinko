
# Always Winning Plinko

Computer vision system that tracks an orange ball falling through a Plinko board and outputs the balls' (x,y) positions with respect to the Plinko board coordinate frame.

This repository is primarily intended to run on a Raspberry Pi 4 paired with an Arducam IMX708 module. Where behavior differs on other platforms or camera modules, notes are provided below.

## Setup (Raspberry Pi 4 + Arducam IMX708)

Hardware the team used (CanaKit Raspberry Pi 4 kit):

- Raspberry Pi 4
- MicroSD card (pre-loaded with NOOBS)
- USB-C power adapter and supply
- Micro HDMI cable
- Case

The team also used an Arducam IMX708 camera module connected via the ribbon cable to the Pi's camera connector.

Initial OS and remote access:

- Flash / insert the MicroSD with Raspberry Pi OS (or use the NOOBS card included in the kit).
- Optionally set up headless access with RealVNC or SSH for remote desktop/control.

IMX708-specific configuration

Before using the Arducam IMX708, add or modify these lines in `/boot/config.txt` on the Pi:

```ini
camera_auto_detect=0
dtoverlay=imx708
```

Reboot the Pi after making these changes to ensure the overlay is applied.

Install the following packages (example list used by the team):

```bash
sudo apt update
sudo apt install -y libcamera-apps v4l-utils python3-picamera2 python3-opencv
```

Arducam note: Some Arducam models require vendor drivers or SDKs. For the IMX708, follow Arducam's model-specific installation guide if additional driver steps are required.

Quick sanity checks (once camera and packages are installed):

```bash
# List video devices
v4l2-ctl --list-devices

# Quick libcamera preview (3 seconds)
libcamera-hello --timeout 3000

# Capture a still image
libcamera-still -o test.jpg
```

If those commands succeed and you can preview/capture, OpenCV-based scripts should be able to access the camera (possibly as `/dev/video0` or via libcamera/Picamera2 bridges).

## Git Clone

Clone the repository (HTTPS):

```bash
git clone https://github.com/diegohmacias/always-winning-plinko.git
cd always-winning-plinko
```

Or with SSH:

```bash
git clone git@github.com:diegohmacias/always-winning-plinko.git
cd always-winning-plinko
```

Note: You may need to create a personal access token for Git credentials when using HTTPS.

## Python environment & dependencies

This project requires Python 3.8+ and a few common scientific/CV packages.

Recommended: create and use a virtual environment when developing off-device:

```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
```

Install Python dependencies from the included `requirements.txt`:

```bash
pip install -r requirements.txt
```

Note: On Raspberry Pi the Picamera2 bindings are usually installed from apt (`sudo apt install python3-picamera2`) rather than pip. See the Setup section above.

Common system packages (Raspberry Pi):

```bash
sudo apt update
sudo apt install -y libatlas-base-dev libjpeg-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev
# Optional: libcamera and python camera bindings
sudo apt install -y libcamera-apps python3-picamera2 v4l-utils
```

## How to run the scripts

Run a script directly with Python (examples):

```bash
# Run individual test scripts
python3 tests/test_ball_detection.py
python3 tests/test_camera.py
python3 tests/test_video.py
```

The `tests/` scripts are simple runnable Python files (they are not necessarily `pytest`-style tests).

## Using a different platform or camera module

If you are not using a Raspberry Pi 4 with an Arducam IMX708, you may need to adjust:

- `/boot/config.txt` overlay: other camera modules use different dtoverlay values (or may rely on automatic detection). Do not set `dtoverlay=imx708` unless you have that sensor.
- Driver and SDK install steps: follow the vendor docs for your specific camera module.
- Camera interface stack: some platforms use `v4l2` directly, others use `libcamera`/`picamera2`. Adjust how the camera is opened in code (OpenCV VideoCapture index, or using Picamera2 APIs).
- Resolution and framerate: IMX708 defaults may differ from other modules; update capture parameters in scripts if necessary.
- Device node: camera may appear at a different `/dev/videoX` index; use `v4l2-ctl --list-devices` to find the correct node.

If you plan to develop primarily off your laptop, the camera-specific packages and system-level steps are unnecessary; you can still run and test non-camera parts of the code.



