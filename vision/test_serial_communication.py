#!/usr/bin/env python3
"""Simple test script: send a dummy normalized X value to the Arduino.

This script matches the message format used by `vision/plinko_vision.py` (a single
floating value followed by a newline, e.g. `0.4213\n`). Use it to verify serial
connectivity and receiver behaviour on the Arduino side.
"""

import serial
import time

# connect to Arduino (adjust port/baud as needed)
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    ser.flush()
    print(f"Opened serial port {SERIAL_PORT} at {BAUD_RATE} baud")
except Exception as e:
    print(f"Error opening serial port: {e}")
    raise

# give Arduino time to reset
time.sleep(1)

# test loop - send a normalized x value every second
while True:
    # simulate normalized ball X in [0,1]
    x_norm = 0.5000
    message = f"{x_norm:.4f}\n"

    # send to Arduino
    print(f"Sending: {message.strip()}")
    ser.write(message.encode('utf-8'))

    # read Arduino's response (if any)
    if ser.in_waiting > 0:
        response = ser.readline().decode('utf-8').rstrip()
        print(f"Arduino says: {response}")

    time.sleep(1)  # send once per second for testing