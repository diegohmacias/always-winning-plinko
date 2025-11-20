#!/usr/bin/env python3
import serial
import time

# connect to Arduino
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1) # adjust baudrate and port as needed
ser.flush()                                          # flush input/output buffers

# give Arduino time to reset
time.sleep(1)

# test loop - send ball positions
while True:
    # simulate ball position
    ball_x = 123.45
    ball_y = 678.90
    
    # format message
    message = f"{ball_x},{ball_y}\n"
    
    # send to Arduino
    print(f"Sending: {message.strip()}")
    ser.write(message.encode('utf-8'))

    # read Arduino's response
    if ser.in_waiting > 0:
        response = ser.readline().decode('utf-8').rstrip()
        print(f"Arduino says: {response}")
    
    time.sleep(1)  # send once per second for testing