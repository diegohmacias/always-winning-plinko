#!/user/bin/env python3
"""
IMX708 live preview using Picamera2 + OpenCV
Press 'q' to quit
"""

from picamera2 import Picamera2
import cv2
import time

def main():
    picam2 = Picamera2() # Initialize Picamera2

    config = picam2.create_preview_configuration(main={"size": (640, 480)}) # set preview configuration
    picam2.configure(config)                                                 # configure the camera
    picam2.start()                                                           # start the camera
    time.sleep(2)                                                            # let sensor/startup settle

    # fps measurement helpers
    prev_time = time.time()
    fps = 0.0

    # capture one frame to get correct shapem or read shape each iteration
    ret_frame = picam2.capture_array()
    if ret_frame is None:
        raise RuntimeError("Failed to capture initial frame")
    
    # note: frame.shape -> (height, width, channels)
    height, width = ret_frame.shape[:2]

    try:
        while True:
            frame = picam2.capture_array()                     # returns a numpy array in RGB order
            if frame is None:
                print("Warning: Failed to capture frame")
                continue
            
            # update wifth/height in case of dynamic resolution change
            height, width = frame.shape[:2]
            
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # convert RGB to BGR for OpenCV display

            # calculate fps
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt > 0 else 0.0 

            # build text strings
            text_res = f"Resolution: {width}x{height}"
            text_fps = f"FPS: {fps:.1f}"

            # text settings
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.6
            thickness = 2

            # compute text sizes (to align right side)
            (w_res, h_res), _ = cv2.getTextSize(text_res, font, scale, thickness)
            (w_fps, h_fps), _ = cv2.getTextSize(text_fps, font, scale, thickness)

            # positions (top-right corner with padding)
            padding = 10
            x_res = frame_bgr.shape[1] - w_res - padding
            y_res = padding + h_res

            x_fps = frame_bgr.shape[1] - w_fps - padding
            y_fps = y_res + h_fps + 5  

            # draw text (white text + black border)
            cv2.putText(frame_bgr, text_res, (x_res, y_res), font, scale, (0, 0, 0), thickness+2)
            cv2.putText(frame_bgr, text_res, (x_res, y_res), font, scale, (255, 255, 255), thickness)

            cv2.putText(frame_bgr, text_fps, (x_fps, y_fps), font, scale, (0, 0, 0), thickness+2)
            cv2.putText(frame_bgr, text_fps, (x_fps, y_fps), font, scale, (255, 255, 255), thickness)

            cv2.imshow("IMX708 Live Preview", frame_bgr)       # display the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break                                           # exit on 'q' key press
    except KeyboardInterrupt:
        pass
    finally:
        picam2.stop()            # stop the camera
        cv2.destroyAllWindows()  # close all OpenCV windows

if __name__ == "__main__":
    main()


