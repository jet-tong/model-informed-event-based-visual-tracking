import time

import cv2


def capture_setup(source=0, fps=None, verbose=True):
    try:
        # If source is an integer, assume 260fps camera and use cv2.CAP_DSHOW
        int_source = int(source)
        cap = cv2.VideoCapture(int_source, cv2.CAP_DSHOW)
    except ValueError:
        # If source is a string, assume video file and use default
        cap = cv2.VideoCapture(source)

    time.sleep(1)

    if not cap.isOpened():
        print(f"Unable to open camera source: {source}")
        exit()

    # Camera Settings
    # Only works when using cv2.CAP_DSHOW from testing
    # 1920 X 1080  @60fps
    # 1280 X 720   @120fps
    # 640  X 360   @260fps (default)
    # cap.set(cv2.CAP_PROP_FPS, 260) # Doesn't work, use resolution

    if fps:
        if fps == 60:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        elif fps == 120:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        elif fps == 260:  # also default
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    current_fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if verbose:
        print("----- Camera Settings -----")
        if cap.isOpened():
            print("Camera is available!")
        print(f"Camera Source: {source}")
        print(f"Camera FPS: {current_fps if current_fps else 'fps not available'}")
        print(f"Resolution: {int(width):4d} x {int(height):4d}")
        print("--------------------------")

    return cap
