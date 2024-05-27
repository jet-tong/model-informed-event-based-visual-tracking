import cv2
import numpy as np

# Pre-compute the logarithmic values for all possible pixel intensities
lut = np.log1p(np.arange(256)).astype(np.float32)


def event_generator(
    frame, frame_gray, prev_frame_gray, IS_LOG_INTENSITY=True, THRESHOLD=None
):
    if THRESHOLD is None:
        THRESHOLD = 0.5 if IS_LOG_INTENSITY else 20

    if IS_LOG_INTENSITY:
        # frame_gray = frame_gray.astype(np.float32)
        # prev_frame_gray = prev_frame_gray.astype(np.float32)
        # diff = np.log1p(frame_gray) - np.log1p(prev_frame_gray)
        frame_gray_log = cv2.LUT(frame_gray, lut)
        prev_frame_gray_log = cv2.LUT(prev_frame_gray, lut)
        diff = frame_gray_log - prev_frame_gray_log
    else:
        # frame_gray = frame_gray.astype(np.float32)
        # prev_frame_gray = prev_frame_gray.astype(np.float32)
        diff = frame_gray - prev_frame_gray

    # Identify positive and negative events
    pos_events = diff > THRESHOLD
    neg_events = diff < -THRESHOLD

    # # Black / White Approach
    # event_img_gray = np.full_like(
    #     frame_gray, 128, dtype=np.uint8
    # )  # Fill with gray value
    # event_img_gray[pos_events] = 255  # Positive events in white
    # event_img_gray[neg_events] = 0  # Negative events in black
    # event_img = cv2.cvtColor(event_img_gray, cv2.COLOR_GRAY2BGR)

    # # Red / Blue Approach
    event_img = np.full_like(frame, 128)  # Fill with gray value
    event_img[pos_events] = [255, 0, 0]  # Positive events (ON) in blue
    event_img[neg_events] = [0, 0, 255]  # Negative events (OFF) in red

    # # cv2.merge() approach
    # # Initialize with 128 for the gray background
    # B_channel = np.full_like(frame_gray, 128, dtype=np.uint8)
    # G_channel = np.full_like(frame_gray, 128, dtype=np.uint8)
    # R_channel = np.full_like(frame_gray, 128, dtype=np.uint8)
    # # Set B channel to 255 where there are positive events and R channel to 255 where there are negative events
    # # The G channel remains the same to maintain the gray color
    # B_channel[pos_events] = 255  # Positive events will be blue
    # R_channel[neg_events] = 255  # Negative events will be red
    # # Merge the single channels into a BGR image
    # event_img = cv2.merge([B_channel, G_channel, R_channel])

    return event_img, pos_events, neg_events


def store_updated_events(frame_gray, prev_frame_gray, pos_events, neg_events):
    # Update pixels for pos_events or neg_events
    combined_events = pos_events | neg_events
    prev_frame_gray[combined_events == 1] = frame_gray.copy()[combined_events == 1]
    return prev_frame_gray
