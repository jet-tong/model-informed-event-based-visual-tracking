import time

import cv2
import numpy as np
from capture_setup import capture_setup
from event_camera_simulator import event_generator, store_updated_events
from hand_tracker import IndexTracker
from kalman_filter import KalmanFilter
from line_profiler import LineProfiler
from local_correlation_search import find_optimal_offset

FRAMES_PER_MP = 10
TRACK_LENGTH = 50


def main():
    cap = capture_setup(0)  # 0 for webcam, "path/to/video" for video; fps=60/120/260

    index_tracker = IndexTracker()

    kf = KalmanFilter(dt=1)
    kf_track = []
    mp_track = []

    # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    # out = cv2.VideoWriter("output.avi", fourcc, current_fps, (int(width), int(height)))

    ret, prev_frame = cap.read()
    prev_frame = cv2.resize(
        prev_frame, (prev_frame.shape[1] // 2, prev_frame.shape[0] // 2)
    )  # Resize

    # Initialize hand model display,
    index_tracker.index_mask = np.zeros(prev_frame.shape[:2], dtype=np.uint8)
    roi_display = np.zeros(
        (prev_frame.shape[0], prev_frame.shape[0], 3), dtype=np.uint8
    )

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    start_time = time.perf_counter()
    frame_count = 0
    total_processing_time_ms = 0
    avg_fps = 0

    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))  # Resize
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # EVENT_GENERATOR
        event_img, pos_events, neg_events = event_generator(
            frame, frame_gray, prev_frame_gray, IS_LOG_INTENSITY=True, THRESHOLD=0.05
        )

        # -----------------------------------
        # Index Finger Tracking with Kalman Filter
        # -----------------------------------

        # Kalman filter predict and correct
        kf.predict()
        offset = None  # So I can display event tracking offset in the GUI
        best_num_of_pixels = 0
        if frame_count % FRAMES_PER_MP == 0:
            # MP Update
            index_tracker.index_section = index_tracker.get_index_section(frame)
            if index_tracker.index_section:
                mp_track.append(index_tracker.index_section[0])
                kf.correct(index_tracker.index_section[0], measurement_type="mp")
        else:
            # EV Update
            if index_tracker.index_mask is not None:
                kf_integer_center = kf.get_integer_state()
                initial_rect = index_tracker.get_index_roi_rect(*kf_integer_center)
                combined_events = pos_events | neg_events

                # Find the best offset using the local search algorithm
                offset, best_num_of_pixels = find_optimal_offset(
                    combined_events,
                    index_tracker.index_mask,
                    initial_rect,
                    max_steps=10,
                    delta=1,
                    verbose=False,
                )

                # Update kf_integer_center with the best offset found
                updated_kf_integer_center = (
                    kf_integer_center[0] + offset[0],
                    kf_integer_center[1] + offset[1],
                )

                # Draw updated_kf_integer_center on event_img
                cv2.circle(event_img, updated_kf_integer_center, 2, (255, 255, 255), -1)

                # Correct the Kalman filter with the updated center
                if best_num_of_pixels > 50:
                    kf.correct(updated_kf_integer_center, measurement_type="ev")

        # Update PIP point based on change in the TIP point, and get ROI
        kf_integer_center = kf.get_integer_state()
        kf_track.append(kf_integer_center)
        if index_tracker.index_section:
            kf_index_section = (
                kf_integer_center,
                (
                    kf_integer_center[0]
                    - index_tracker.index_section[0][0]
                    + index_tracker.index_section[1][0],
                    kf_integer_center[1]
                    - index_tracker.index_section[0][1]
                    + index_tracker.index_section[1][1],
                ),
            )
            index_tracker.update_index_mask(frame, kf_index_section)
            index_tracker.rect = index_tracker.get_index_roi_rect(*kf_index_section[0])

        # -----------------------------------
        # Track history
        # -----------------------------------
        # Manage tracking history
        if len(kf_track) > TRACK_LENGTH:
            kf_track.pop(0)
        if len(mp_track) * FRAMES_PER_MP > TRACK_LENGTH:
            mp_track.pop(0)

        # Draw the tracking history
        roi_display = event_img.copy()
        for point in kf_track:
            cv2.circle(roi_display, point, 1, (0, 255, 0), -1)  # Green
            cv2.circle(event_img, point, 1, (0, 255, 0), -1)  # Green
        for point in mp_track:
            cv2.circle(roi_display, point, 1, (255, 0, 0), -1)  # Blue
            cv2.circle(event_img, point, 1, (255, 0, 0), -1)  # Blue

        # -----------------------------------
        # ROI and Kalman Filter
        # -----------------------------------

        if index_tracker.index_section:
            # Draw contour of index_mask
            contours, _ = cv2.findContours(
                index_tracker.index_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(roi_display, contours, -1, (0, 255, 0), 1)

        # Crop ROI
        rect = index_tracker.get_index_roi_rect(*kf_integer_center)
        roi_display = index_tracker.crop_index_roi(roi_display, rect)

        if roi_display.size > 0:
            roi_display = cv2.resize(roi_display, (frame.shape[0], frame.shape[0]))
        else:
            roi_display = np.zeros((frame.shape[0], frame.shape[0], 3), dtype=np.uint8)

        # Timer for processing time
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000
        total_processing_time_ms += processing_time_ms
        start_time = end_time
        frame_count += 1

        # Calc average FPS every 100 frames
        if frame_count >= 100:
            # Calculate average FPS over the last 100 frames
            avg_fps = 100000 / total_processing_time_ms
            print(f"Average FPS over last 100 frames: {avg_fps:.2f}")

            # Reset counters
            total_processing_time_ms = 0
            frame_count = 0

        # Display FPS on the image
        cv2.putText(
            img=event_img,
            text=f"Calc FPS: {avg_fps:.1f}",
            org=(0, 15),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        # Convert binary mask to 3-channel image
        index_mask_colour = cv2.cvtColor(index_tracker.index_mask, cv2.COLOR_GRAY2BGR)
        # Display the visualization
        # combined_display = np.hstack((event_img, index_mask_colour, roi_display))
        combined_display = np.hstack((frame, event_img, roi_display))
        cv2.imshow("Combined Display", combined_display)
        # out.write(event_img)

        # GUI control for exiting
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Update pixels for pos_events or neg_events
        prev_frame_gray = store_updated_events(
            frame_gray, prev_frame_gray, pos_events, neg_events
        )

    # out.release()
    index_tracker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    profiler = LineProfiler()
    profiler.add_function(main)
    profiler.add_function(find_optimal_offset)
    profiler.enable_by_count()
    main()
    profiler.print_stats()
