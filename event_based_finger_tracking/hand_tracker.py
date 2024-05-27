import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.hand_connections = tuple(self.mp_hands.HAND_CONNECTIONS)
        self.hand_mask = None
        self.dilated_hand_mask = None
        self.hand_contour = None
        self.hand_model_display = None
        self.hand_landmarks = None

    def update_hand_landmarks(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False  # Performance boost
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        self.hand_landmarks = results.multi_hand_landmarks

    def update_hand_contour(self, frame):
        # Initialize an empty mask with the same size as the frame.
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Check if any hands are detected
        if not self.hand_landmarks:
            self.hand_model_display = np.zeros_like(frame)
            return

        for hand_landmarks in self.hand_landmarks:
            # Draw connections including the extra connection between [1] and [5]
            for connection in self.hand_connections + (
                (
                    self.mp_hands.HandLandmark.THUMB_CMC,
                    self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                ),
            ):
                start_idx, end_idx = connection
                x1, y1 = int(
                    hand_landmarks.landmark[start_idx].x * frame.shape[1]
                ), int(hand_landmarks.landmark[start_idx].y * frame.shape[0])
                x2, y2 = int(hand_landmarks.landmark[end_idx].x * frame.shape[1]), int(
                    hand_landmarks.landmark[end_idx].y * frame.shape[0]
                )

                cv2.line(mask, (x1, y1), (x2, y2), color=(255), thickness=5)

        # Use circular dilation to better maintain the shape of the hand
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (6, 6)
        )  
        mask = cv2.dilate(mask, kernel, iterations=2)
        hand_model_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If you only want to keep the largest contour (which is presumably the hand):
        if contours:
            # Find the largest contour based on area
            max_contour = max(contours, key=cv2.contourArea)
            # Draw only the largest contour
            cv2.drawContours(hand_model_display, [max_contour], -1, (0, 255, 0), 2)
            self.hand_model_display = hand_model_display
            self.hand_contour = max_contour

        self.hand_mask = mask
        self.dilated_hand_mask = cv2.dilate(mask, kernel, iterations=3)

    def process(self, frame):
        self.update_hand_landmarks(frame)
        self.update_hand_contour(frame)

    def close(self):
        self.hands.close()

    # ----------------------------------------------------------------------------
    # OLD CODE BELOW
    # ----------------------------------------------------------------------------

    def draw_landmarks_old(self, frame):
        # frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)  # flip for mirror view
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False  # Performance boost
        results = self.hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

        return frame

    def process_backup(self, frame):
        # frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)  # flip for mirror view
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False  # Performance boost
        results = self.hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Initialize an empty mask with the same size as the frame.
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Check if any hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw connections including the extra connection between [1] and [5]
                for connection in self.hand_connections + (
                    (
                        self.mp_hands.HandLandmark.THUMB_CMC,
                        self.mp_hands.HandLandmark.INDEX_FINGER_MCP,
                    ),
                ):
                    start_idx = connection[0]
                    end_idx = connection[1]

                    start_point = hand_landmarks.landmark[start_idx]
                    end_point = hand_landmarks.landmark[end_idx]

                    x1, y1 = int(start_point.x * frame.shape[1]), int(
                        start_point.y * frame.shape[0]
                    )
                    x2, y2 = int(end_point.x * frame.shape[1]), int(
                        end_point.y * frame.shape[0]
                    )

                    cv2.line(mask, (x1, y1), (x2, y2), color=(255), thickness=5)

        # Use circular dilation to better maintain the shape of the hand
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (6, 6)
        ) 
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.hand_model_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        # If you only want to keep the largest contour (which is presumably the hand):
        if contours:
            # Find the largest contour based on area
            max_contour = max(contours, key=cv2.contourArea)
            # Draw only the largest contour
            cv2.drawContours(self.hand_model_display, [max_contour], -1, (0, 255, 0), 2)
            # Or if you want to use this contour for comparison later, store it in a variable:
            self.hand_contour = max_contour


class IndexTracker:
    def __init__(self):
        # self.mp_drawing = mp.solutions.drawing_utils
        # self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,  # 0: Lite, 1: Full
        )

        self.index_section = None
        self.index_mask = None
        self.rect = None

        # Variables for the index finger model
        self.index_finger_thickness = 10

    def get_index_coords(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = self.hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            index_finger_tip = results.multi_hand_landmarks[0].landmark[
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]  # or .landmark[8]

            return (
                int(index_finger_tip.x * frame.shape[1]),
                int(index_finger_tip.y * frame.shape[0]),
            )

        return None

    def get_index_section(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = self.hands.process(frame)
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            index_finger_tip = results.multi_hand_landmarks[0].landmark[
                self.mp_hands.HandLandmark.INDEX_FINGER_TIP
            ]  # or .landmark[8]

            index_finger_mcp = results.multi_hand_landmarks[0].landmark[
                self.mp_hands.HandLandmark.INDEX_FINGER_PIP
            ]  # or .landmark[7]

            # return both
            return (
                (
                    int(index_finger_tip.x * frame.shape[1]),
                    int(index_finger_tip.y * frame.shape[0]),
                ),
                (
                    int(index_finger_mcp.x * frame.shape[1]),
                    int(index_finger_mcp.y * frame.shape[0]),
                ),
            )

        return None

    def update_index_mask(self, frame, index_section):  # TODO: Make this BINARY MASK!
        mask = np.zeros(
            frame.shape[:2], dtype=np.uint8
        )  # TODO: Store blank mask in __init__ instead

        tip, mcp = index_section
        cv2.line(
            mask, tip, mcp, color=(255), thickness=self.index_finger_thickness
        )  # Arbitrary thickness

        self.index_mask = mask

    def process(self, frame):
        self.index_section = self.get_index_section(frame)
        if self.index_section:
            self.update_index_mask(frame, self.index_section)
            self.rect = self.get_index_roi_rect(*self.index_section[0])

    def get_index_roi_rect(self, index_x, index_y, size=40):
        return (
            int(index_x - size // 2),
            int(index_y - size // 2),
            int(index_x + size // 2),
            int(index_y + size // 2),
        )

    def crop_index_roi(self, frame, rect):
        x1, y1, x2, y2 = rect
        return frame[y1:y2, x1:x2]

    def draw_index_roi_rect(self, frame, rect):
        x1, y1, x2, y2 = rect
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def close(self):
        self.hands.close()
