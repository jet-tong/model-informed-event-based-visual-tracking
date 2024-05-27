# Event-Based Finger Tracking

## Overview
This project proposes to bridge this gap by leveraging simulated event-based vision via the usage of a high frame-rate camera. To demonstrate the potential of this approach, a lightweight, high frame-rate, and real-time finger tracking system is implemented by integrating simulated event-based vision with MediaPipe's hand tracking module via sensor fusion with a Kalman filter. 

## Features
- **Event Camera Simulation**: Simulates event-based vision data stream.
- **Hand Tracking**: MediaPipe Hand Tracking model and segmentation algorithms.
- **Local Correllation Search**: Event-based Local Correlation Search algorithm using binary cross-correlation.
- **Kalman Filter**: Velocity/Acceleration/Jerk model Kalman filters.

## Installation

> Note: this project uses Python 3.11.4

1. Clone the repository:
```bash
git clone https://github.com/jet-tong/model-informed-event-based-visual-tracking.git
cd model-informed-event-based-visual-tracking
```

2. Install the required packages (numpy, opencv, mediapipe)
```bash
cd event_based_finger_tracking
pip install -r requirements.txt
```

3. Run the main script:
```bash
python main.py
```