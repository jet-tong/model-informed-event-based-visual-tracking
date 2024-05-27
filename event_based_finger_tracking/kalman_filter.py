import numpy as np


class KalmanFilter:
    def __init__(self, dt, model="velocity"):
        if model == "velocity":
            self.init_velocity_model(dt)
        elif model == "acceleration":
            self.init_acceleration_model(dt)
        elif model == "jerk":
            self.init_jerk_model(dt)
        else:
            raise ValueError("Invalid model type")

    def init_velocity_model(self, dt):
        # * Velocity model
        self.state_dim = 4
        self.meas_dim = 2
        # fmt: off
        # State transition matrix: constant velocity model
        self.A = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        # Measurement matrix (state -> measurement space)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        # State covariance matrix (uncertainty in the state) (Initial)
        # self.P = np.eye(state_dim) * 10
        self.P = np.diag([1, 1, 10, 10]) * 10
        # Process noise covariance matrix (uncertainty in the model)
        # self.Q = np.eye(state_dim) * 10
        self.Q = np.diag([1, 1, 10, 10]) * 10
        # Measurement noise covariance matrix (uncertainty in the measurement)
        self.R = np.eye(self.meas_dim) * 1000
        self.R_mp = np.eye(self.meas_dim) * 100 # Trust MediaPipe more
        self.R_ev = np.eye(self.meas_dim) * 1000 # Trust Event-based less
        # Initial state vector (px, py, vx, vy)
        self.x = np.zeros((self.state_dim, 1))
        # fmt: on

    def init_acceleration_model(self, dt):
        # * Acceleration model
        self.state_dim = 6
        self.meas_dim = 2
        # fmt: off
        # State transition matrix: constant acceleration model
        self.A = np.array([[1, 0, dt, 0, dt**2 / 2, 0],
                            [0, 1, 0, dt, 0, dt**2 / 2],
                            [0, 0, 1, 0, dt, 0],
                            [0, 0, 0, 1, 0, dt],
                            [0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 1]])
        # self.A = np.array([[1, 0, dt, 0, 0, 0],
        #                     [0, 1, 0, dt, 0, 0],
        #                     [0, 0, 1, 0, dt, 0],
        #                     [0, 0, 0, 1, 0, dt],
        #                     [0, 0, 0, 0, 1, 0],
        #                     [0, 0, 0, 0, 0, 1]]) # Without O(dt^2) terms
        # Measurement matrix (state -> measurement space)
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0]])
        # State covariance matrix (uncertainty in the state)
        # self.P = np.eye(state_dim) * 10
        self.P = np.diag([1, 1, 10, 10, 100, 100]) * 10
        # Process noise covariance matrix (uncertainty in the model)
        # self.Q = np.eye(state_dim) * 10
        self.Q = np.diag([1, 1, 10, 10, 100, 100]) * 10
        # Measurement noise covariance matrix (uncertainty in the measurement)
        self.R = np.eye(self.meas_dim) * 10
        # Initial state vector (px, py, vx, vy, ax, ay)
        self.x = np.zeros((self.state_dim, 1))
        # fmt: on

    def init_jerk_model(self, dt):
        # * Jerk model
        self.state_dim = 8
        self.meas_dim = 2
        # fmt: off
        # State transition matrix: constant jerk model
        self.A = np.array([[1, 0, dt, 0, dt**2 / 2, 0, dt**3 / 6, 0],
                          [0, 1, 0, dt, 0, dt**2 / 2, 0, dt**3 / 6],
                          [0, 0, 1, 0, dt, 0, dt**2 / 2, 0],
                          [0, 0, 0, 1, 0, dt, 0, dt**2 / 2],
                          [0, 0, 0, 0, 1, 0, dt, 0],
                          [0, 0, 0, 0, 0, 1, 0, dt],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 0, 0, 1]])
        # self.A = np.array([[1, 0, dt, 0, 0, 0, 0, 0],
        #                   [0, 1, 0, dt, 0, 0, 0, 0],
        #                   [0, 0, 1, 0, dt, 0, 0, 0],
        #                   [0, 0, 0, 1, 0, dt, 0, 0],
        #                   [0, 0, 0, 0, 1, 0, dt, 0],
        #                   [0, 0, 0, 0, 0, 1, 0, dt],
        #                   [0, 0, 0, 0, 0, 0, 1, 0],
        #                   [0, 0, 0, 0, 0, 0, 0, 1]]) # Without O(dt^2) terms
        # Measurement matrix (state -> measurement space)
        self.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0, 0, 0]])
        # State covariance matrix (uncertainty in the state)
        # self.P = np.eye(state_dim) * 10
        self.P = np.diag([1, 1, 10, 10, 100, 100, 1000, 1000]) * 10
        # Process noise covariance matrix (uncertainty in the model)
        # self.Q = np.eye(state_dim) * 10
        self.Q = np.diag([1, 1, 10, 10, 100, 100, 1000, 1000]) * 10
        # Measurement noise covariance matrix (uncertainty in the measurement)
        self.R = np.eye(self.meas_dim) * 10
        # Initial state vector (px, py, vx, vy, ax, ay, jx, jy)
        self.x = np.zeros((self.state_dim, 1))
        # fmt: on

    def predict(self):
        # Predict state: x_{k|k-1} = A x_{k-1|k-1}
        self.x = np.dot(self.A, self.x)
        # Predict covariance: P_{k|k-1} = A P_{k-1|k-1} A^T + Q
        self.P = np.dot(self.A, np.dot(self.P, self.A.T)) + self.Q

    def correct(self, z, measurement_type=None):

        if measurement_type == "ev":
            R = self.R_ev
        elif measurement_type == "mp":
            R = self.R_mp
        else:
            R = self.R

        # Reshape measurement to column vector
        z = np.array(z).reshape(-1, 1)

        # Compute optimal Kalman Gain K (not sure how it works)
        # S = H P H^T + R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + R
        # K = P H^T S^{-1}
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))

        # Correct state: x_{k|k} = x_{k|k-1} + K (z_k - H x_{k|k-1})
        y = z - np.dot(self.H, self.x)
        self.x += np.dot(K, y)
        # Correct covariance: P_{k|k} = (I - K H) P_{k|k-1}
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

    def get_state(self):
        return self.x[0], self.x[1]

    def get_integer_state(self):
        return int(self.x[0]), int(self.x[1])
