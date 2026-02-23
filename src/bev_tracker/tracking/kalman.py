from __future__ import annotations

import numpy as np

# State vector: [x, y, vx, vy]  (position + velocity in world coords)
# Observation:  [x, y]          (measured world position)
_DIM_X = 4
_DIM_Z = 2


def _transition_matrix(dt: float = 1.0) -> np.ndarray:
    """Constant-velocity state-transition matrix F."""
    F = np.eye(_DIM_X, dtype=np.float64)
    F[0, 2] = dt
    F[1, 3] = dt
    return F


def _observation_matrix() -> np.ndarray:
    """Observation matrix H: extracts (x, y) from state."""
    H = np.zeros((_DIM_Z, _DIM_X), dtype=np.float64)
    H[0, 0] = 1.0
    H[1, 1] = 1.0
    return H


class KalmanFilter:
    """Minimal constant-velocity Kalman filter for 2-D world-space tracking.

    State:  [x, y, vx, vy]
    Obs:    [x, y]

    This avoids the filterpy dependency so the tracking module has no
    heavy runtime requirements beyond numpy.

    Args:
        initial_xy: Starting (x, y) world position (metres).
        dt: Time step between frames (default 1 — frame-rate agnostic).
        process_noise: Scalar variance added to diagonal of Q each step.
        measurement_noise: Scalar variance for observation covariance R.
    """

    def __init__(
        self,
        initial_xy: tuple[float, float],
        dt: float = 1.0,
        process_noise: float = 0.1,
        measurement_noise: float = 1.0,
    ) -> None:
        x0, y0 = initial_xy

        self.F = _transition_matrix(dt)
        self.H = _observation_matrix()
        self.Q = np.eye(_DIM_X, dtype=np.float64) * process_noise
        self.R = np.eye(_DIM_Z, dtype=np.float64) * measurement_noise
        self.P = np.eye(_DIM_X, dtype=np.float64) * 10.0  # initial uncertainty

        self.x = np.array([x0, y0, 0.0, 0.0], dtype=np.float64)

    # ── Core predict / update ─────────────────────────────────────────────────

    def predict(self) -> None:
        """Advance the state estimate by one time step."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, measurement: tuple[float, float]) -> None:
        """Correct the state estimate with a new observation."""
        z = np.array(measurement, dtype=np.float64)
        y = z - self.H @ self.x  # innovation
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        self.x = self.x + K @ y
        self.P = (np.eye(_DIM_X) - K @ self.H) @ self.P

    # ── Accessors ─────────────────────────────────────────────────────────────

    @property
    def position(self) -> tuple[float, float]:
        """Current (x, y) position estimate."""
        return float(self.x[0]), float(self.x[1])

    @property
    def velocity(self) -> tuple[float, float]:
        """Current (vx, vy) velocity estimate."""
        return float(self.x[2]), float(self.x[3])
