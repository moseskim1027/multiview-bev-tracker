from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Track:
    """A confirmed object track in the shared world coordinate frame.

    Attributes:
        id: Globally unique integer identifier.
        world_xy: Current (x, y) position estimate (metres).
        velocity: Current (vx, vy) velocity estimate (metres/frame).
        embedding: EMA-smoothed ReID feature vector.
        kalman_filter: filterpy KalmanFilter instance managing state.
        last_seen: Frame index of the most recent matched detection.
        age: Frames elapsed since the last matched detection.
            Incremented each frame; reset to 0 on match.
        history: List of past world_xy positions (for trail rendering).
    """

    id: int
    world_xy: tuple[float, float] = (0.0, 0.0)
    velocity: tuple[float, float] = (0.0, 0.0)
    embedding: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))
    kalman_filter: Any = field(default=None, repr=False)
    last_seen: int = 0
    age: int = 0
    history: list[tuple[float, float]] = field(default_factory=list)

    def update_history(self, max_length: int = 30) -> None:
        """Append current position to history, capping at max_length."""
        self.history.append(self.world_xy)
        if len(self.history) > max_length:
            self.history = self.history[-max_length:]

    @property
    def is_confirmed(self) -> bool:
        """True once the track has been matched at least once after creation."""
        return self.last_seen > 0
