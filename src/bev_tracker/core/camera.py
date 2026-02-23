from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Camera:
    """Represents one physical camera in the multi-camera rig.

    Attributes:
        id: Unique string identifier (e.g. "cam0").
        homography: 3×3 matrix mapping the ground plane to image coordinates.
            Used as H in:  p_world = H_inv @ [u, v, 1]^T
        source: Video file path or device index used to open a capture stream.
        detector: Loaded YOLOv8 model instance (set after initialisation).
        reid_model: Loaded OSNet model instance (set after initialisation).
    """

    id: str
    homography: np.ndarray  # shape (3, 3), dtype float64
    source: str | int = ""
    detector: Any = field(default=None, repr=False)
    reid_model: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.homography = np.asarray(self.homography, dtype=np.float64)
        if self.homography.shape != (3, 3):
            raise ValueError(f"homography must be shape (3, 3), got {self.homography.shape}")

    @property
    def homography_inv(self) -> np.ndarray:
        """Inverse homography: image pixel → world coordinate."""
        return np.linalg.inv(self.homography)
