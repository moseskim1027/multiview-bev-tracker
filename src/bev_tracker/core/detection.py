from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """A single person/object detection from one camera frame.

    Attributes:
        cam_id: Originating camera identifier.
        bbox: Bounding box as [x1, y1, x2, y2] in image pixels.
        confidence: Detector confidence score in [0, 1].
        world_xy: (x, y) position on the shared world ground plane (metres).
            Populated by the projection step; (nan, nan) until then.
        embedding: Normalised L2 ReID feature vector (e.g. 512-D).
            Populated by the ReID step; empty array until then.
    """

    cam_id: str
    bbox: np.ndarray  # shape (4,) — [x1, y1, x2, y2]
    confidence: float
    world_xy: tuple[float, float] = field(default=(float("nan"), float("nan")))
    embedding: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float32))

    def __post_init__(self) -> None:
        self.bbox = np.asarray(self.bbox, dtype=np.float32)
        if self.bbox.shape != (4,):
            raise ValueError(f"bbox must be shape (4,), got {self.bbox.shape}")

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Pixel coordinate at the foot of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return float((x1 + x2) / 2), float(y2)

    @property
    def has_world_position(self) -> bool:
        return not (np.isnan(self.world_xy[0]) or np.isnan(self.world_xy[1]))

    @property
    def has_embedding(self) -> bool:
        return self.embedding.size > 0
