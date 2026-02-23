"""Core dataclasses shared across the pipeline."""

from bev_tracker.core.camera import Camera
from bev_tracker.core.detection import Detection
from bev_tracker.core.track import Track

__all__ = ["Camera", "Detection", "Track"]
