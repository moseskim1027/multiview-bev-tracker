from __future__ import annotations

import numpy as np

from bev_tracker.core.camera import Camera
from bev_tracker.core.detection import Detection


def project_to_world(detection: Detection, camera: Camera) -> None:
    """Back-project a detection's foot-point to world coordinates in-place.

    Uses the bottom-center of the bounding box as the ground contact point,
    then applies the inverse homography to map from image pixels to the shared
    world coordinate frame (metres).

    The homography relationship is::

        [u, v, 1]^T  ~  H @ [x_world, y_world, 1]^T

    so the inverse gives::

        [x_world, y_world, w]^T  =  H_inv @ [u, v, 1]^T
        (x_world, y_world)       =  (x_world / w, y_world / w)

    Args:
        detection: Detection whose ``world_xy`` field will be updated.
        camera: Camera whose ``homography_inv`` matrix is used.
    """
    u, v = detection.bottom_center
    point_img = np.array([u, v, 1.0], dtype=np.float64)
    point_world = camera.homography_inv @ point_img
    point_world /= point_world[2]  # normalise homogeneous coordinate
    detection.world_xy = (float(point_world[0]), float(point_world[1]))


def project_detections(detections: list[Detection], camera: Camera) -> None:
    """Project all detections from one camera to world coordinates in-place.

    Convenience wrapper that calls :func:`project_to_world` for each detection.

    Args:
        detections: List of detections from ``camera``.
        camera: The originating camera with a calibrated homography.
    """
    for det in detections:
        project_to_world(det, camera)
