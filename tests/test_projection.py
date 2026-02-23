"""Unit tests for the homography projection module."""

import numpy as np
import pytest

from bev_tracker.core.camera import Camera
from bev_tracker.core.detection import Detection
from bev_tracker.projection.homography import project_detections, project_to_world


def _det(x1=10.0, y1=20.0, x2=50.0, y2=80.0, cam_id="cam0") -> Detection:
    return Detection(cam_id=cam_id, bbox=[x1, y1, x2, y2], confidence=0.9)


def _cam(H: np.ndarray | None = None) -> Camera:
    if H is None:
        H = np.eye(3)
    return Camera(id="cam0", homography=H)


class TestProjectToWorld:
    def test_identity_homography(self):
        """With H=I, world coords equal bottom-center pixel coords."""
        det = _det(x1=10, y1=20, x2=50, y2=80)
        cam = _cam()
        project_to_world(det, cam)
        u, v = det.bottom_center  # (30, 80)
        assert det.world_xy == pytest.approx((u, v))

    def test_sets_has_world_position(self):
        det = _det()
        cam = _cam()
        assert not det.has_world_position
        project_to_world(det, cam)
        assert det.has_world_position

    def test_scale_homography(self):
        """H = scale * I maps (u,v) -> (u/scale, v/scale)."""
        scale = 2.0
        H = np.diag([scale, scale, 1.0])
        cam = _cam(H)
        det = _det(x1=0, y1=0, x2=40, y2=60)  # bottom-center = (20, 60)
        project_to_world(det, cam)
        assert det.world_xy == pytest.approx((20 / scale, 60 / scale))

    def test_translation_homography(self):
        """H with translation offset should shift world coords."""
        tx, ty = 5.0, -3.0
        H = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
        cam = _cam(H)
        # bottom-center = (30, 80)
        det = _det(x1=10, y1=20, x2=50, y2=80)
        project_to_world(det, cam)
        # p_world = H_inv @ [30, 80, 1]  with H_inv having tx=-5, ty=3
        assert det.world_xy == pytest.approx((30 - tx, 80 - ty))

    def test_result_is_tuple_of_floats(self):
        det = _det()
        project_to_world(det, _cam())
        x, y = det.world_xy
        assert isinstance(x, float)
        assert isinstance(y, float)

    def test_bottom_center_used(self):
        """Projection uses foot of bbox, not centre of box."""
        det = _det(x1=0, y1=0, x2=100, y2=200)
        cam = _cam()  # identity
        project_to_world(det, cam)
        # bottom-center should be (50, 200) not (50, 100)
        assert det.world_xy[1] == pytest.approx(200.0)


class TestProjectDetections:
    def test_all_detections_projected(self):
        dets = [_det(x1=i * 10, y1=0, x2=i * 10 + 5, y2=20) for i in range(4)]
        cam = _cam()
        project_detections(dets, cam)
        assert all(d.has_world_position for d in dets)

    def test_empty_list_is_noop(self):
        cam = _cam()
        project_detections([], cam)  # should not raise

    def test_modifies_detections_in_place(self):
        det = _det()
        original_id = id(det)
        cam = _cam()
        project_detections([det], cam)
        assert id(det) == original_id
        assert det.has_world_position
