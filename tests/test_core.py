"""Unit tests for core dataclasses."""

import numpy as np
import pytest

from bev_tracker.core.camera import Camera
from bev_tracker.core.detection import Detection
from bev_tracker.core.track import Track


# ── Camera ────────────────────────────────────────────────────────────────────

class TestCamera:
    def _identity_cam(self, cam_id: str = "cam0") -> Camera:
        return Camera(id=cam_id, homography=np.eye(3))

    def test_homography_stored_as_float64(self):
        cam = self._identity_cam()
        assert cam.homography.dtype == np.float64

    def test_homography_shape_enforced(self):
        with pytest.raises(ValueError):
            Camera(id="bad", homography=np.eye(4))

    def test_homography_inv_is_inverse(self):
        H = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1.0]])
        cam = Camera(id="cam0", homography=H)
        product = cam.homography @ cam.homography_inv
        np.testing.assert_allclose(product, np.eye(3), atol=1e-10)

    def test_list_homography_coerced(self):
        cam = Camera(id="cam0", homography=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert isinstance(cam.homography, np.ndarray)
        assert cam.homography.shape == (3, 3)

    def test_detector_and_reid_default_none(self):
        cam = self._identity_cam()
        assert cam.detector is None
        assert cam.reid_model is None


# ── Detection ─────────────────────────────────────────────────────────────────

class TestDetection:
    def _det(self, **kwargs) -> Detection:
        defaults = dict(
            cam_id="cam0",
            bbox=[10.0, 20.0, 50.0, 80.0],
            confidence=0.9,
        )
        defaults.update(kwargs)
        return Detection(**defaults)

    def test_bbox_stored_as_float32(self):
        det = self._det()
        assert det.bbox.dtype == np.float32

    def test_bbox_shape_enforced(self):
        with pytest.raises(ValueError):
            Detection(cam_id="cam0", bbox=[1, 2, 3], confidence=0.8)

    def test_bottom_center(self):
        det = self._det(bbox=[10.0, 20.0, 50.0, 80.0])
        u, v = det.bottom_center
        assert u == pytest.approx(30.0)
        assert v == pytest.approx(80.0)

    def test_world_xy_defaults_to_nan(self):
        det = self._det()
        assert not det.has_world_position

    def test_has_world_position_after_set(self):
        det = self._det()
        det.world_xy = (1.5, 3.0)
        assert det.has_world_position

    def test_embedding_defaults_empty(self):
        det = self._det()
        assert not det.has_embedding

    def test_has_embedding_after_set(self):
        det = self._det()
        det.embedding = np.ones(512, dtype=np.float32)
        assert det.has_embedding


# ── Track ─────────────────────────────────────────────────────────────────────

class TestTrack:
    def _track(self, **kwargs) -> Track:
        defaults = dict(id=0, world_xy=(1.0, 2.0))
        defaults.update(kwargs)
        return Track(**defaults)

    def test_default_age_zero(self):
        t = self._track()
        assert t.age == 0

    def test_history_appended(self):
        t = self._track()
        t.update_history()
        assert t.history == [(1.0, 2.0)]

    def test_history_capped(self):
        t = self._track()
        for i in range(40):
            t.world_xy = (float(i), 0.0)
            t.update_history(max_length=10)
        assert len(t.history) == 10

    def test_is_confirmed_false_initially(self):
        t = self._track()
        assert not t.is_confirmed

    def test_is_confirmed_true_after_last_seen_set(self):
        t = self._track(last_seen=5)
        assert t.is_confirmed
