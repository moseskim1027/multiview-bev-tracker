"""Unit tests for the Pipeline and BEVRenderer.

Heavy deps (cv2, torch, ultralytics) are mocked throughout.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bev_tracker.core.track import Track
from bev_tracker.pipeline import Pipeline, build_cameras, load_config
from bev_tracker.visualization.renderer import BEVRenderer

# ── Fixtures ──────────────────────────────────────────────────────────────────

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


@pytest.fixture()
def cfg() -> dict:
    return load_config(_CONFIG_PATH)


# ── load_config ───────────────────────────────────────────────────────────────


class TestLoadConfig:
    def test_returns_dict(self, cfg):
        assert isinstance(cfg, dict)

    def test_has_required_sections(self, cfg):
        for key in ("detection", "reid", "tracking", "cameras"):
            assert key in cfg

    def test_cameras_is_list(self, cfg):
        assert isinstance(cfg["cameras"], list)
        assert len(cfg["cameras"]) >= 1


# ── build_cameras ─────────────────────────────────────────────────────────────


class TestBuildCameras:
    def test_creates_correct_count(self, cfg):
        det = MagicMock()
        reid = MagicMock()
        cams = build_cameras(cfg, det, reid)
        assert len(cams) == len(cfg["cameras"])

    def test_homography_shape(self, cfg):
        cams = build_cameras(cfg, MagicMock(), MagicMock())
        for cam in cams:
            assert cam.homography.shape == (3, 3)

    def test_detector_assigned(self, cfg):
        det = MagicMock()
        cams = build_cameras(cfg, det, MagicMock())
        assert all(c.detector is det for c in cams)


# ── BEVRenderer ───────────────────────────────────────────────────────────────


class TestBEVRenderer:
    def _renderer(self, **kwargs) -> BEVRenderer:
        defaults = dict(width=200, height=150, x_range=(-5.0, 5.0), y_range=(-5.0, 5.0))
        defaults.update(kwargs)
        return BEVRenderer(**defaults)

    def test_world_to_pixel_centre(self):
        r = self._renderer()
        u, v = r.world_to_pixel(0.0, 0.0)
        assert u == pytest.approx(100, abs=2)
        assert v == pytest.approx(75, abs=2)

    def test_world_to_pixel_top_left(self):
        r = self._renderer()
        u, v = r.world_to_pixel(-5.0, 5.0)
        assert u == 0
        assert v == 0

    def test_world_to_pixel_bottom_right(self):
        r = self._renderer()
        u, v = r.world_to_pixel(5.0, -5.0)
        assert u == 200
        assert v == 150

    def test_render_returns_correct_shape(self):
        r = self._renderer()
        mock_cv2 = MagicMock()
        mock_cv2.LINE_AA = 16
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        mock_cv2.circle = MagicMock()
        mock_cv2.putText = MagicMock()
        mock_cv2.line = MagicMock()
        with patch("bev_tracker.visualization.renderer.cv2", mock_cv2):
            canvas = r.render([])
        assert canvas.shape == (150, 200, 3)
        assert canvas.dtype == np.uint8

    def test_render_with_track(self):
        r = self._renderer()
        track = Track(id=0, world_xy=(1.0, 1.0), history=[(0.5, 0.5), (1.0, 1.0)])
        mock_cv2 = MagicMock()
        mock_cv2.LINE_AA = 16
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        with patch("bev_tracker.visualization.renderer.cv2", mock_cv2):
            r.render([track])
        # cv2.circle should have been called (at least position dot)
        assert mock_cv2.circle.called

    def test_render_no_trails_skips_line(self):
        r = self._renderer(show_trails=False)
        track = Track(id=0, world_xy=(0.0, 0.0), history=[(0.0, 0.0), (1.0, 0.0)])
        mock_cv2 = MagicMock()
        mock_cv2.LINE_AA = 16
        mock_cv2.FONT_HERSHEY_SIMPLEX = 0
        with patch("bev_tracker.visualization.renderer.cv2", mock_cv2):
            r.render([track])
        mock_cv2.line.assert_not_called()


# ── Pipeline.step ─────────────────────────────────────────────────────────────


class TestPipelineStep:
    def _make_pipeline(self) -> Pipeline:
        """Build a Pipeline with all heavy components mocked."""
        p = Pipeline.__new__(Pipeline)
        from bev_tracker.core.camera import Camera
        from bev_tracker.tracking.tracker import Tracker

        cam = Camera(id="cam0", homography=np.eye(3), source="")
        cam.detector = MagicMock()
        cam.reid_model = MagicMock()

        p.cameras = [cam]
        p.detector = MagicMock()
        p.detector.detect.return_value = []
        p.reid = MagicMock()
        p.tracker = Tracker()
        p.renderer = None
        return p

    def test_step_returns_tracks_key(self):
        p = self._make_pipeline()
        result = p.step({"cam0": np.zeros((100, 100, 3), dtype=np.uint8)})
        assert "tracks" in result

    def test_step_returns_bev_none_when_renderer_disabled(self):
        p = self._make_pipeline()
        result = p.step({"cam0": np.zeros((100, 100, 3), dtype=np.uint8)})
        assert result["bev"] is None

    def test_step_skips_missing_camera_frame(self):
        p = self._make_pipeline()
        # No frame provided — should not raise
        result = p.step({})
        assert result["tracks"] == []

    def test_step_accumulates_tracks_across_calls(self):
        from bev_tracker.core.detection import Detection

        p = self._make_pipeline()
        det = Detection(cam_id="cam0", bbox=[0, 0, 10, 20], confidence=0.9, world_xy=(1.0, 1.0))
        p.detector.detect.return_value = [det]
        p.reid.extract_batch = MagicMock()

        frame = {"cam0": np.zeros((100, 100, 3), dtype=np.uint8)}
        p.step(frame)
        result = p.step(frame)
        assert len(result["tracks"]) >= 1
