"""Unit tests for the Detector wrapper.

Ultralytics is not installed in the test environment, so the model loading
path is mocked.  The tests focus on the Detection objects produced and the
filtering / class-selection logic.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from bev_tracker.core.detection import Detection
from bev_tracker.detection.detector import Detector


def _make_result(bboxes: list[list[float]], confs: list[float]) -> MagicMock:
    """Build a fake Ultralytics result object."""
    result = MagicMock()
    boxes = MagicMock()
    boxes.xyxy.cpu().numpy.return_value = np.array(bboxes, dtype=np.float32).reshape(-1, 4)
    boxes.conf.cpu().numpy.return_value = np.array(confs, dtype=np.float32)
    result.boxes = boxes
    return result


def _inject_model(detector: Detector, bboxes, confs) -> Detector:
    """Bypass lazy loading by injecting a pre-built mock directly."""
    fake_results = [_make_result(bboxes, confs)]
    detector._model = MagicMock(return_value=fake_results)
    return detector


@pytest.fixture()
def detector() -> Detector:
    return Detector(model_path="yolov8n.pt", confidence_threshold=0.5, device="cpu")


class TestDetectorInit:
    def test_defaults(self, detector: Detector):
        assert detector.confidence_threshold == 0.5
        assert detector.device == "cpu"
        assert detector.target_classes == {0}

    def test_model_not_loaded_at_init(self, detector: Detector):
        assert detector._model is None

    def test_custom_target_classes(self):
        d = Detector(target_classes={0, 2})
        assert d.target_classes == {0, 2}


class TestDetectorDetect:
    def _run(self, detector: Detector, bboxes, confs) -> list[Detection]:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _inject_model(detector, bboxes, confs)
        return detector.detect(frame, cam_id="cam0")

    def test_returns_list_of_detections(self, detector: Detector):
        dets = self._run(detector, [[10, 20, 50, 80]], [0.9])
        assert len(dets) == 1
        assert isinstance(dets[0], Detection)

    def test_bbox_shape(self, detector: Detector):
        dets = self._run(detector, [[10, 20, 50, 80]], [0.9])
        assert dets[0].bbox.shape == (4,)

    def test_cam_id_tagged(self, detector: Detector):
        dets = self._run(detector, [[10, 20, 50, 80]], [0.9])
        assert dets[0].cam_id == "cam0"

    def test_confidence_stored(self, detector: Detector):
        dets = self._run(detector, [[10, 20, 50, 80]], [0.75])
        assert dets[0].confidence == pytest.approx(0.75)

    def test_world_xy_unset(self, detector: Detector):
        dets = self._run(detector, [[10, 20, 50, 80]], [0.9])
        assert not dets[0].has_world_position

    def test_embedding_unset(self, detector: Detector):
        dets = self._run(detector, [[10, 20, 50, 80]], [0.9])
        assert not dets[0].has_embedding

    def test_multiple_detections(self, detector: Detector):
        dets = self._run(
            detector,
            [[0, 0, 10, 20], [100, 100, 200, 300]],
            [0.8, 0.6],
        )
        assert len(dets) == 2

    def test_empty_frame_returns_empty_list(self, detector: Detector):
        dets = self._run(detector, [], [])
        assert dets == []

    def test_missing_ultralytics_raises_import_error(self):
        d = Detector()
        orig = sys.modules.get("ultralytics", None)
        sys.modules["ultralytics"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="ultralytics"):
                d._load()
        finally:
            if orig is None:
                sys.modules.pop("ultralytics", None)
            else:
                sys.modules["ultralytics"] = orig
