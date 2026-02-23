"""Integration tests: real YOLOv8n on real WILDTRACK frames.

Requires:
    - models/yolov8n.pt
    - data/wildtrack/
"""

from pathlib import Path

import numpy as np
import pytest

MODELS_DIR = Path("models")
WILDTRACK_ROOT = Path("data/wildtrack")
YOLO_WEIGHTS = MODELS_DIR / "yolov8n.pt"

_skip = pytest.mark.skipif(
    not YOLO_WEIGHTS.exists() or not WILDTRACK_ROOT.exists(),
    reason="models/yolov8n.pt or data/wildtrack/ not found",
)


@pytest.fixture(scope="module")
def detector():
    from bev_tracker.detection.detector import Detector

    return Detector(model_path=str(YOLO_WEIGHTS), confidence_threshold=0.3, device="cpu")


@pytest.fixture(scope="module")
def wildtrack_frame_c1():
    from bev_tracker.datasets.wildtrack import WildtrackDataset

    ds = WildtrackDataset(WILDTRACK_ROOT)
    return ds.load_frame(cam_idx=0, frame_idx=0)  # C1, first annotated frame


@pytest.mark.integration
@_skip
class TestDetectorIntegration:
    def test_returns_detections(self, detector, wildtrack_frame_c1):
        """Frame 0 / C1 has visible pedestrians — must detect at least one."""
        dets = detector.detect(wildtrack_frame_c1, cam_id="C1")
        assert len(dets) >= 1

    def test_detection_cam_id(self, detector, wildtrack_frame_c1):
        dets = detector.detect(wildtrack_frame_c1, cam_id="C1")
        for det in dets:
            assert det.cam_id == "C1"

    def test_bbox_shape_and_dtype(self, detector, wildtrack_frame_c1):
        dets = detector.detect(wildtrack_frame_c1, cam_id="C1")
        for det in dets:
            assert det.bbox.shape == (4,)
            assert det.bbox.dtype == np.float32

    def test_bbox_valid_coords(self, detector, wildtrack_frame_c1):
        """Each bbox must satisfy x1 < x2 and y1 < y2."""
        dets = detector.detect(wildtrack_frame_c1, cam_id="C1")
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            assert x1 < x2, f"x1={x1} >= x2={x2}"
            assert y1 < y2, f"y1={y1} >= y2={y2}"

    def test_bbox_within_image(self, detector, wildtrack_frame_c1):
        h, w = wildtrack_frame_c1.shape[:2]
        dets = detector.detect(wildtrack_frame_c1, cam_id="C1")
        for det in dets:
            x1, y1, x2, y2 = det.bbox
            assert 0 <= x1 and x2 <= w, f"x range [{x1},{x2}] outside [0,{w}]"
            assert 0 <= y1 and y2 <= h, f"y range [{y1},{y2}] outside [0,{h}]"

    def test_confidence_above_threshold(self, detector, wildtrack_frame_c1):
        dets = detector.detect(wildtrack_frame_c1, cam_id="C1")
        for det in dets:
            assert det.confidence >= 0.3

    def test_world_xy_starts_nan(self, detector, wildtrack_frame_c1):
        """Detector must not populate world_xy — that is the projector's job."""
        dets = detector.detect(wildtrack_frame_c1, cam_id="C1")
        for det in dets:
            assert np.isnan(det.world_xy[0])
            assert np.isnan(det.world_xy[1])

    def test_bottom_center_within_image(self, detector, wildtrack_frame_c1):
        h, w = wildtrack_frame_c1.shape[:2]
        dets = detector.detect(wildtrack_frame_c1, cam_id="C1")
        for det in dets:
            u, v = det.bottom_center
            assert 0 <= u <= w, f"foot x={u} outside [0,{w}]"
            assert 0 <= v <= h, f"foot y={v} outside [0,{h}]"
