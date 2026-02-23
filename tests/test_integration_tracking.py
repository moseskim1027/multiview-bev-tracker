"""Integration tests: full detect → project → reid → track pipeline on WILDTRACK.

Requires:
    - models/yolov8n.pt
    - models/osnet_x0_25_market.pth
    - data/wildtrack/
"""

from pathlib import Path

import numpy as np
import pytest

MODELS_DIR = Path("models")
WILDTRACK_ROOT = Path("data/wildtrack")
YOLO_WEIGHTS = MODELS_DIR / "yolov8n.pt"
OSNET_WEIGHTS = MODELS_DIR / "osnet_x0_25_market.pth"

_WORLD_X_RANGE = (-3.5, 9.5)
_WORLD_Y_RANGE = (-9.5, 27.5)

_skip = pytest.mark.skipif(
    not YOLO_WEIGHTS.exists() or not OSNET_WEIGHTS.exists() or not WILDTRACK_ROOT.exists(),
    reason="Models or data/wildtrack/ not found",
)


# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def detector():
    from bev_tracker.detection.detector import Detector

    return Detector(model_path=str(YOLO_WEIGHTS), confidence_threshold=0.3, device="cpu")


@pytest.fixture(scope="module")
def extractor():
    from bev_tracker.reid.extractor import ReIDExtractor

    return ReIDExtractor(
        model_name="osnet_x0_25",
        weights_path=str(OSNET_WEIGHTS),
        device="cpu",
    )


@pytest.fixture(scope="module")
def dataset():
    from bev_tracker.datasets.wildtrack import WildtrackDataset

    return WildtrackDataset(WILDTRACK_ROOT)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_frame(frame_idx, dataset, detector, extractor):
    """Detect → project → reid for all cameras; return merged detections."""
    from bev_tracker.core.camera import Camera
    from bev_tracker.projection.homography import project_detections

    all_dets = []
    for cam_idx in range(dataset.num_cameras):
        frame = dataset.load_frame(cam_idx, frame_idx)
        cam_id = f"C{cam_idx + 1}"

        dets = detector.detect(frame, cam_id=cam_id)

        cam = Camera(id=cam_id, homography=dataset.homography(cam_idx))
        project_detections(dets, cam)
        extractor.extract_batch(dets, frame)

        all_dets.extend(dets)
    return all_dets


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@_skip
class TestProjectionWithRealCalibration:
    """Validate that detected foot-points project into the Wildtrack world grid."""

    def test_projected_world_xy_finite(self, detector, dataset):
        """All projected detections from C1 must yield finite world coordinates."""
        from bev_tracker.core.camera import Camera
        from bev_tracker.projection.homography import project_detections

        frame = dataset.load_frame(cam_idx=0, frame_idx=0)
        dets = detector.detect(frame, cam_id="C1")
        cam = Camera(id="C1", homography=dataset.homography(0))
        project_detections(dets, cam)

        for det in dets:
            x, y = det.world_xy
            assert np.isfinite(x), f"x={x} is not finite"
            assert np.isfinite(y), f"y={y} is not finite"

    def test_gt_world_roundtrip(self, dataset):
        """Ground-truth world → image → world round-trip must be <5 cm error."""
        H = dataset.homography(0)
        H_inv = np.linalg.inv(H)

        anns = dataset.load_annotations(0)
        errors = []
        for ann in anns[:10]:
            xw, yw = ann["world_xy"]
            # Forward: world → image
            pt_w = np.array([xw, yw, 1.0])
            pt_img = H @ pt_w
            pt_img /= pt_img[2]
            # Inverse: image → world
            pt_back = H_inv @ pt_img
            pt_back /= pt_back[2]
            err = np.sqrt((pt_back[0] - xw) ** 2 + (pt_back[1] - yw) ** 2)
            errors.append(err)

        max_err = max(errors)
        assert max_err < 0.05, f"Max round-trip error = {max_err:.4f} m (threshold 0.05 m)"


@pytest.mark.integration
@_skip
class TestTrackerWithRealDetections:
    """Run the full pipeline over several Wildtrack frames and sanity-check tracks."""

    N_FRAMES = 5  # subset for speed

    def test_tracks_created(self, detector, extractor, dataset):
        from bev_tracker.tracking.tracker import Tracker

        tracker = Tracker(max_age=10, dist_threshold=3.0, cost_threshold=5.0)
        final_tracks = None
        for frame_idx in range(self.N_FRAMES):
            dets = _run_frame(frame_idx, dataset, detector, extractor)
            final_tracks = tracker.step(dets)

        assert final_tracks is not None
        assert len(final_tracks) >= 1, "Expected ≥1 active track after 5 frames"

    def test_track_ids_are_non_negative_integers(self, detector, extractor, dataset):
        from bev_tracker.tracking.tracker import Tracker

        tracker = Tracker(max_age=10, dist_threshold=3.0, cost_threshold=5.0)
        for frame_idx in range(self.N_FRAMES):
            dets = _run_frame(frame_idx, dataset, detector, extractor)
            tracks = tracker.step(dets)

        for t in tracks:
            assert isinstance(t.id, int)
            assert t.id >= 0

    def test_track_positions_finite(self, detector, extractor, dataset):
        """Track positions must be finite (no NaN or inf from Kalman extrapolation)."""
        from bev_tracker.tracking.tracker import Tracker

        tracker = Tracker(max_age=10, dist_threshold=3.0, cost_threshold=5.0)
        for frame_idx in range(self.N_FRAMES):
            dets = _run_frame(frame_idx, dataset, detector, extractor)
            tracks = tracker.step(dets)

        for t in tracks:
            x, y = t.world_xy
            assert np.isfinite(x), f"Track {t.id} x={x} is not finite"
            assert np.isfinite(y), f"Track {t.id} y={y} is not finite"

    def test_track_embeddings_normalised(self, detector, extractor, dataset):
        """Every active track with an embedding must have L2 norm ≈ 1."""
        from bev_tracker.tracking.tracker import Tracker

        tracker = Tracker(max_age=10, dist_threshold=3.0, cost_threshold=5.0)
        for frame_idx in range(self.N_FRAMES):
            dets = _run_frame(frame_idx, dataset, detector, extractor)
            tracks = tracker.step(dets)

        for t in tracks:
            if t.embedding.size > 0:
                norm = float(np.linalg.norm(t.embedding))
                assert abs(norm - 1.0) < 0.01, f"Track {t.id} embedding norm = {norm:.4f}"

    def test_persistent_tracks_gain_history(self, detector, extractor, dataset):
        """Tracks matched across multiple frames must accumulate position history."""
        from bev_tracker.tracking.tracker import Tracker

        tracker = Tracker(max_age=15, dist_threshold=3.0, cost_threshold=5.0)
        for frame_idx in range(self.N_FRAMES):
            dets = _run_frame(frame_idx, dataset, detector, extractor)
            tracks = tracker.step(dets)

        # At least one track should have been matched more than once
        long_tracks = [t for t in tracks if len(t.history) >= 2]
        assert len(long_tracks) >= 1, "Expected ≥1 track with history ≥ 2 positions"
