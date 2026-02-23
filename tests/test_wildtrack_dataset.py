"""Unit tests for WildtrackDataset calibration and annotation parsing.

These tests do NOT require torch, ultralytics, or torchreid.
They require the Wildtrack data to be present at data/wildtrack/.
Tests are skipped automatically if the data is missing.
"""

from pathlib import Path

import numpy as np
import pytest

from bev_tracker.datasets.wildtrack import (
    CAMERA_NAMES,
    GRID_STEP_M,
    GRID_X_CELLS,
    ORIGIN_X_M,
    ORIGIN_Y_M,
    WildtrackDataset,
    compute_homography,
    load_extrinsic,
    load_intrinsic,
    position_id_to_world,
)

WILDTRACK_ROOT = Path("data/wildtrack")
CALIB_AVAILABLE = (WILDTRACK_ROOT / "calibrations").exists()

skip_no_data = pytest.mark.skipif(
    not CALIB_AVAILABLE,
    reason="Wildtrack calibration data not found at data/wildtrack/",
)


# ── position_id_to_world ──────────────────────────────────────────────────────


class TestPositionIdToWorld:
    def test_origin_cell(self):
        x, y = position_id_to_world(0)
        assert x == pytest.approx(ORIGIN_X_M)
        assert y == pytest.approx(ORIGIN_Y_M)

    def test_x_increments_within_row(self):
        x0, _ = position_id_to_world(0)
        x1, _ = position_id_to_world(1)
        assert x1 - x0 == pytest.approx(GRID_STEP_M)

    def test_y_increments_across_rows(self):
        _, y0 = position_id_to_world(0)
        _, y1 = position_id_to_world(GRID_X_CELLS)
        assert y1 - y0 == pytest.approx(GRID_STEP_M)

    def test_known_value(self):
        """positionID 456826 from annotation frame 0."""
        x, y = position_id_to_world(456826)
        # x = -3.0 + 0.025 * (456826 % 480)
        expected_x = ORIGIN_X_M + GRID_STEP_M * (456826 % GRID_X_CELLS)
        expected_y = ORIGIN_Y_M + GRID_STEP_M * (456826 // GRID_X_CELLS)
        assert x == pytest.approx(expected_x)
        assert y == pytest.approx(expected_y)

    def test_result_type(self):
        x, y = position_id_to_world(100)
        assert isinstance(x, float)
        assert isinstance(y, float)


# ── load_intrinsic ────────────────────────────────────────────────────────────


@skip_no_data
class TestLoadIntrinsic:
    def test_shape(self):
        path = WILDTRACK_ROOT / "calibrations" / "intrinsic_zero" / "intr_CVLab1.xml"
        K = load_intrinsic(path)
        assert K.shape == (3, 3)

    def test_dtype(self):
        path = WILDTRACK_ROOT / "calibrations" / "intrinsic_zero" / "intr_CVLab1.xml"
        K = load_intrinsic(path)
        assert K.dtype == np.float64

    def test_bottom_row(self):
        """K must end with [0, 0, 1]."""
        path = WILDTRACK_ROOT / "calibrations" / "intrinsic_zero" / "intr_CVLab1.xml"
        K = load_intrinsic(path)
        np.testing.assert_array_equal(K[2], [0, 0, 1])

    def test_focal_lengths_positive(self):
        path = WILDTRACK_ROOT / "calibrations" / "intrinsic_zero" / "intr_CVLab1.xml"
        K = load_intrinsic(path)
        assert K[0, 0] > 0  # fx
        assert K[1, 1] > 0  # fy

    def test_all_cameras_loadable(self):
        for name in CAMERA_NAMES:
            path = WILDTRACK_ROOT / "calibrations" / "intrinsic_zero" / f"intr_{name}.xml"
            K = load_intrinsic(path)
            assert K.shape == (3, 3)


# ── load_extrinsic ────────────────────────────────────────────────────────────


@skip_no_data
class TestLoadExtrinsic:
    def test_shapes(self):
        path = WILDTRACK_ROOT / "calibrations" / "extrinsic" / "extr_CVLab1.xml"
        rvec, tvec = load_extrinsic(path)
        assert rvec.shape == (3,)
        assert tvec.shape == (3,)

    def test_dtype(self):
        path = WILDTRACK_ROOT / "calibrations" / "extrinsic" / "extr_CVLab1.xml"
        rvec, tvec = load_extrinsic(path)
        assert rvec.dtype == np.float64
        assert tvec.dtype == np.float64

    def test_all_cameras_loadable(self):
        for name in CAMERA_NAMES:
            path = WILDTRACK_ROOT / "calibrations" / "extrinsic" / f"extr_{name}.xml"
            rvec, tvec = load_extrinsic(path)
            assert rvec.shape == (3,)
            assert tvec.shape == (3,)


# ── compute_homography ────────────────────────────────────────────────────────


@skip_no_data
class TestComputeHomography:
    def _load(self, cam_idx: int):
        name = CAMERA_NAMES[cam_idx]
        K = load_intrinsic(WILDTRACK_ROOT / "calibrations" / "intrinsic_zero" / f"intr_{name}.xml")
        rvec, tvec = load_extrinsic(
            WILDTRACK_ROOT / "calibrations" / "extrinsic" / f"extr_{name}.xml"
        )
        return K, rvec, tvec

    def test_shape(self):
        K, rvec, tvec = self._load(0)
        H = compute_homography(K, rvec, tvec)
        assert H.shape == (3, 3)

    def test_invertible(self):
        K, rvec, tvec = self._load(0)
        H = compute_homography(K, rvec, tvec)
        assert abs(np.linalg.det(H)) > 1e-6

    def test_all_cameras_produce_valid_H(self):
        for i in range(len(CAMERA_NAMES)):
            K, rvec, tvec = self._load(i)
            H = compute_homography(K, rvec, tvec)
            assert H.shape == (3, 3)
            assert abs(np.linalg.det(H)) > 1e-6


# ── WildtrackDataset ──────────────────────────────────────────────────────────


@skip_no_data
class TestWildtrackDataset:
    @pytest.fixture()
    def ds(self) -> WildtrackDataset:
        return WildtrackDataset(WILDTRACK_ROOT)

    def test_num_cameras(self, ds):
        assert ds.num_cameras == 7

    def test_num_frames(self, ds):
        assert ds.num_frames == 400

    def test_homography_shape(self, ds):
        H = ds.homography(0)
        assert H.shape == (3, 3)

    def test_homography_cached(self, ds):
        H1 = ds.homography(0)
        H2 = ds.homography(0)
        assert H1 is H2  # same object — cached

    def test_load_annotations_returns_list(self, ds):
        anns = ds.load_annotations(0)
        assert isinstance(anns, list)
        assert len(anns) > 0

    def test_annotation_structure(self, ds):
        ann = ds.load_annotations(0)[0]
        assert "personID" in ann
        assert "world_xy" in ann
        assert "views" in ann
        assert len(ann["views"]) == 7

    def test_world_xy_in_metres(self, ds):
        """All world positions must fall within the known Wildtrack grid bounds."""
        anns = ds.load_annotations(0)
        for ann in anns:
            x, y = ann["world_xy"]
            assert -3.5 <= x <= 9.5, f"x={x} out of range"
            assert -9.5 <= y <= 27.5, f"y={y} out of range"
