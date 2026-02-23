"""Unit tests for the ReIDExtractor.

torchreid, torch, and cv2 are not installed in the test environment.
Tests mock _preprocess and _run_model so they exercise the public API
logic (L2 normalisation, in-place mutation, clamping) without heavy deps.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

from bev_tracker.core.detection import Detection
from bev_tracker.reid.extractor import ReIDExtractor

# ── Helpers ───────────────────────────────────────────────────────────────────

_FEAT_DIM = 512


def _raw_feat(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(_FEAT_DIM).astype(np.float32)


def _inject_mocks(extractor: ReIDExtractor, raw: np.ndarray | None = None) -> np.ndarray:
    """Replace _preprocess and _run_model with lightweight mocks.

    Returns the raw (un-normalised) vector that _run_model will return.
    """
    if raw is None:
        raw = _raw_feat()
    dummy_nchw = np.zeros((1, 3, 256, 128), dtype=np.float32)
    extractor._preprocess = MagicMock(return_value=dummy_nchw)
    extractor._run_model = MagicMock(return_value=raw.copy())
    return raw


def _make_frame(h: int = 100, w: int = 60) -> np.ndarray:
    return np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)


def _det(x1=5, y1=5, x2=55, y2=95) -> Detection:
    return Detection(cam_id="cam0", bbox=[x1, y1, x2, y2], confidence=0.9)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def extractor() -> ReIDExtractor:
    return ReIDExtractor(model_name="osnet_x0_25", weights_path="", device="cpu")


# ── Init ──────────────────────────────────────────────────────────────────────


class TestReIDExtractorInit:
    def test_model_not_loaded_at_init(self, extractor: ReIDExtractor):
        assert extractor._model is None

    def test_defaults(self, extractor: ReIDExtractor):
        assert extractor.model_name == "osnet_x0_25"
        assert extractor.device == "cpu"

    def test_missing_torchreid_raises(self):
        e = ReIDExtractor()
        orig = sys.modules.get("torchreid", None)
        sys.modules["torchreid"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="torchreid"):
                e._load()
        finally:
            if orig is None:
                sys.modules.pop("torchreid", None)
            else:
                sys.modules["torchreid"] = orig


# ── extract ───────────────────────────────────────────────────────────────────


class TestExtract:
    def test_output_shape(self, extractor: ReIDExtractor):
        _inject_mocks(extractor)
        emb = extractor.extract(_make_frame())
        assert emb.shape == (_FEAT_DIM,)

    def test_output_dtype(self, extractor: ReIDExtractor):
        _inject_mocks(extractor)
        emb = extractor.extract(_make_frame())
        assert emb.dtype == np.float32

    def test_l2_normalised(self, extractor: ReIDExtractor):
        _inject_mocks(extractor)
        emb = extractor.extract(_make_frame())
        assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)

    def test_empty_crop_returns_zero_vector(self, extractor: ReIDExtractor):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        emb = extractor.extract(empty)
        assert emb.shape == (_FEAT_DIM,)
        np.testing.assert_array_equal(emb, 0.0)

    def test_preprocess_called_once(self, extractor: ReIDExtractor):
        _inject_mocks(extractor)
        extractor.extract(_make_frame())
        extractor._preprocess.assert_called_once()

    def test_run_model_called_once(self, extractor: ReIDExtractor):
        _inject_mocks(extractor)
        extractor.extract(_make_frame())
        extractor._run_model.assert_called_once()


# ── extract_from_detection ────────────────────────────────────────────────────


class TestExtractFromDetection:
    def test_sets_embedding_in_place(self, extractor: ReIDExtractor):
        _inject_mocks(extractor)
        det = _det()
        assert not det.has_embedding
        extractor.extract_from_detection(det, _make_frame())
        assert det.has_embedding

    def test_embedding_is_normalised(self, extractor: ReIDExtractor):
        _inject_mocks(extractor)
        det = _det()
        extractor.extract_from_detection(det, _make_frame())
        assert np.linalg.norm(det.embedding) == pytest.approx(1.0, abs=1e-5)

    def test_bbox_clamped_to_frame(self, extractor: ReIDExtractor):
        """Out-of-bounds bbox coordinates are clamped, not raised."""
        _inject_mocks(extractor)
        det = Detection(cam_id="cam0", bbox=[-10, -10, 200, 200], confidence=0.9)
        extractor.extract_from_detection(det, _make_frame(h=50, w=50))
        assert det.has_embedding


# ── extract_batch ─────────────────────────────────────────────────────────────


class TestExtractBatch:
    def test_all_detections_get_embeddings(self, extractor: ReIDExtractor):
        _inject_mocks(extractor)
        dets = [_det() for _ in range(3)]
        extractor.extract_batch(dets, _make_frame())
        assert all(d.has_embedding for d in dets)

    def test_empty_list_is_noop(self, extractor: ReIDExtractor):
        extractor.extract_batch([], _make_frame())  # should not raise
