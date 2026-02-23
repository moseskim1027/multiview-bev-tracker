"""Integration tests: real OSNet-x0.25 on real WILDTRACK crops.

Requires:
    - models/osnet_x0_25_market.pth
    - data/wildtrack/
"""

from pathlib import Path

import numpy as np
import pytest

MODELS_DIR = Path("models")
WILDTRACK_ROOT = Path("data/wildtrack")
OSNET_WEIGHTS = MODELS_DIR / "osnet_x0_25_market.pth"

_skip = pytest.mark.skipif(
    not OSNET_WEIGHTS.exists() or not WILDTRACK_ROOT.exists(),
    reason="models/osnet_x0_25_market.pth or data/wildtrack/ not found",
)


@pytest.fixture(scope="module")
def extractor():
    from bev_tracker.reid.extractor import ReIDExtractor

    return ReIDExtractor(
        model_name="osnet_x0_25",
        weights_path=str(OSNET_WEIGHTS),
        device="cpu",
    )


@pytest.fixture(scope="module")
def ann0():
    """First annotation entry for frame 0 with at least one visible view."""
    import json

    with open(WILDTRACK_ROOT / "annotations_positions" / "00000000.json") as f:
        raw = json.load(f)
    # Pick the first person visible in C1 (viewNum 0, coords != -1)
    for ann in raw:
        v = ann["views"][0]
        if v["xmin"] != -1:
            return ann
    pytest.skip("No person visible in C1 in frame 0")


@pytest.fixture(scope="module")
def crop_c1(ann0):
    """BGR crop of ann0's bounding box in camera C1."""
    import cv2

    frame = cv2.imread(str(WILDTRACK_ROOT / "Image_subsets" / "C1" / "00000000.png"))
    v = ann0["views"][0]
    x1, y1, x2, y2 = v["xmin"], v["ymin"], v["xmax"], v["ymax"]
    x1, x2 = max(0, x1), min(frame.shape[1], x2)
    y1, y2 = max(0, y1), min(frame.shape[0], y2)
    return frame[y1:y2, x1:x2]


@pytest.mark.integration
@_skip
class TestReIDIntegration:
    def test_embedding_shape(self, extractor, crop_c1):
        emb = extractor.extract(crop_c1)
        assert emb.shape == (512,)

    def test_embedding_dtype(self, extractor, crop_c1):
        emb = extractor.extract(crop_c1)
        assert emb.dtype == np.float32

    def test_embedding_l2_normalised(self, extractor, crop_c1):
        emb = extractor.extract(crop_c1)
        norm = float(np.linalg.norm(emb))
        assert abs(norm - 1.0) < 1e-5, f"L2 norm = {norm}, expected ~1.0"

    def test_embedding_not_all_zeros(self, extractor, crop_c1):
        emb = extractor.extract(crop_c1)
        assert np.any(emb != 0.0)

    def test_empty_crop_returns_zeros(self, extractor):
        """Zero-size crop must return a zero vector without raising."""
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        emb = extractor.extract(empty)
        assert emb.shape == (512,)
        assert np.all(emb == 0.0)

    def test_different_persons_have_different_embeddings(self, extractor, ann0):
        """Two different persons in the same frame should not be identical."""
        import json

        import cv2

        frame = cv2.imread(str(WILDTRACK_ROOT / "Image_subsets" / "C1" / "00000000.png"))
        with open(WILDTRACK_ROOT / "annotations_positions" / "00000000.json") as f:
            raw = json.load(f)

        # Collect two distinct crops from C1
        crops = []
        for ann in raw:
            v = ann["views"][0]
            if v["xmin"] == -1:
                continue
            x1, y1, x2, y2 = v["xmin"], v["ymin"], v["xmax"], v["ymax"]
            x1, x2 = max(0, x1), min(frame.shape[1], x2)
            y1, y2 = max(0, y1), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(extractor.extract(crop))
            if len(crops) == 2:
                break

        if len(crops) < 2:
            pytest.skip("Fewer than 2 visible persons in C1 frame 0")

        cosine_sim = float(np.dot(crops[0], crops[1]))
        # Embeddings of different persons should not be identical
        assert cosine_sim < 0.999, f"Two distinct persons have cosine sim = {cosine_sim:.4f}"

    def test_same_crop_deterministic(self, extractor, crop_c1):
        """Two forward passes on the same crop must produce identical embeddings."""
        emb1 = extractor.extract(crop_c1)
        emb2 = extractor.extract(crop_c1)
        np.testing.assert_array_equal(emb1, emb2)
