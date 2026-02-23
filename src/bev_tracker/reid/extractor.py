from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bev_tracker.core.detection import Detection

if TYPE_CHECKING:
    import numpy.typing as npt

# Input size expected by OSNet (torchreid default)
_OSNET_INPUT_H = 256
_OSNET_INPUT_W = 128

# ImageNet normalisation constants
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ReIDExtractor:
    """Wraps an OSNet-x0.25 torchreid model to produce L2-normalised embeddings.

    The underlying torchreid model is loaded lazily on first call.  Weights
    are loaded from ``weights_path`` if provided; otherwise torchreid will
    use random initialisation (useful only for tests / shape checks).

    Args:
        model_name: torchreid model name, e.g. ``"osnet_x0_25"``.
        weights_path: Path to a pre-downloaded ``.pth`` weights file, or
            ``""`` to skip weight loading.
        device: Torch device string — ``"cpu"``, ``"mps"``, or ``"cuda"``.
    """

    def __init__(
        self,
        model_name: str = "osnet_x0_25",
        weights_path: str = "",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.weights_path = weights_path
        self.device = device
        self._model = None  # lazy-loaded

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the torchreid OSNet model on first use."""
        try:
            import torchreid
        except ImportError as exc:
            raise ImportError(
                "torchreid is required for ReID. Install it with:\n"
                "pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
            ) from exc

        import torch

        model = torchreid.models.build_model(
            name=self.model_name,
            num_classes=1,  # not used for feature extraction
            pretrained=False,
        )
        if self.weights_path:
            state = torch.load(self.weights_path, map_location="cpu")
            # torchreid checkpoints may be nested under 'state_dict'
            state_dict = state.get("state_dict", state)
            model.load_state_dict(state_dict, strict=False)

        model = model.to(self.device)
        model.eval()
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    def _preprocess(self, crop: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Resize, colour-convert and ImageNet-normalise a BGR crop.

        Returns:
            Float32 numpy array of shape ``(1, 3, H, W)`` ready for the model.
        """
        import cv2

        img = cv2.resize(crop, (_OSNET_INPUT_W, _OSNET_INPUT_H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        # HWC → NCHW
        return img.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    def _run_model(self, input_nchw: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Forward-pass a preprocessed ``(1, 3, H, W)`` array through the model.

        Returns:
            Raw (un-normalised) feature vector of shape ``(D,)``.
        """
        import torch

        tensor = torch.from_numpy(input_nchw).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)  # (1, D)
        return feat.cpu().numpy().squeeze(0)  # (D,)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, crop: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Return a normalised L2 embedding for a single BGR crop.

        Args:
            crop: HxWx3 uint8 BGR image (OpenCV convention).

        Returns:
            1-D float32 numpy array of length equal to the model's feature
            dimension (512 for osnet_x0_25).  Always L2-normalised.
        """
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)

        preprocessed = self._preprocess(crop)
        feat = self._run_model(preprocessed)

        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat.astype(np.float32)

    def extract_from_detection(
        self,
        detection: Detection,
        frame: npt.NDArray[np.uint8],
    ) -> None:
        """Crop the detection from ``frame`` and populate its embedding in-place."""
        x1, y1, x2, y2 = map(int, detection.bbox)
        h, w = frame.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        crop = frame[y1:y2, x1:x2]
        detection.embedding = self.extract(crop)

    def extract_batch(
        self,
        detections: list[Detection],
        frame: npt.NDArray[np.uint8],
    ) -> None:
        """Populate embeddings for all detections in a list in-place."""
        for det in detections:
            self.extract_from_detection(det, frame)
