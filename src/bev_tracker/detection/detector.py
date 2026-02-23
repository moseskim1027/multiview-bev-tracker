from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bev_tracker.core.detection import Detection

if TYPE_CHECKING:
    import numpy.typing as npt


class Detector:
    """Thin wrapper around a YOLOv8-nano model for per-camera detection.

    The underlying Ultralytics model is loaded lazily on first call so that
    importing this module does not trigger a GPU/model initialisation.

    Args:
        model_path: Path to YOLOv8 weights file or Ultralytics model name
            (e.g. ``"yolov8n.pt"``).  The weights are downloaded automatically
            on first use if not found locally.
        confidence_threshold: Minimum detector confidence to keep a detection.
        device: Torch device string — ``"cpu"``, ``"mps"``, or ``"cuda"``.
        target_classes: Set of COCO class ids to retain.  Defaults to
            ``{0}`` (person only).
    """

    DEFAULT_TARGET_CLASSES = {0}  # COCO: 0 = person

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        target_classes: set[int] | None = None,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.target_classes = (
            target_classes if target_classes is not None else self.DEFAULT_TARGET_CLASSES
        )
        self._model = None  # lazy-loaded

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the YOLO model on first use."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for detection. Install it with: pip install ultralytics"
            ) from exc
        self._model = YOLO(self.model_path)

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: npt.NDArray[np.uint8], cam_id: str) -> list[Detection]:
        """Run detection on a single BGR frame.

        Args:
            frame: HxWx3 uint8 BGR image (OpenCV convention).
            cam_id: Camera identifier to tag each returned Detection.

        Returns:
            List of Detection objects above the confidence threshold,
            filtered to ``target_classes``.  world_xy and embedding are
            left at their defaults (nan and empty) — filled in by later
            pipeline stages.
        """
        results = self.model(
            frame,
            device=self.device,
            verbose=False,
            conf=self.confidence_threshold,
            classes=list(self.target_classes),
        )

        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()  # (N, 4)
            confs = result.boxes.conf.cpu().numpy()  # (N,)
            for bbox, conf in zip(boxes_xyxy, confs):
                detections.append(
                    Detection(
                        cam_id=cam_id,
                        bbox=bbox.astype(np.float32),
                        confidence=float(conf),
                    )
                )

        return detections
