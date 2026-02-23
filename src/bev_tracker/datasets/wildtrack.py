"""WILDTRACK multi-camera dataset loader.

Dataset layout expected under ``root``:
    root/
        Image_subsets/C{1-7}/<frame>.png
        calibrations/
            extrinsic/extr_<name>.xml
            intrinsic_zero/intr_<name>.xml
        annotations_positions/<frame>.json

Reference: "WILDTRACK: A Multi-camera HD Dataset for Dense Unscripted
Pedestrian Detection" — Chavdarova et al., CVPR 2018.
"""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

# Camera folder names match the calibration file suffixes (order = C1…C7)
CAMERA_NAMES: list[str] = [
    "CVLab1",
    "CVLab2",
    "CVLab3",
    "CVLab4",
    "IDIAP1",
    "IDIAP2",
    "IDIAP3",
]

# Ground-plane grid (world coordinates in metres)
GRID_X_CELLS = 480  # columns
GRID_Y_CELLS = 1440  # rows
GRID_STEP_M = 0.025  # 2.5 cm per cell
ORIGIN_X_M = -3.0  # world x of cell (0, 0)
ORIGIN_Y_M = -9.0  # world y of cell (0, 0)

# Calibration files store translations in centimetres; multiply K by this scale
# to build a homography that outputs world coords in metres.
_CM_TO_M = 0.01


# ── Calibration helpers ───────────────────────────────────────────────────────


def load_intrinsic(xml_path: Path) -> np.ndarray:
    """Parse an OpenCV XML intrinsic file and return the 3×3 camera matrix K."""
    tree = ElementTree.parse(xml_path)
    node = tree.find("camera_matrix")
    rows = int(node.find("rows").text)
    cols = int(node.find("cols").text)
    data = np.fromstring(node.find("data").text, dtype=np.float64, sep=" ")
    return data.reshape(rows, cols)


def load_extrinsic(xml_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse an OpenCV XML extrinsic file.

    Returns:
        ``(rvec, tvec)`` where *tvec* is in **centimetres** (as stored).
    """
    tree = ElementTree.parse(xml_path)
    rvec = np.array(tree.find("rvec").text.split(), dtype=np.float64)
    tvec = np.array(tree.find("tvec").text.split(), dtype=np.float64)
    return rvec, tvec


def compute_homography(
    K: np.ndarray,
    rvec: np.ndarray,
    tvec_cm: np.ndarray,
) -> np.ndarray:
    """Build a 3×3 homography mapping the ground plane (z=0) to image pixels.

    The ground plane is parameterised in **metres** so the returned homography
    is consistent with the rest of the pipeline.

    The standard projection for a world point P=(X, Y, 0) is::

        p_img  ~  K @ [R | t] @ [X, Y, 0, 1]^T
               =  K @ [r₁ | r₂ | t] @ [X, Y, 1]^T

    where t is in the same units as the world coordinates (metres here).

    Args:
        K: 3×3 intrinsic matrix.
        rvec: 3-element Rodrigues rotation vector.
        tvec_cm: 3-element translation vector in centimetres.

    Returns:
        3×3 homography H such that ``p_img ~ H @ [x_m, y_m, 1]^T``.
    """
    import cv2

    R, _ = cv2.Rodrigues(rvec)
    t_m = (tvec_cm * _CM_TO_M).reshape(3, 1)
    H = K @ np.hstack([R[:, 0:1], R[:, 1:2], t_m])
    return H


def position_id_to_world(position_id: int) -> tuple[float, float]:
    """Convert a WILDTRACK ``positionID`` to world (x, y) in metres."""
    x = ORIGIN_X_M + GRID_STEP_M * (position_id % GRID_X_CELLS)
    y = ORIGIN_Y_M + GRID_STEP_M * (position_id // GRID_X_CELLS)
    return float(x), float(y)


# ── Dataset class ─────────────────────────────────────────────────────────────


class WildtrackDataset:
    """Load frames, calibrations, and annotations from a WILDTRACK dataset copy.

    Args:
        root: Path to the dataset root (containing ``Image_subsets/``,
            ``calibrations/``, and ``annotations_positions/``).
    """

    def __init__(self, root: str | Path = "data/wildtrack") -> None:
        self.root = Path(root)
        self._homographies: dict[int, np.ndarray] = {}  # cam_idx → H, lazy-cached

    # ── Internal path helpers ─────────────────────────────────────────────────

    def _intr_path(self, cam_idx: int) -> Path:
        name = CAMERA_NAMES[cam_idx]
        return self.root / "calibrations" / "intrinsic_zero" / f"intr_{name}.xml"

    def _extr_path(self, cam_idx: int) -> Path:
        name = CAMERA_NAMES[cam_idx]
        return self.root / "calibrations" / "extrinsic" / f"extr_{name}.xml"

    def _frame_path(self, cam_idx: int, frame_idx: int) -> Path:
        # Files are named 00000000.png, 00000005.png, 00000010.png …
        fname = f"{frame_idx * 5:08d}.png"
        return self.root / "Image_subsets" / f"C{cam_idx + 1}" / fname

    def _annotation_path(self, frame_idx: int) -> Path:
        fname = f"{frame_idx * 5:08d}.json"
        return self.root / "annotations_positions" / fname

    # ── Calibration ───────────────────────────────────────────────────────────

    def homography(self, cam_idx: int) -> np.ndarray:
        """Return the cached 3×3 H for camera *cam_idx* (0-based).

        H maps ``[x_m, y_m, 1]^T`` (world metres, z=0) → image pixel.
        """
        if cam_idx not in self._homographies:
            K = load_intrinsic(self._intr_path(cam_idx))
            rvec, tvec_cm = load_extrinsic(self._extr_path(cam_idx))
            self._homographies[cam_idx] = compute_homography(K, rvec, tvec_cm)
        return self._homographies[cam_idx]

    # ── Frames ────────────────────────────────────────────────────────────────

    def load_frame(self, cam_idx: int, frame_idx: int) -> np.ndarray:
        """Load and return a BGR uint8 frame.

        Args:
            cam_idx: 0-based camera index (0 = C1, …, 6 = C7).
            frame_idx: 0-based frame index (0 = first annotation frame).
        """
        import cv2

        path = self._frame_path(cam_idx, frame_idx)
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Frame not found: {path}")
        return img

    def synchronized_frames(self, frame_idx: int) -> dict[str, np.ndarray]:
        """Return ``{cam_id: BGR_frame}`` for all 7 cameras at *frame_idx*."""
        return {f"C{i + 1}": self.load_frame(i, frame_idx) for i in range(7)}

    # ── Annotations ───────────────────────────────────────────────────────────

    def load_annotations(self, frame_idx: int) -> list[dict]:
        """Parse the annotation JSON for *frame_idx*.

        Returns:
            List of dicts::

                {
                    "personID": int,
                    "world_xy": (x_m, y_m),    # metres
                    "views": [{"viewNum": int, "xmin": int, ...}, ...]
                }

            Views with all coordinates == -1 are not visible in that camera.
        """
        with open(self._annotation_path(frame_idx)) as f:
            raw = json.load(f)
        result = []
        for ann in raw:
            x, y = position_id_to_world(ann["positionID"])
            result.append(
                {
                    "personID": ann["personID"],
                    "world_xy": (x, y),
                    "views": ann["views"],
                }
            )
        return result

    # ── Metadata ──────────────────────────────────────────────────────────────

    @property
    def num_frames(self) -> int:
        """Number of annotated frames in the dataset."""
        return len(list((self.root / "annotations_positions").glob("*.json")))

    @property
    def num_cameras(self) -> int:
        return len(CAMERA_NAMES)
