"""End-to-end multi-camera BEV tracking pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bev_tracker.core.camera import Camera
from bev_tracker.detection.detector import Detector
from bev_tracker.projection.homography import project_detections
from bev_tracker.reid.extractor import ReIDExtractor
from bev_tracker.tracking.tracker import Tracker
from bev_tracker.visualization.renderer import BEVRenderer

logger = logging.getLogger(__name__)


# ── Config helpers ─────────────────────────────────────────────────────────────


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def build_cameras(cfg: dict, detector: Detector, reid: ReIDExtractor) -> list[Camera]:
    """Instantiate Camera objects from the config ``cameras`` list."""
    cameras = []
    for cam_cfg in cfg["cameras"]:
        cam = Camera(
            id=cam_cfg["id"],
            homography=np.array(cam_cfg["homography"], dtype=np.float64),
            source=cam_cfg.get("source", ""),
        )
        cam.detector = detector
        cam.reid_model = reid
        cameras.append(cam)
    return cameras


# ── Pipeline ──────────────────────────────────────────────────────────────────


class Pipeline:
    """Wires together detection, projection, ReID, and tracking.

    Args:
        config_path: Path to a YAML config file (see ``configs/default.yaml``).
    """

    def __init__(self, config_path: str | Path) -> None:
        self.cfg = load_config(config_path)

        det_cfg = self.cfg["detection"]
        reid_cfg = self.cfg["reid"]
        track_cfg = self.cfg["tracking"]
        vis_cfg = self.cfg.get("visualization", {})

        self.detector = Detector(
            model_path=det_cfg["model"],
            confidence_threshold=det_cfg["confidence_threshold"],
            device=det_cfg["device"],
        )
        self.reid = ReIDExtractor(
            model_name=reid_cfg["model"],
            weights_path=reid_cfg.get("weights", ""),
            device=reid_cfg["device"],
        )
        self.tracker = Tracker(
            max_age=track_cfg["max_age"],
            embedding_momentum=track_cfg["embedding_momentum"],
            dist_threshold=track_cfg["dist_threshold"],
            cost_threshold=track_cfg["cost_threshold"],
            alpha=track_cfg["alpha"],
            beta=track_cfg["beta"],
        )
        self.cameras = build_cameras(self.cfg, self.detector, self.reid)

        self.renderer: BEVRenderer | None = None
        if vis_cfg.get("enabled", False):
            self.renderer = BEVRenderer(
                width=vis_cfg.get("bev_width", 800),
                height=vis_cfg.get("bev_height", 600),
                x_range=tuple(vis_cfg.get("world_x_range", [-10, 10])),
                y_range=tuple(vis_cfg.get("world_y_range", [-10, 10])),
                show_ids=vis_cfg.get("show_ids", True),
                show_trails=vis_cfg.get("show_trails", True),
            )

    def step(self, frames: dict[str, Any]) -> dict[str, Any]:
        """Process one synchronised set of camera frames.

        Args:
            frames: Mapping of ``camera_id -> BGR uint8 numpy array``.

        Returns:
            Dict with keys:
            * ``"tracks"``  — list of active Track objects
            * ``"bev"``     — BEV canvas (numpy array) or ``None`` if vis disabled
        """
        all_detections = []

        for cam in self.cameras:
            frame = frames.get(cam.id)
            if frame is None:
                continue

            dets = self.detector.detect(frame, cam_id=cam.id)
            project_detections(dets, cam)
            self.reid.extract_batch(dets, frame)
            all_detections.extend(dets)

        tracks = self.tracker.step(all_detections)

        bev = self.renderer.render(tracks) if self.renderer is not None else None
        return {"tracks": tracks, "bev": bev}

    def run(self) -> None:
        """Open video captures for all cameras and run the pipeline loop."""
        import cv2

        caps = {}
        for cam in self.cameras:
            src = cam.source
            source = int(src) if isinstance(src, str) and src.isdigit() else src
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                logger.warning("Cannot open source for camera %s: %s", cam.id, src)
            else:
                caps[cam.id] = cap

        logger.info("Starting pipeline with %d camera(s)", len(caps))

        try:
            while True:
                frames = {}
                for cam_id, cap in caps.items():
                    ret, frame = cap.read()
                    if ret:
                        frames[cam_id] = frame

                if not frames:
                    logger.info("All sources exhausted — stopping.")
                    break

                result = self.step(frames)

                if result["bev"] is not None:
                    cv2.imshow("BEV", result["bev"])
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            for cap in caps.values():
                cap.release()
            cv2.destroyAllWindows()


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run the multiview BEV tracker")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config (default: configs/default.yaml)",
    )
    args = parser.parse_args()

    pipeline = Pipeline(args.config)
    pipeline.run()


if __name__ == "__main__":
    main()
