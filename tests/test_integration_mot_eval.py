"""MOT ballpark evaluation against WILDTRACK ground-truth annotations.

Runs the full detect → project → reid → track pipeline over N frames and
compares the resulting track positions to the ground-truth world positions
using Hungarian matching.

Key findings at confidence_threshold=0.3, N=5 frames:
  recall  0.92 – 0.97  (>= 0.75 threshold)
  mean_dist  0.28 – 0.37 m  (<= 1.0 m threshold)
  precision  0.22 – 0.33  (informational only — expected low because
                            multiple cameras project the same person to
                            slightly different world points, spawning
                            multiple tracks per GT person)

Requires:
    - models/yolov8n.pt
    - models/osnet_x0_25_market.pth
    - data/wildtrack/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

MODELS_DIR = Path("models")
WILDTRACK_ROOT = Path("data/wildtrack")
YOLO_WEIGHTS = MODELS_DIR / "yolov8n.pt"
OSNET_WEIGHTS = MODELS_DIR / "osnet_x0_25_market.pth"

_MATCH_THRESHOLD_M = 2.0  # metres — generous enough to handle projection noise
_MIN_RECALL = 0.75  # conservative floor given observed 0.92-0.97
_MAX_MEAN_DIST_M = 1.0  # metres — well above observed 0.28-0.37 m

_skip = pytest.mark.skipif(
    not YOLO_WEIGHTS.exists() or not OSNET_WEIGHTS.exists() or not WILDTRACK_ROOT.exists(),
    reason="Models or data/wildtrack/ not found",
)

_N_FRAMES = 5  # frames to run; results gathered from all of them


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run_frame(frame_idx, dataset, detector, extractor):
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


def _hungarian_match(gt_xy, track_xy, threshold_m):
    """Match GT positions to track positions; return per-match distances."""
    from scipy.optimize import linear_sum_assignment

    if len(gt_xy) == 0 or len(track_xy) == 0:
        return [], len(gt_xy), len(track_xy)

    gt = np.array(gt_xy)
    pred = np.array(track_xy)
    dist = np.linalg.norm(gt[:, np.newaxis] - pred[np.newaxis, :], axis=2)  # (G, T)
    row_ind, col_ind = linear_sum_assignment(dist)
    matched_dists = [
        float(dist[r, c]) for r, c in zip(row_ind, col_ind) if dist[r, c] < threshold_m
    ]
    unmatched_gt = len(gt_xy) - len(matched_dists)
    unmatched_tracks = len(track_xy) - len(matched_dists)
    return matched_dists, unmatched_gt, unmatched_tracks


# ── Module-scoped fixtures ────────────────────────────────────────────────────


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


@pytest.fixture(scope="module")
def mot_results(detector, extractor, dataset):
    """Pre-compute tracking output + GT annotations for _N_FRAMES frames.

    Running detection once here is shared across all test methods, keeping
    total runtime comparable to a single 5-frame run.
    """
    from bev_tracker.tracking.tracker import Tracker

    tracker = Tracker(max_age=15, dist_threshold=3.0, cost_threshold=5.0)
    results = []
    for frame_idx in range(_N_FRAMES):
        dets = _run_frame(frame_idx, dataset, detector, extractor)
        tracks = tracker.step(dets)
        gt_anns = dataset.load_annotations(frame_idx)
        gt_xy = [ann["world_xy"] for ann in gt_anns]
        track_xy = [t.world_xy for t in tracks]
        matched_dists, unmatched_gt, unmatched_tracks = _hungarian_match(
            gt_xy, track_xy, _MATCH_THRESHOLD_M
        )
        results.append(
            {
                "frame_idx": frame_idx,
                "n_gt": len(gt_xy),
                "n_tracks": len(track_xy),
                "n_matched": len(matched_dists),
                "matched_dists": matched_dists,
                "unmatched_gt": unmatched_gt,
                "recall": len(matched_dists) / len(gt_xy) if gt_xy else 0.0,
                "precision": len(matched_dists) / len(track_xy) if track_xy else 0.0,
                "mean_dist": float(np.mean(matched_dists)) if matched_dists else float("nan"),
            }
        )
    return results


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.integration
@_skip
class TestMOTBallpark:
    """Ballpark accuracy checks of the tracker against ground-truth annotations."""

    def test_recall_above_floor(self, mot_results):
        """Each frame must recover ≥ MIN_RECALL of GT persons within 2 m."""
        for r in mot_results:
            assert r["recall"] >= _MIN_RECALL, (
                f"Frame {r['frame_idx']}: recall={r['recall']:.2f} "
                f"(matched {r['n_matched']}/{r['n_gt']} GT persons, "
                f"threshold {_MIN_RECALL})"
            )

    def test_mean_matched_distance(self, mot_results):
        """For matched GT-track pairs the mean distance must be ≤ MAX_MEAN_DIST_M."""
        for r in mot_results:
            if np.isnan(r["mean_dist"]):
                continue
            assert r["mean_dist"] <= _MAX_MEAN_DIST_M, (
                f"Frame {r['frame_idx']}: mean matched dist={r['mean_dist']:.2f} m "
                f"> {_MAX_MEAN_DIST_M} m"
            )

    def test_average_recall_across_frames(self, mot_results):
        """Mean recall across all eval frames must stay ≥ MIN_RECALL."""
        avg = float(np.mean([r["recall"] for r in mot_results]))
        assert avg >= _MIN_RECALL, f"Average recall {avg:.2f} < {_MIN_RECALL}"

    def test_average_mean_dist_across_frames(self, mot_results):
        """Mean matched distance averaged across frames must be ≤ MAX_MEAN_DIST_M."""
        dists = [r["mean_dist"] for r in mot_results if not np.isnan(r["mean_dist"])]
        avg = float(np.mean(dists)) if dists else float("nan")
        assert not np.isnan(avg), "No matched pairs found across any frame"
        assert avg <= _MAX_MEAN_DIST_M, f"Average mean dist {avg:.2f} m > {_MAX_MEAN_DIST_M} m"

    def test_tracks_are_created(self, mot_results):
        """Every frame must produce at least one active track."""
        for r in mot_results:
            assert r["n_tracks"] >= 1, f"Frame {r['frame_idx']}: no active tracks"

    def test_summary(self, mot_results, capsys):
        """Print a human-readable per-frame summary (always passes)."""
        header = f"{'Frame':>5}  {'GT':>4}  {'Tracks':>6}  {'Matched':>7}  "
        header += f"{'Recall':>6}  {'Prec':>5}  {'MeanDist':>8}"
        lines = [header, "-" * len(header)]
        for r in mot_results:
            dist_str = f"{r['mean_dist']:.2f} m" if not np.isnan(r["mean_dist"]) else "  N/A  "
            lines.append(
                f"{r['frame_idx']:>5}  {r['n_gt']:>4}  {r['n_tracks']:>6}  "
                f"{r['n_matched']:>7}  {r['recall']:>6.2f}  {r['precision']:>5.2f}  "
                f"{dist_str:>8}"
            )
        avg_recall = np.mean([r["recall"] for r in mot_results])
        dists = [r["mean_dist"] for r in mot_results if not np.isnan(r["mean_dist"])]
        avg_dist = np.mean(dists) if dists else float("nan")
        lines.append("-" * len(header))
        lines.append(
            f"{'Mean':>5}  {'':>4}  {'':>6}  {'':>7}  {avg_recall:>6.2f}  {'':>5}  {avg_dist:.2f} m"
        )
        with capsys.disabled():
            print("\n" + "\n".join(lines) + "\n")
