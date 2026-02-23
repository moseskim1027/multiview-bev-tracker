from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from bev_tracker.core.detection import Detection
from bev_tracker.core.track import Track

_LARGE_COST = 1e6


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2] between two L2-normalised vectors.

    Returns 1.0 (maximum meaningful distance) if either vector is empty
    or zero-norm, so unembedded detections are not favoured by appearance.
    """
    if a.size == 0 or b.size == 0:
        return 1.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)


def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def compute_cost_matrix(
    tracks: list[Track],
    detections: list[Detection],
    dist_threshold: float = 2.0,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> np.ndarray:
    """Build the (T × D) cost matrix combining spatial and embedding distance.

    For each (track, detection) pair:
    * If spatial distance exceeds ``dist_threshold`` the cost is set to a large
      sentinel value so the Hungarian solver ignores the pair.
    * Otherwise cost = alpha * spatial_dist + beta * cosine_dist.

    Args:
        tracks: List of active Track objects (rows).
        detections: List of Detection objects (columns).
        dist_threshold: Maximum world-space distance (metres) to consider a
            match feasible.
        alpha: Weight for the spatial-distance component.
        beta: Weight for the embedding-distance component.

    Returns:
        Float64 array of shape ``(len(tracks), len(detections))``.
    """
    T, D = len(tracks), len(detections)
    cost = np.full((T, D), _LARGE_COST, dtype=np.float64)

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            spatial = _euclidean(track.world_xy, det.world_xy)
            if spatial > dist_threshold:
                continue
            emb_dist = _cosine_distance(track.embedding, det.embedding)
            cost[i, j] = alpha * spatial + beta * emb_dist

    return cost


def associate(
    tracks: list[Track],
    detections: list[Detection],
    dist_threshold: float = 2.0,
    cost_threshold: float = 5.0,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Run the Hungarian algorithm to match tracks to detections.

    Args:
        tracks: Active tracks.
        detections: Current-frame detections.
        dist_threshold: Spatial gate passed to :func:`compute_cost_matrix`.
        cost_threshold: Matches with cost above this value are rejected.
        alpha: Spatial weight.
        beta: Embedding weight.

    Returns:
        Tuple of:
        * ``matches``: List of ``(track_idx, det_idx)`` pairs.
        * ``unmatched_tracks``: Track indices with no valid match.
        * ``unmatched_dets``: Detection indices with no valid match.
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    cost = compute_cost_matrix(tracks, detections, dist_threshold, alpha, beta)
    row_idx, col_idx = linear_sum_assignment(cost)

    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    matches: list[tuple[int, int]] = []

    for r, c in zip(row_idx, col_idx):
        if cost[r, c] > cost_threshold:
            continue
        matches.append((r, c))
        matched_tracks.add(r)
        matched_dets.add(c)

    unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
    unmatched_dets = [j for j in range(len(detections)) if j not in matched_dets]

    return matches, unmatched_tracks, unmatched_dets
