from __future__ import annotations

import numpy as np

from bev_tracker.core.detection import Detection
from bev_tracker.core.track import Track
from bev_tracker.tracking.association import associate
from bev_tracker.tracking.kalman import KalmanFilter


class Tracker:
    """World-space multi-object tracker using Kalman prediction and Hungarian
    association.

    Tracks are created for unmatched detections, updated on match, and pruned
    when they exceed ``max_age`` frames without a detection.

    Args:
        max_age: Frames a track can go unmatched before deletion.
        embedding_momentum: EMA coefficient for updating a track's embedding on
            match. ``track.emb = momentum * old + (1-momentum) * new``.
        dist_threshold: Spatial gate (metres) passed to the association step.
        cost_threshold: Maximum association cost to accept a match.
        alpha: Spatial weight in the cost function.
        beta: Embedding weight in the cost function.
        trail_length: Maximum history positions stored per track.
    """

    def __init__(
        self,
        max_age: int = 30,
        embedding_momentum: float = 0.9,
        dist_threshold: float = 2.0,
        cost_threshold: float = 5.0,
        alpha: float = 0.6,
        beta: float = 0.4,
        trail_length: int = 30,
    ) -> None:
        self.max_age = max_age
        self.embedding_momentum = embedding_momentum
        self.dist_threshold = dist_threshold
        self.cost_threshold = cost_threshold
        self.alpha = alpha
        self.beta = beta
        self.trail_length = trail_length

        self._tracks: list[Track] = []
        self._next_id: int = 0
        self._frame_idx: int = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def tracks(self) -> list[Track]:
        """All currently active tracks."""
        return self._tracks

    # ── Private helpers ───────────────────────────────────────────────────────

    def _new_track(self, det: Detection) -> Track:
        kf = KalmanFilter(initial_xy=det.world_xy)
        track = Track(
            id=self._next_id,
            world_xy=det.world_xy,
            velocity=(0.0, 0.0),
            embedding=det.embedding.copy() if det.embedding.size > 0 else np.empty(0, np.float32),
            kalman_filter=kf,
            last_seen=self._frame_idx,
            age=0,
        )
        self._next_id += 1
        return track

    def _predict(self) -> None:
        for track in self._tracks:
            track.kalman_filter.predict()
            track.world_xy = track.kalman_filter.position
            track.velocity = track.kalman_filter.velocity
            track.age += 1

    def _update_track(self, track: Track, det: Detection) -> None:
        track.kalman_filter.update(det.world_xy)
        track.world_xy = track.kalman_filter.position
        track.velocity = track.kalman_filter.velocity
        track.last_seen = self._frame_idx
        track.age = 0
        track.update_history(self.trail_length)

        # EMA embedding update
        if det.embedding.size > 0:
            if track.embedding.size == 0:
                track.embedding = det.embedding.copy()
            else:
                track.embedding = (
                    self.embedding_momentum * track.embedding
                    + (1.0 - self.embedding_momentum) * det.embedding
                )
            norm = np.linalg.norm(track.embedding)
            if norm > 0:
                track.embedding /= norm

    def _prune(self) -> None:
        self._tracks = [t for t in self._tracks if t.age < self.max_age]

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, detections: list[Detection]) -> list[Track]:
        """Process one frame of detections and return the updated track list.

        Args:
            detections: All detections from the current frame (all cameras
                merged, with world_xy populated).

        Returns:
            The current list of active tracks after update and pruning.
        """
        self._frame_idx += 1
        self._predict()

        matches, unmatched_tracks, unmatched_dets = associate(
            self._tracks,
            detections,
            dist_threshold=self.dist_threshold,
            cost_threshold=self.cost_threshold,
            alpha=self.alpha,
            beta=self.beta,
        )

        for track_idx, det_idx in matches:
            self._update_track(self._tracks[track_idx], detections[det_idx])

        for det_idx in unmatched_dets:
            self._tracks.append(self._new_track(detections[det_idx]))

        self._prune()
        return self._tracks
