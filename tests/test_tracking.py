"""Unit tests for the tracking module (Kalman, association, Tracker)."""

import numpy as np
import pytest

from bev_tracker.core.detection import Detection
from bev_tracker.core.track import Track
from bev_tracker.tracking.association import associate, compute_cost_matrix
from bev_tracker.tracking.kalman import KalmanFilter
from bev_tracker.tracking.tracker import Tracker

# ── Helpers ───────────────────────────────────────────────────────────────────


def _det(x: float, y: float, emb: np.ndarray | None = None, cam_id: str = "cam0") -> Detection:
    det = Detection(
        cam_id=cam_id,
        bbox=[0.0, 0.0, 10.0, 20.0],
        confidence=0.9,
        world_xy=(x, y),
    )
    if emb is not None:
        det.embedding = emb
    return det


def _unit_emb(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


# ── KalmanFilter ──────────────────────────────────────────────────────────────


class TestKalmanFilter:
    def test_initial_position(self):
        kf = KalmanFilter((3.0, 4.0))
        assert kf.position == pytest.approx((3.0, 4.0))

    def test_initial_velocity_zero(self):
        kf = KalmanFilter((0.0, 0.0))
        assert kf.velocity == pytest.approx((0.0, 0.0))

    def test_predict_advances_position(self):
        kf = KalmanFilter((0.0, 0.0))
        kf.x[2] = 1.0  # set vx = 1
        kf.predict()
        assert kf.position[0] == pytest.approx(1.0)

    def test_update_corrects_towards_measurement(self):
        kf = KalmanFilter((0.0, 0.0))
        kf.predict()
        kf.update((5.0, 5.0))
        x, y = kf.position
        # Should move toward the measurement
        assert x > 0.0
        assert y > 0.0

    def test_multiple_predict_update_cycles(self):
        kf = KalmanFilter((0.0, 0.0))
        for i in range(10):
            kf.predict()
            kf.update((float(i), 0.0))
        # After tracking a rightward-moving target, position should be positive
        assert kf.position[0] > 0.0

    def test_covariance_decreases_after_update(self):
        kf = KalmanFilter((0.0, 0.0))
        kf.predict()
        p_before = np.trace(kf.P)
        kf.update((0.0, 0.0))
        assert np.trace(kf.P) < p_before


# ── compute_cost_matrix ───────────────────────────────────────────────────────


class TestComputeCostMatrix:
    def _track(self, x, y, emb=None) -> Track:
        t = Track(id=0, world_xy=(x, y))
        if emb is not None:
            t.embedding = emb
        return t

    def test_shape(self):
        tracks = [self._track(0, 0), self._track(1, 0)]
        dets = [_det(0, 0), _det(1, 0), _det(2, 0)]
        cost = compute_cost_matrix(tracks, dets)
        assert cost.shape == (2, 3)

    def test_large_cost_for_distant_pair(self):
        tracks = [self._track(0, 0)]
        dets = [_det(100, 100)]
        cost = compute_cost_matrix(tracks, dets, dist_threshold=2.0)
        assert cost[0, 0] > 1e5

    def test_low_cost_for_close_identical_embedding(self):
        emb = _unit_emb(0)
        tracks = [self._track(0, 0, emb=emb)]
        dets = [_det(0.1, 0.0, emb=emb)]
        cost = compute_cost_matrix(tracks, dets, dist_threshold=2.0)
        assert cost[0, 0] < 0.5  # close position + same embedding

    def test_empty_tracks(self):
        cost = compute_cost_matrix([], [_det(0, 0)])
        assert cost.shape == (0, 1)

    def test_empty_detections(self):
        cost = compute_cost_matrix([self._track(0, 0)], [])
        assert cost.shape == (1, 0)


# ── associate ─────────────────────────────────────────────────────────────────


class TestAssociate:
    def _track(self, x, y, emb=None) -> Track:
        t = Track(id=0, world_xy=(x, y))
        if emb is not None:
            t.embedding = emb
        return t

    def test_perfect_match(self):
        emb = _unit_emb(0)
        tracks = [self._track(1.0, 0.0, emb=emb)]
        dets = [_det(1.0, 0.0, emb=emb)]
        matches, unmatched_t, unmatched_d = associate(tracks, dets)
        assert matches == [(0, 0)]
        assert unmatched_t == []
        assert unmatched_d == []

    def test_no_match_when_too_far(self):
        tracks = [self._track(0.0, 0.0)]
        dets = [_det(50.0, 50.0)]
        matches, unmatched_t, unmatched_d = associate(tracks, dets, dist_threshold=2.0)
        assert matches == []
        assert unmatched_t == [0]
        assert unmatched_d == [0]

    def test_empty_tracks(self):
        matches, unmatched_t, unmatched_d = associate([], [_det(0, 0)])
        assert matches == []
        assert unmatched_t == []
        assert unmatched_d == [0]

    def test_empty_detections(self):
        matches, unmatched_t, unmatched_d = associate([self._track(0, 0)], [])
        assert matches == []
        assert unmatched_t == [0]
        assert unmatched_d == []

    def test_two_tracks_two_dets_correct_assignment(self):
        emb0 = _unit_emb(0)
        emb1 = _unit_emb(1)
        tracks = [self._track(0.0, 0.0, emb0), self._track(5.0, 0.0, emb1)]
        dets = [_det(5.1, 0.0, emb1), _det(0.1, 0.0, emb0)]
        matches, _, _ = associate(tracks, dets)
        match_dict = dict(matches)
        assert match_dict[0] == 1  # track 0 (at 0,0) matches det at (0.1, 0)
        assert match_dict[1] == 0  # track 1 (at 5,0) matches det at (5.1, 0)


# ── Tracker ───────────────────────────────────────────────────────────────────


class TestTracker:
    def test_new_detection_creates_track(self):
        tracker = Tracker()
        tracker.step([_det(1.0, 2.0)])
        assert len(tracker.tracks) == 1
        assert tracker.tracks[0].world_xy == pytest.approx((1.0, 2.0), abs=0.5)

    def test_two_close_detections_across_frames_stay_one_track(self):
        tracker = Tracker(dist_threshold=2.0)
        tracker.step([_det(0.0, 0.0)])
        tracker.step([_det(0.1, 0.0)])
        assert len(tracker.tracks) == 1

    def test_track_id_increments(self):
        tracker = Tracker()
        tracker.step([_det(0.0, 0.0)])
        tracker.step([_det(10.0, 0.0)])  # far enough to be a new track
        ids = [t.id for t in tracker.tracks]
        assert len(set(ids)) == len(ids)  # all unique

    def test_track_pruned_after_max_age(self):
        tracker = Tracker(max_age=3)
        tracker.step([_det(0.0, 0.0)])  # create track
        for _ in range(4):
            tracker.step([])  # no detections — age increments
        assert len(tracker.tracks) == 0

    def test_track_age_resets_on_match(self):
        tracker = Tracker(max_age=5)
        tracker.step([_det(0.0, 0.0)])
        for _ in range(3):
            tracker.step([])  # age grows to 3
        tracker.step([_det(0.0, 0.0)])  # re-match
        assert tracker.tracks[0].age == 0

    def test_embedding_updated_on_match(self):
        emb_init = _unit_emb(0)
        emb_new = _unit_emb(1)
        tracker = Tracker(embedding_momentum=0.9)
        tracker.step([_det(0.0, 0.0, emb=emb_init)])
        tracker.step([_det(0.0, 0.0, emb=emb_new)])
        # Embedding should differ from the initial one
        assert not np.allclose(tracker.tracks[0].embedding, emb_init)

    def test_history_recorded(self):
        tracker = Tracker(trail_length=5)
        for i in range(3):
            tracker.step([_det(float(i), 0.0)])
        assert len(tracker.tracks[0].history) == 2  # updated twice after first step
