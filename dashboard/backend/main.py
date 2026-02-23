"""FastAPI backend for the BEV Tracker dashboard.

Runs the full detect → project → reid → track pipeline over WILDTRACK frames
in a background thread.  REST endpoints expose cached frame data (camera images
with track overlays + BEV image) to the React frontend.
"""

from __future__ import annotations

import base64
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Resolve project root so bev_tracker is importable ─────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from bev_tracker.core.camera import Camera  # noqa: E402
from bev_tracker.datasets.wildtrack import WildtrackDataset  # noqa: E402
from bev_tracker.detection.detector import Detector  # noqa: E402
from bev_tracker.projection.homography import project_detections  # noqa: E402
from bev_tracker.reid.extractor import ReIDExtractor  # noqa: E402
from bev_tracker.tracking.tracker import Tracker  # noqa: E402
from bev_tracker.visualization.renderer import _PALETTE, BEVRenderer  # noqa: E402

# ── Palette helpers ───────────────────────────────────────────────────────────
# _PALETTE stores BGR tuples (OpenCV convention).  Convert to CSS hex for the UI.

_MAX_DEMO_FRAMES = 50  # cap to keep memory and startup time reasonable

# BEV canvas constants — must match the BEVRenderer config in _startup()
# Extended 1.5 m on every side beyond the WILDTRACK annotation grid
# so peripheral camera coverage is visible.
_BEV_W, _BEV_H = 300, 900
_BEV_X_RANGE = (-4.5, 10.5)  # was (-3, 9)  — 15 m wide
_BEV_Y_RANGE = (-10.5, 28.5)  # was (-9, 27) — 39 m tall


def _palette_hex(track_id: int) -> str:
    b, g, r = _PALETTE[track_id % len(_PALETTE)]
    return f"#{r:02x}{g:02x}{b:02x}"


def _palette_bgr(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


def _world_to_bev(xw: float, yw: float) -> tuple[int, int]:
    """Map world metres to canvas pixels, clamped to the canvas bounds."""
    x_min, x_max = _BEV_X_RANGE
    y_min, y_max = _BEV_Y_RANGE
    u = int((xw - x_min) / (x_max - x_min) * _BEV_W)
    v = int((1.0 - (yw - y_min) / (y_max - y_min)) * _BEV_H)
    return max(0, min(u, _BEV_W - 1)), max(0, min(v, _BEV_H - 1))


def _world_to_bev_raw(xw: float, yw: float) -> tuple[int, int]:
    """Same as _world_to_bev but without clamping (for footprint hulls)."""
    x_min, x_max = _BEV_X_RANGE
    y_min, y_max = _BEV_Y_RANGE
    u = int((xw - x_min) / (x_max - x_min) * _BEV_W)
    v = int((1.0 - (yw - y_min) / (y_max - y_min)) * _BEV_H)
    return u, v


# Camera colours in OpenCV BGR order.
# _CAM_FILL_COLORS are used for filled polygons in the coverage map and as the
# canonical legend swatches exported to the frontend.
_CAM_FILL_COLORS: list[tuple[int, int, int]] = [
    (50, 60, 210),  # C1 — red
    (50, 210, 60),  # C2 — green
    (210, 60, 50),  # C3 — blue
    (50, 210, 210),  # C4 — yellow
    (210, 50, 210),  # C5 — magenta
    (210, 210, 50),  # C6 — cyan
    (150, 150, 150),  # C7 — grey
]
_CAM_OUTLINE_COLORS: list[tuple[int, int, int]] = [
    tuple(max(0, c - 60) for c in bgr)
    for bgr in _CAM_FILL_COLORS  # type: ignore[misc]
]


def _bgr_to_css(bgr: tuple[int, int, int]) -> str:
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def _camera_hulls() -> list[np.ndarray | None]:
    """Compute convex-hull BEV polygons for each camera's ground footprint."""
    if not _s.cameras:
        return []
    W_img, H_img = 1920, 1080
    top = H_img // 3
    sample_pts = (
        [(int(f * W_img), H_img) for f in [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]]
        + [(0, v) for v in range(H_img, top - 1, -H_img // 8)]
        + [(W_img, v) for v in range(top, H_img + 1, H_img // 8)]
    )
    hulls: list[np.ndarray | None] = []
    for cam in _s.cameras:
        H_inv = cam.homography_inv
        bev_pts = []
        for u, v in sample_pts:
            pt = H_inv @ np.array([float(u), float(v), 1.0])
            if pt[2] < 0.05:
                continue
            xw, yw = pt[0] / pt[2], pt[1] / pt[2]
            if abs(xw) > 40 or abs(yw) > 60:
                continue
            bev_pts.append(_world_to_bev_raw(xw, yw))
        if len(bev_pts) < 3:
            hulls.append(None)
        else:
            hulls.append(cv2.convexHull(np.array(bev_pts, dtype=np.int32)))
    return hulls


def _draw_camera_footprints(canvas: np.ndarray) -> None:
    """Draw camera footprint outlines onto *canvas* (portrait BEV space)."""
    for cam_idx, hull in enumerate(_camera_hulls()):
        if hull is None:
            continue
        cv2.polylines(canvas, [hull], True, _CAM_OUTLINE_COLORS[cam_idx], 1, lineType=cv2.LINE_AA)


def _make_coverage_image() -> str:
    """Render a static portrait diagram of camera footprints over the world grid.

    Called once at startup after _s.cameras is populated.  The result is cached
    and served from GET /api/bev-coverage so the frontend can display it on demand.
    """
    canvas = np.full((_BEV_H, _BEV_W, 3), 22, dtype=np.uint8)  # dark background

    # World-coordinate grid lines at 5 m intervals
    grid_color = (45, 45, 55)
    xlo, xhi = _BEV_X_RANGE
    ylo, yhi = _BEV_Y_RANGE
    import math as _math

    for xi in range(int(_math.floor(xlo / 5)) * 5, int(_math.ceil(xhi / 5)) * 5 + 1, 5):
        u, _ = _world_to_bev(float(xi), ylo)
        cv2.line(canvas, (u, 0), (u, _BEV_H - 1), grid_color, 1)
    for yi in range(int(_math.floor(ylo / 5)) * 5, int(_math.ceil(yhi / 5)) * 5 + 1, 5):
        _, v = _world_to_bev(xlo, float(yi))
        cv2.line(canvas, (0, v), (_BEV_W - 1, v), grid_color, 1)

    # Axis labels (metres) every 10 m
    label_col = (90, 90, 110)
    for xi in range(int(_math.ceil(xlo / 10)) * 10, int(xhi) + 1, 10):
        u, v0 = _world_to_bev(float(xi), yhi)
        cv2.putText(
            canvas,
            f"{xi}m",
            (u - 10, v0 + 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            label_col,
            1,
            cv2.LINE_AA,
        )
    for yi in range(int(_math.ceil(ylo / 10)) * 10, int(yhi) + 1, 10):
        u0, v = _world_to_bev(xlo, float(yi))
        cv2.putText(
            canvas,
            f"{yi}m",
            (u0 + 2, v + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            label_col,
            1,
            cv2.LINE_AA,
        )

    # Origin crosshair
    u0, v0 = _world_to_bev(0.0, 0.0)
    cv2.line(canvas, (u0 - 8, v0), (u0 + 8, v0), (90, 90, 100), 1)
    cv2.line(canvas, (u0, v0 - 8), (u0, v0 + 8), (90, 90, 100), 1)

    # Filled camera footprints (50 % alpha) then outlines on top
    hulls = _camera_hulls()
    fill_layer = canvas.copy()
    for cam_idx, hull in enumerate(hulls):
        if hull is not None:
            cv2.fillPoly(fill_layer, [hull], _CAM_FILL_COLORS[cam_idx])
    cv2.addWeighted(fill_layer, 0.50, canvas, 0.50, 0, canvas)

    # Outlines on top of the fills
    _draw_camera_footprints(canvas)

    # Rotate to landscape so the coverage map matches the BEV panel orientation.
    return _encode_jpeg(cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE), quality=90)


def _render_bev_selected(track_dicts: list[dict], selected_id: int) -> np.ndarray:
    """Render a BEV with the selected track highlighted; all others drawn grey."""
    canvas = np.zeros((_BEV_H, _BEV_W, 3), dtype=np.uint8)

    def _in_bounds(px: int, py: int) -> bool:
        return 0 <= px < _BEV_W and 0 <= py < _BEV_H

    # Non-selected tracks — visible grey circles + grey trails
    for t in track_dicts:
        if t["id"] == selected_id:
            continue
        xw, yw = t["world_xy"]
        u, v = _world_to_bev_raw(xw, yw)  # unclamped — skip if outside canvas
        if not _in_bounds(u, v):
            continue
        history = t.get("history", [])
        for i in range(1, len(history)):
            pu, pv = _world_to_bev_raw(*history[i - 1])
            qu, qv = _world_to_bev_raw(*history[i])
            if not (_in_bounds(pu, pv) and _in_bounds(qu, qv)):
                continue
            grey = int(80 * i / len(history))
            cv2.line(canvas, (pu, pv), (qu, qv), (grey, grey, grey), 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (u, v), 5, (110, 110, 110), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (u, v), 5, (160, 160, 160), 1, lineType=cv2.LINE_AA)

    # Selected track — colored, larger, with trail and highlight ring
    for t in track_dicts:
        if t["id"] != selected_id:
            continue
        xw, yw = t["world_xy"]
        u, v = _world_to_bev_raw(xw, yw)  # unclamped
        if not _in_bounds(u, v):
            continue
        bgr = _palette_bgr(t["id"])
        history = t.get("history", [])
        for i in range(1, len(history)):
            pu, pv = _world_to_bev_raw(*history[i - 1])
            qu, qv = _world_to_bev_raw(*history[i])
            if not (_in_bounds(pu, pv) and _in_bounds(qu, qv)):
                continue
            alpha = i / len(history)
            faded = tuple(int(c * alpha) for c in bgr)
            cv2.line(canvas, (pu, pv), (qu, qv), faded, 2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (u, v), 9, bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (u, v), 9, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (u, v), 13, (0, 220, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(t["id"]),
            (u + 15, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            lineType=cv2.LINE_AA,
        )
    return cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)


# ── Application state ─────────────────────────────────────────────────────────


class _State:
    dataset: WildtrackDataset | None = None
    detector: Detector | None = None
    extractor: ReIDExtractor | None = None
    tracker: Tracker | None = None
    renderer: BEVRenderer | None = None
    cameras: list[Camera] = []
    coverage_image: str | None = None  # base64 JPEG of static camera-footprint diagram

    # frame_idx → serialised dict ready for the API response
    cache: dict[int, dict] = {}
    processed_up_to: int = -1
    is_ready: bool = False
    lock: threading.Lock = threading.Lock()


_s = _State()

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="BEV Tracker Dashboard API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Image helpers ─────────────────────────────────────────────────────────────

_CAM_OUT_W, _CAM_OUT_H = 640, 360  # camera thumbnail size


def _encode_jpeg(img: np.ndarray, quality: int = 75) -> str:
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode()


def _draw_bbox_overlays(
    frame: np.ndarray,
    track_dicts: list[dict],
    cam_dets: list,
) -> np.ndarray:
    """Draw colored bounding boxes on *frame* for detections matched to tracks.

    Each detection is matched to the nearest confirmed track (within
    _BBOX_MATCH_RADIUS metres in world space).  Unmatched detections
    (noise / unconfirmed tracks) are not drawn.
    """
    if not track_dicts or not cam_dets:
        return frame
    _BBOX_MATCH_RADIUS = 1.5  # metres

    img = frame.copy()
    for det in cam_dets:
        if not det.has_world_position:
            continue
        dxy = np.array(det.world_xy)
        # Find the closest confirmed track within the match radius
        best_t, best_dist = None, _BBOX_MATCH_RADIUS
        for t in track_dicts:
            d = float(np.linalg.norm(dxy - np.array(t["world_xy"])))
            if d < best_dist:
                best_dist, best_t = d, t
        if best_t is None:
            continue

        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        b, g, r = _palette_bgr(best_t["id"])
        # Colored bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (b, g, r), 2, lineType=cv2.LINE_AA)
        # Label with filled background
        label = f"#{best_t['id']}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, y1 - lh - 6), (x1 + lw + 4, y1), (b, g, r), -1)
        cv2.putText(
            img,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    return img


# ── Frame processing ──────────────────────────────────────────────────────────


def _process_frame(frame_idx: int) -> dict:
    """Run the full pipeline for one frame.  Caller must hold _s.lock."""
    ds = _s.dataset
    dets_by_cam: list[list] = []  # per-camera detections (pre-dedup) for bbox drawing
    all_dets = []
    raw_frames: list[np.ndarray] = []

    for cam_idx in range(ds.num_cameras):
        frame = ds.load_frame(cam_idx, frame_idx)
        raw_frames.append(frame)
        cam_id = f"C{cam_idx + 1}"
        dets = _s.detector.detect(frame, cam_id=cam_id)
        project_detections(dets, _s.cameras[cam_idx])
        _s.extractor.extract_batch(dets, frame)
        dets_by_cam.append(dets)
        all_dets.extend(dets)

    # Deduplicate detections: the same person appears in multiple cameras at
    # slightly different world positions.  Keep only the highest-confidence
    # detection within a 1 m radius to avoid spawning duplicate tracks.
    _DEDUP_RADIUS = 1.0  # metres
    deduped: list = []
    for det in sorted(all_dets, key=lambda d: -d.confidence):
        xy = np.array(det.world_xy)
        if any(np.linalg.norm(xy - np.array(k.world_xy)) < _DEDUP_RADIUS for k in deduped):
            continue
        deduped.append(det)

    tracks = _s.tracker.step(deduped)

    # Build per-track, per-camera bounding boxes.  For each camera's detections,
    # find the closest active track (within _BBOX_MATCH_RADIUS) and record its bbox.
    # These are used by the frontend canvas to draw exact bboxes instead of
    # projected dots, both in the default view and when a track is selected.
    _BBOX_MATCH_RADIUS = 1.5  # metres
    track_bboxes: dict[int, dict[int, list[float]]] = {}
    for cam_idx, cam_dets in enumerate(dets_by_cam):
        for det in cam_dets:
            if not det.has_world_position:
                continue
            dxy = np.array(det.world_xy)
            best_t, best_dist = None, _BBOX_MATCH_RADIUS
            for t in tracks:
                d = float(np.linalg.norm(dxy - np.array(t.world_xy)))
                if d < best_dist:
                    best_dist, best_t = d, t
            if best_t is not None:
                # Keep the detection closest to the track per camera
                existing = track_bboxes.get(best_t.id, {}).get(cam_idx)
                if existing is None or best_dist < existing[4]:
                    track_bboxes.setdefault(best_t.id, {})[cam_idx] = [
                        float(det.bbox[0]),
                        float(det.bbox[1]),
                        float(det.bbox[2]),
                        float(det.bbox[3]),
                        best_dist,  # stored temporarily for comparison; stripped below
                    ]

    # Strip the distance sentinel from bboxes
    for tid in track_bboxes:
        for ci in track_bboxes[tid]:
            track_bboxes[tid][ci] = track_bboxes[tid][ci][:4]

    # Serialise tracks — filter noisy / ghost tracks:
    #   1. Must have been seen in at least _MIN_HITS frames (not a flash detection)
    #   2. Current position must be inside the visible BEV world range
    _MIN_HITS = 6
    xlo, xhi = _BEV_X_RANGE
    ylo, yhi = _BEV_Y_RANGE
    track_dicts = [
        {
            "id": t.id,
            "world_xy": [float(t.world_xy[0]), float(t.world_xy[1])],
            "history": [[float(p[0]), float(p[1])] for p in t.history[-20:]],
            "color": _palette_hex(t.id),
            "camera_bboxes": track_bboxes.get(t.id, {}),
        }
        for t in tracks
        if len(t.history) >= _MIN_HITS
        and xlo <= t.world_xy[0] <= xhi
        and ylo <= t.world_xy[1] <= yhi
    ]

    # Camera thumbnails — two sets:
    #   cam_images       : bounding-box overlays for confirmed tracks (default view)
    #   cam_images_plain : raw thumbnails (used when a track is selected so the
    #                      canvas can draw just the highlighted bbox/dot)
    cam_images: list[str] = []
    cam_images_plain: list[str] = []
    for cam_idx, (raw, cam_dets) in enumerate(zip(raw_frames, dets_by_cam)):
        # Plain thumbnail — no overlays
        thumb_plain = cv2.resize(raw, (_CAM_OUT_W, _CAM_OUT_H))
        cam_images_plain.append(_encode_jpeg(thumb_plain))
        # Bbox overlay drawn on full-res frame, then resized
        overlay = _draw_bbox_overlays(raw, track_dicts, cam_dets)
        cam_images.append(_encode_jpeg(cv2.resize(overlay, (_CAM_OUT_W, _CAM_OUT_H))))

    # BEV render — rotate 90° CW so the image is landscape for the horizontal panel.
    bev = _s.renderer.render(tracks)
    bev = cv2.rotate(bev, cv2.ROTATE_90_CLOCKWISE)
    bev_b64 = _encode_jpeg(bev, quality=85)

    return {
        "frame_idx": frame_idx,
        "tracks": track_dicts,
        "camera_images": cam_images,
        "cam_images_plain": cam_images_plain,
        "bev_image": bev_b64,
    }


def _background_worker() -> None:
    """Process frames 0 … _MAX_DEMO_FRAMES sequentially and cache results."""
    max_frames = min(_s.dataset.num_frames, _MAX_DEMO_FRAMES)
    for idx in range(max_frames):
        with _s.lock:
            data = _process_frame(idx)
            _s.cache[idx] = data
            _s.processed_up_to = idx


# ── Lifecycle ─────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def _startup() -> None:
    data_root = _PROJECT_ROOT / "data" / "wildtrack"
    yolo_path = _PROJECT_ROOT / "models" / "yolov8n.pt"
    osnet_path = _PROJECT_ROOT / "models" / "osnet_x0_25_market.pth"

    ds = WildtrackDataset(data_root)
    _s.dataset = ds
    _s.detector = Detector(str(yolo_path), confidence_threshold=0.3, device="cpu")
    _s.extractor = ReIDExtractor("osnet_x0_25", str(osnet_path), device="cpu")
    _s.tracker = Tracker(max_age=5, dist_threshold=3.0, cost_threshold=5.0)
    # Portrait BEV — extended 1.5 m beyond the annotation grid on every side
    _s.renderer = BEVRenderer(
        width=_BEV_W,
        height=_BEV_H,
        x_range=_BEV_X_RANGE,
        y_range=_BEV_Y_RANGE,
        show_ids=True,
        show_trails=True,
    )
    _s.cameras = [
        Camera(id=f"C{i + 1}", homography=ds.homography(i)) for i in range(ds.num_cameras)
    ]
    _s.coverage_image = _make_coverage_image()
    _s.is_ready = True

    t = threading.Thread(target=_background_worker, daemon=True)
    t.start()


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/api/info")
async def info() -> dict:
    return {
        "num_frames": min(_s.dataset.num_frames, _MAX_DEMO_FRAMES) if _s.dataset else 0,
        "num_cameras": _s.dataset.num_cameras if _s.dataset else 0,
        "processed_up_to": _s.processed_up_to,
        "is_ready": _s.is_ready,
    }


@app.get("/api/bev-coverage")
async def get_bev_coverage():
    if not _s.is_ready or _s.coverage_image is None:
        return JSONResponse({"error": "loading"}, status_code=503)
    legend = [
        {"cam": f"C{i + 1}", "color": _bgr_to_css(bgr)}
        for i, bgr in enumerate(_CAM_FILL_COLORS[: len(_s.cameras)])
    ]
    return {"image": _s.coverage_image, "legend": legend}


@app.get("/api/homographies")
async def get_homographies():
    if not _s.is_ready:
        return JSONResponse({"error": "models loading"}, status_code=503)
    return {
        "homographies": [cam.homography.tolist() for cam in _s.cameras],
        "orig_w": 1920,
        "orig_h": 1080,
    }


@app.get("/api/frame/{frame_idx}")
async def get_frame(
    frame_idx: int,
    selected: int | None = Query(default=None, description="Highlight this track ID"),
):
    if not _s.is_ready:
        return JSONResponse({"error": "models loading"}, status_code=503)
    if frame_idx not in _s.cache:
        return JSONResponse(
            {"error": "frame not yet processed", "processed_up_to": _s.processed_up_to},
            status_code=404,
        )

    cached = _s.cache[frame_idx]
    if selected is None:
        return cached

    # Selected view: plain camera frames (no overlays) so the canvas can draw
    # just the one track; BEV re-rendered with grey non-selected + colored selected.
    track_dicts = cached["tracks"]
    return {
        "frame_idx": frame_idx,
        "tracks": track_dicts,
        "camera_images": cached["cam_images_plain"],
        "bev_image": _encode_jpeg(_render_bev_selected(track_dicts, selected), quality=85),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
