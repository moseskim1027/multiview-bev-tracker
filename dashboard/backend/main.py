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
from fastapi import FastAPI
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


def _palette_hex(track_id: int) -> str:
    b, g, r = _PALETTE[track_id % len(_PALETTE)]
    return f"#{r:02x}{g:02x}{b:02x}"


def _palette_bgr(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


# ── Application state ─────────────────────────────────────────────────────────


class _State:
    dataset: WildtrackDataset | None = None
    detector: Detector | None = None
    extractor: ReIDExtractor | None = None
    tracker: Tracker | None = None
    renderer: BEVRenderer | None = None
    cameras: list[Camera] = []

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


def _draw_track_overlays(frame: np.ndarray, track_dicts: list[dict], cam_idx: int) -> np.ndarray:
    """Back-project each track's world position into this camera and draw a dot."""
    H = _s.cameras[cam_idx].homography
    img = frame.copy()
    h, w = img.shape[:2]
    for t in track_dicts:
        xw, yw = t["world_xy"]
        pt = H @ np.array([xw, yw, 1.0])
        if abs(pt[2]) < 1e-8:
            continue
        u, v = int(pt[0] / pt[2]), int(pt[1] / pt[2])
        if not (0 <= u < w and 0 <= v < h):
            continue
        bgr = _palette_bgr(t["id"])
        cv2.circle(img, (u, v), 18, bgr, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (u, v), 19, (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            img,
            str(t["id"]),
            (u + 22, v + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
    return img


# ── Frame processing ──────────────────────────────────────────────────────────


def _process_frame(frame_idx: int) -> dict:
    """Run the full pipeline for one frame.  Caller must hold _s.lock."""
    ds = _s.dataset
    all_dets = []
    raw_frames: list[np.ndarray] = []

    for cam_idx in range(ds.num_cameras):
        frame = ds.load_frame(cam_idx, frame_idx)
        raw_frames.append(frame)
        cam_id = f"C{cam_idx + 1}"
        dets = _s.detector.detect(frame, cam_id=cam_id)
        project_detections(dets, _s.cameras[cam_idx])
        _s.extractor.extract_batch(dets, frame)
        all_dets.extend(dets)

    tracks = _s.tracker.step(all_dets)

    # Serialise tracks — no numpy / dataclass objects
    track_dicts = [
        {
            "id": t.id,
            "world_xy": [float(t.world_xy[0]), float(t.world_xy[1])],
            "history": [[float(p[0]), float(p[1])] for p in t.history[-20:]],
            "color": _palette_hex(t.id),
        }
        for t in tracks
    ]

    # Camera thumbnails with track overlay
    cam_images: list[str] = []
    for cam_idx, raw in enumerate(raw_frames):
        overlay = _draw_track_overlays(raw, track_dicts, cam_idx)
        thumb = cv2.resize(overlay, (_CAM_OUT_W, _CAM_OUT_H))
        cam_images.append(_encode_jpeg(thumb))

    # BEV render
    bev = _s.renderer.render(tracks)
    bev_b64 = _encode_jpeg(bev, quality=85)

    return {
        "frame_idx": frame_idx,
        "tracks": track_dicts,
        "camera_images": cam_images,
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
    _s.tracker = Tracker(max_age=15, dist_threshold=3.0, cost_threshold=5.0)
    # Portrait BEV — world is 12 m wide × 36 m tall (1:3 aspect)
    _s.renderer = BEVRenderer(
        width=300,
        height=900,
        x_range=(-3.0, 9.0),
        y_range=(-9.0, 27.0),
        show_ids=True,
        show_trails=True,
    )
    _s.cameras = [
        Camera(id=f"C{i + 1}", homography=ds.homography(i)) for i in range(ds.num_cameras)
    ]
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


@app.get("/api/frame/{frame_idx}")
async def get_frame(frame_idx: int):
    if not _s.is_ready:
        return JSONResponse({"error": "models loading"}, status_code=503)
    if frame_idx not in _s.cache:
        return JSONResponse(
            {"error": "frame not yet processed", "processed_up_to": _s.processed_up_to},
            status_code=404,
        )
    return _s.cache[frame_idx]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
