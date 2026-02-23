# multiview-bev-tracker

A **training-free** multi-camera Bird's Eye View (BEV) stitching and Re-Identification (ReID)
tracking system. Uses classical geometry for spatial alignment, pretrained embeddings for
identity, and Kalman + Hungarian for temporal consistency — no labeled data or model
fine-tuning required.

## Architecture

```
Multi-camera frames
      ↓
2D Detector (YOLOv8-nano, pretrained)
      ↓
Bounding-box bottom-center → ground-plane projection (per-camera homography)
      ↓
Global BEV coordinates
      ↓
ReID embedding extraction (OSNet-x0.25, pretrained)
      ↓
Kalman prediction + Hungarian association
      ↓
Unified world tracks
```

## Key Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| Detector | YOLOv8-nano (Ultralytics) | Fastest pretrained 2D detector |
| ReID | OSNet-x0.25 (torchreid) | Lightest accurate embedding model |
| Tracker | Custom Kalman + Hungarian | World-coordinate SORT variant |
| Stitching | Per-camera homography | No depth prediction needed |

## Repository Layout

```
multiview-bev-tracker/
├── .github/workflows/ci.yml   # Lint-only CI
├── configs/
│   ├── default.yaml           # Generic runtime configuration
│   └── wildtrack.yaml         # WILDTRACK-specific configuration
├── data/                      # Input data (gitignored — see Dataset Setup below)
├── models/                    # Downloaded weights (gitignored — see Model Setup below)
├── scripts/
│   ├── run_tracker.py         # CLI entry point
│   └── setup_models.py        # Model weight downloader
├── src/bev_tracker/
│   ├── core/                  # Shared dataclasses (Camera, Detection, Track)
│   ├── datasets/              # Dataset loaders (WildtrackDataset)
│   ├── detection/             # YOLOv8 wrapper
│   ├── projection/            # Homography projection
│   ├── reid/                  # OSNet embedding extractor
│   ├── tracking/              # Kalman filter + Hungarian association + Tracker
│   ├── utils/                 # Shared utilities (device auto-detection)
│   ├── visualization/         # BEV renderer
│   └── pipeline.py            # Main loop
└── tests/
```

## Dataset Setup — WILDTRACK

The built-in dataset loader targets the
[WILDTRACK Multi-Camera Dataset](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/)
(Chavdarova et al., CVPR 2018) — a 7-camera, HD, synchronized pedestrian dataset
with ground-truth calibration and annotations.

### 1. Download

Request or download the dataset from the official source:

> **https://www.epfl.ch/labs/cvlab/data/data-wildtrack/**

You will receive (or can unzip) a directory called `Wildtrack_dataset/` containing:

```
Wildtrack_dataset/
├── Image_subsets/C{1-7}/   # 400 synchronized PNG frames per camera
├── calibrations/
│   ├── extrinsic/          # extr_<CameraName>.xml  (rvec + tvec in cm)
│   └── intrinsic_zero/     # intr_<CameraName>.xml  (3×3 camera matrix K)
└── annotations_positions/  # 00000000.json … (personID, positionID, bboxes)
```

### 2. Place data

Copy (or symlink) the dataset contents to `data/wildtrack/` inside this repo:

```bash
cp -r /path/to/Wildtrack_dataset/. data/wildtrack/
# or: ln -s /path/to/Wildtrack_dataset data/wildtrack
```

The `data/` directory is gitignored, so nothing is committed.

### 3. Verify

```bash
python - <<'EOF'
from bev_tracker.datasets import WildtrackDataset
ds = WildtrackDataset("data/wildtrack")
print(f"Cameras : {ds.num_cameras}")   # 7
print(f"Frames  : {ds.num_frames}")    # 400
print(f"H[C1]   : {ds.homography(0).shape}")  # (3, 3)
EOF
```

### Coordinate system

| Property | Value |
|----------|-------|
| Grid | 480 × 1440 cells |
| Cell size | 2.5 cm (0.025 m) |
| World X range | −3.0 m → 9.0 m |
| World Y range | −9.0 m → 27.0 m |
| Origin | `positionID = 0` → (−3.0 m, −9.0 m) |

Homographies are computed from the OpenCV XML calibration files and map the
z=0 ground plane (in **metres**) to image pixels. This is fully consistent with
the `project_to_world` / `Camera.homography_inv` API used throughout the pipeline.

---

## Model Setup

Download pretrained weights before running the pipeline or integration tests:

```bash
# Install gdown if not already present
pip install gdown

# Download YOLOv8n + OSNet-x0.25 (Market-1501) to models/
python scripts/setup_models.py
```

Individual downloads:

```bash
python scripts/setup_models.py --yolo    # YOLOv8n only
python scripts/setup_models.py --osnet   # OSNet only
```

torchreid must be installed from source (not on PyPI):

```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

---

## Quickstart (Docker)

```bash
# Build and run
docker compose up

# Run against custom video/config
docker compose run tracker python scripts/run_tracker.py \
  --config configs/default.yaml
```

## Local Development

```bash
pip install -e ".[dev]"

# Lint
ruff check .
ruff format --check .

# Tests
pytest
```

## Prerequisites

- Python 3.11+
- Per-camera homography matrices (ground-plane calibration)
- Synchronized multi-camera frames or video files
