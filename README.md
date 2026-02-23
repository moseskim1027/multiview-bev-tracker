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
│   └── default.yaml           # Runtime configuration
├── data/                      # Input video/image data (gitignored)
├── models/                    # Downloaded weights (gitignored)
├── scripts/
│   └── run_tracker.py         # CLI entry point
├── src/bev_tracker/
│   ├── core/                  # Shared dataclasses (Camera, Detection, Track)
│   ├── detection/             # YOLOv8 wrapper
│   ├── projection/            # Homography projection
│   ├── reid/                  # OSNet embedding extractor
│   ├── tracking/              # Kalman filter + Hungarian association + Tracker
│   ├── visualization/         # BEV renderer
│   └── pipeline.py            # Main loop
└── tests/
```

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
