#!/usr/bin/env python
"""Download required model weights to models/.

Usage:
    python scripts/setup_models.py            # download everything
    python scripts/setup_models.py --yolo     # YOLOv8n only
    python scripts/setup_models.py --osnet    # OSNet-x0.25 only

Models downloaded:
    models/yolov8n.pt            — YOLOv8-nano detector (via ultralytics)
    models/osnet_x0_25_market.pth — OSNet-x0.25 ReID (Market-1501, via gdown)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

MODELS_DIR = Path("models")
YOLO_WEIGHTS = MODELS_DIR / "yolov8n.pt"
OSNET_WEIGHTS = MODELS_DIR / "osnet_x0_25_market.pth"

# Google Drive file ID for osnet_x0_25 Market-1501 (256×128, amsgrad, ep180)
# Source: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO
OSNET_GDRIVE_ID = "1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"


# ── YOLOv8n ──────────────────────────────────────────────────────────────────


def download_yolov8n() -> None:
    if YOLO_WEIGHTS.exists():
        print(f"[yolov8n] Already exists: {YOLO_WEIGHTS}")
        return

    print("[yolov8n] Downloading via ultralytics …")
    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit("ultralytics not installed — run: pip install ultralytics")

    # ultralytics downloads to CWD first, then we move it to models/
    model = YOLO("yolov8n.pt")  # noqa: F841 — triggers download
    cwd_weights = Path("yolov8n.pt")
    if cwd_weights.exists():
        shutil.move(str(cwd_weights), YOLO_WEIGHTS)
        print(f"[yolov8n] Saved to {YOLO_WEIGHTS}")
    else:
        # Ultralytics may cache elsewhere in newer versions
        print(
            f"[yolov8n] Weight file not found in CWD after download. "
            f"Locate it in the ultralytics cache and copy to {YOLO_WEIGHTS}."
        )


# ── OSNet-x0.25 ───────────────────────────────────────────────────────────────


def download_osnet() -> None:
    if OSNET_WEIGHTS.exists():
        print(f"[osnet] Already exists: {OSNET_WEIGHTS}")
        return

    print("[osnet] Downloading via gdown …")
    try:
        import gdown
    except ImportError:
        sys.exit("gdown not installed — run: pip install gdown")

    url = f"https://drive.google.com/uc?id={OSNET_GDRIVE_ID}"
    gdown.download(url, str(OSNET_WEIGHTS), quiet=False)

    if OSNET_WEIGHTS.exists():
        print(f"[osnet] Saved to {OSNET_WEIGHTS}")
    else:
        print(
            "[osnet] Download may have failed. Manual alternative:\n"
            "  1. Visit https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO\n"
            "  2. Download osnet_x0_25 (Market-1501, 256x128)\n"
            f"  3. Save as {OSNET_WEIGHTS}"
        )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model weights")
    parser.add_argument("--yolo", action="store_true", help="Download YOLOv8n only")
    parser.add_argument("--osnet", action="store_true", help="Download OSNet only")
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    # If no flags, download everything
    all_ = not (args.yolo or args.osnet)

    if args.yolo or all_:
        download_yolov8n()
    if args.osnet or all_:
        download_osnet()

    print("\nDone. Verify with:")
    print("  ls -lh models/")


if __name__ == "__main__":
    main()
