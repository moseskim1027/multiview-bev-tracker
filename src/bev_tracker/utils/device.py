"""Device auto-detection for M1 (MPS), CUDA, and CPU."""

from __future__ import annotations


def get_device() -> str:
    """Return the best available torch device string.

    Priority:
    1. ``mps``  — Apple Silicon (arm64 Python + macOS >= 12.3)
    2. ``cuda`` — NVIDIA GPU
    3. ``cpu``  — always available

    Returns:
        One of ``"mps"``, ``"cuda"``, or ``"cpu"``.
    """
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
