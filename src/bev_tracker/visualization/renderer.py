from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bev_tracker.core.track import Track

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import numpy.typing as npt

# Colour palette — one BGR colour per track id (cycles)
_PALETTE = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
]


def _track_colour(track_id: int) -> tuple[int, int, int]:
    return _PALETTE[track_id % len(_PALETTE)]


class BEVRenderer:
    """Renders a 2-D top-down BEV map of active world-space tracks.

    Draws filled circles for current positions, trail polylines, and ID
    labels onto a blank canvas using OpenCV.

    Args:
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        x_range: World X extent ``(x_min, x_max)`` in metres.
        y_range: World Y extent ``(y_min, y_max)`` in metres.
        show_ids: Whether to draw track ID labels.
        show_trails: Whether to draw historical position trails.
    """

    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        x_range: tuple[float, float] = (-10.0, 10.0),
        y_range: tuple[float, float] = (-10.0, 10.0),
        show_ids: bool = True,
        show_trails: bool = True,
    ) -> None:
        self.width = width
        self.height = height
        self.x_range = x_range
        self.y_range = y_range
        self.show_ids = show_ids
        self.show_trails = show_trails

    # ── Coordinate conversion ─────────────────────────────────────────────────

    def world_to_pixel(self, x: float, y: float) -> tuple[int, int]:
        """Map a world (x, y) in metres to canvas (u, v) in pixels.

        World x increases rightward; world y increases upward.
        Canvas v increases downward, so y-axis is flipped.
        """
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range
        u = int((x - x_min) / (x_max - x_min) * self.width)
        v = int((1.0 - (y - y_min) / (y_max - y_min)) * self.height)
        return u, v

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_trail(self, canvas: npt.NDArray, track: Track, colour: tuple) -> None:
        if len(track.history) < 2:
            return
        pts = [self.world_to_pixel(*p) for p in track.history]
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            faded = tuple(int(c * alpha) for c in colour)
            cv2.line(canvas, pts[i - 1], pts[i], faded, 1, lineType=cv2.LINE_AA)

    def _draw_track(self, canvas: npt.NDArray, track: Track) -> None:
        colour = _track_colour(track.id)
        px, py = self.world_to_pixel(*track.world_xy)

        if self.show_trails:
            self._draw_trail(canvas, track, colour)

        cv2.circle(canvas, (px, py), 6, colour, -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 6, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        if self.show_ids:
            cv2.putText(
                canvas,
                str(track.id),
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def render(self, tracks: list[Track]) -> npt.NDArray[np.uint8]:
        """Return a HxWx3 uint8 BGR canvas with all tracks drawn.

        Args:
            tracks: Active tracks from the current pipeline step.

        Returns:
            Rendered BEV image (new array each call).
        """
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for track in tracks:
            self._draw_track(canvas, track)
        return canvas
