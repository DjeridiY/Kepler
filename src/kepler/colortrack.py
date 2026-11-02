"""Offline color tracker: a free fallback that follows a colored object from a start box."""

import cv2
import numpy as np

from kepler.video import read_frames

_HALF_WIDTH = np.array([10, 60, 60])


class TrackError(Exception):
    pass


def _circular_hue_median(hue: np.ndarray) -> int:
    """Median hue, robust to red pixels straddling the 0/179 wrap."""
    direct = int(np.median(hue))
    shifted = (int(np.median((hue + 90) % 180)) - 90) % 180

    def spread(m: int) -> float:
        d = np.abs(hue - m)
        return float(np.mean(np.minimum(d, 180 - d)))

    return direct if spread(direct) <= spread(shifted) else shifted


def sample_color(image_bgr: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """HSV bounds (low, high) around the median color of the central half of `box` (x2, y2 inclusive).

    A low hue above the high hue means the range wraps through 0/179.
    """
    x1, y1, x2, y2 = box
    w, h = x2 - x1 + 1, y2 - y1 + 1
    cx1, cy1 = x1 + w // 4, y1 + h // 4
    patch = image_bgr[cy1 : cy1 + max(1, h // 2), cx1 : cx1 + max(1, w // 2)]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int64)
    median = np.array([_circular_hue_median(hsv[:, 0]), *np.median(hsv[:, 1:], axis=0).astype(np.int64)])
    low, high = median - _HALF_WIDTH, median + _HALF_WIDTH
    low[0] %= 180
    high[0] %= 180
    low[1:] = np.clip(low[1:], 0, 255)
    high[1:] = np.clip(high[1:], 0, 255)
    return low.astype(np.uint8), high.astype(np.uint8)


def _mask(image_bgr: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    if low[0] <= high[0]:
        return cv2.inRange(hsv, low, high)
    wrap_high = np.array([179, high[1], high[2]], np.uint8)
    wrap_low = np.array([0, low[1], low[2]], np.uint8)
    return cv2.bitwise_or(cv2.inRange(hsv, low, wrap_high), cv2.inRange(hsv, wrap_low, high))


def color_track(
    path: str, box0: tuple[int, int, int, int], *, max_jump_px: float = 80.0
) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the frames where the object was found and its sub-pixel centers there, shape (N, 2)."""
    x1, y1, x2, y2 = box0
    previous = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    bounds = None
    indices: list[int] = []
    centers: list[np.ndarray] = []
    total = 0
    for frame in read_frames(path):
        total += 1
        if bounds is None:
            bounds = sample_color(frame.image, box0)
        _, _, stats, centroids = cv2.connectedComponentsWithStats(_mask(frame.image, *bounds))
        near = [k for k in range(1, len(stats)) if np.linalg.norm(centroids[k] - previous) <= max_jump_px]
        if not near:
            continue
        best = max(near, key=lambda k: stats[k, cv2.CC_STAT_AREA])
        previous = centroids[best]
        indices.append(frame.index)
        centers.append(previous)
    lost = total - len(indices)
    if len(indices) < 0.5 * total:
        raise TrackError(f"object lost on {lost}/{total} frames")
    return np.array(indices, dtype=int), np.array(centers, dtype=float).reshape(-1, 2)
