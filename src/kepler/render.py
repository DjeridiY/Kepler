"""Ghost overlay: draw the predicted motion and its uncertainty on the real video."""

from dataclasses import dataclass

import cv2
import numpy as np

from kepler.synth import _encode
from kepler.video import frame_times, nominal_fps, read_frames

GHOST_RADIUS_PX = 18
_CYAN = (255, 255, 0)
_MAGENTA = (255, 0, 255)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_BAND_ALPHA = 0.35
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.8
_SHIFT = 4  # cv2 fixed-point bits, so the ghost is drawn at sub-pixel positions


@dataclass(frozen=True)
class Overlay:
    times: np.ndarray  # seconds, one per source frame (kepler.video.frame_times)
    predicted_px: np.ndarray  # (N, 2) predicted object center in pixels, one per source frame
    band_px: float  # uncertainty radius in pixels (>= 0)
    split_time: float  # first time of the unseen part
    labels: list[str]  # lines drawn top-left, already formatted by the caller


def pendulum_to_px(theta: np.ndarray, pivot_px: tuple[float, float], length_px: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    return np.column_stack((pivot_px[0] + length_px * np.sin(theta), pivot_px[1] + length_px * np.cos(theta)))


def _fixed(point: np.ndarray) -> tuple[int, int]:
    return round(point[0] * (1 << _SHIFT)), round(point[1] * (1 << _SHIFT))


def _draw_text(img: np.ndarray, lines: list[str]) -> None:
    y = 10
    for line in lines:
        (w, h), base = cv2.getTextSize(line, _FONT, _FONT_SCALE, 2)
        cv2.rectangle(img, (5, y - 5), (15 + w, y + h + base + 5), _BLACK, cv2.FILLED)
        cv2.putText(img, line, (10, y + h), _FONT, _FONT_SCALE, _WHITE, 2, cv2.LINE_AA)
        y += h + base + 15


def _draw_frame(img: np.ndarray, k: int, overlay: Overlay, trail_s: float) -> np.ndarray:
    t = overlay.times[k]
    fit = t < overlay.split_time
    color = _CYAN if fit else _MAGENTA
    center = _fixed(overlay.predicted_px[k])
    band = img.copy()
    cv2.circle(band, center, round((GHOST_RADIUS_PX + overlay.band_px) * (1 << _SHIFT)), color, 3, cv2.LINE_AA, _SHIFT)
    img = cv2.addWeighted(band, _BAND_ALPHA, img, 1 - _BAND_ALPHA, 0)
    in_trail = (overlay.times > t) & (overlay.times <= t + trail_s)
    for point in overlay.predicted_px[in_trail]:
        cv2.circle(img, _fixed(point), 3 << _SHIFT, color, cv2.FILLED, cv2.LINE_AA, _SHIFT)
    cv2.circle(img, center, GHOST_RADIUS_PX << _SHIFT, color, 3, cv2.LINE_AA, _SHIFT)
    _draw_text(img, [*overlay.labels, "FIT" if fit else "PREDICTION"])
    return img


def render_overlay(src: str, dst: str, overlay: Overlay, *, trail_s: float = 0.5) -> None:
    """Write `src` with the ghost drawn on every frame to `dst` (H.264), keeping the source timestamps."""
    times = frame_times(src)
    if len(overlay.predicted_px) != len(times):
        raise ValueError(f"{len(overlay.predicted_px)} predicted positions for {len(times)} frames")
    frames = [_draw_frame(f.image, f.index, overlay, trail_s) for f in read_frames(src)]
    _encode(dst, frames, times, nominal_fps(src))
