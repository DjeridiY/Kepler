"""Pixel-to-meter scale from an A4 reference sheet lying in the plane of motion."""

import cv2
import numpy as np

A4_LONG_M = 0.297
A4_SHORT_M = 0.210


class ScaleError(Exception):
    pass


def _px_per_meter(side_a_px: float, side_b_px: float, long_m: float, short_m: float, max_aspect_error: float) -> float:
    long_px, short_px = max(side_a_px, side_b_px), min(side_a_px, side_b_px)
    aspect, expected = long_px / short_px, long_m / short_m
    if abs(aspect / expected - 1) > max_aspect_error:
        raise ScaleError(f"reference aspect {aspect:.3f} does not match {expected:.3f}: sheet not face-on")
    return (long_px / long_m + short_px / short_m) / 2


def px_per_meter_from_box(
    box: tuple[float, float, float, float],
    long_m: float = A4_LONG_M,
    short_m: float = A4_SHORT_M,
    max_aspect_error: float = 0.10,
) -> float:
    x1, y1, x2, y2 = box
    return _px_per_meter(abs(x2 - x1), abs(y2 - y1), long_m, short_m, max_aspect_error)


def px_per_meter_from_mask(
    mask: np.ndarray, long_m: float = A4_LONG_M, short_m: float = A4_SHORT_M, max_aspect_error: float = 0.10
) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ScaleError("empty reference mask")
    _, (w, h), _ = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    return _px_per_meter(w, h, long_m, short_m, max_aspect_error)


def to_meters(points_px: np.ndarray, px_per_m: float, origin_px: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Convert (N, 2) pixel points to meters, x to the right and y up."""
    if px_per_m <= 0:
        raise ValueError(f"px_per_m must be positive, got {px_per_m}")
    ox, oy = origin_px
    pts = np.asarray(points_px, dtype=float)
    return np.column_stack(((pts[:, 0] - ox) / px_per_m, (oy - pts[:, 1]) / px_per_m))
