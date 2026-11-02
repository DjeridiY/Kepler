"""Period, damping and pendulum geometry: what a stopwatch and a tape measure can check."""

import math
from dataclasses import dataclass

import numpy as np


class MeasureError(Exception):
    pass


@dataclass(frozen=True)
class Period:
    period_s: float
    n_cycles: int
    std_s: float


def fit_pivot(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Kasa algebraic circle fit of the bob positions (N, 2): (center xy, radius)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        raise MeasureError("need at least 3 points to fit a pivot")
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    (d, e, f), *_ = np.linalg.lstsq(a, -(x**2 + y**2), rcond=None)
    center = np.array([-d / 2, -e / 2])
    return center, float(np.sqrt(center @ center - f))


def angle_series(points: np.ndarray, pivot: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Angle from the downward vertical (positive to the right, y up), pivot, mean length."""
    pts = np.asarray(points, dtype=float)
    if pivot is None:
        pivot, _ = fit_pivot(pts)
    dx, dy = pts[:, 0] - pivot[0], pivot[1] - pts[:, 1]
    return np.arctan2(dx, dy), pivot, float(np.mean(np.hypot(dx, dy)))


def _upward_crossings(t: np.ndarray, s: np.ndarray) -> np.ndarray:
    i = np.flatnonzero((s[:-1] <= 0) & (s[1:] > 0))
    return t[i] - s[i] * (t[i + 1] - t[i]) / (s[i + 1] - s[i])


def period(t: np.ndarray, signal: np.ndarray) -> Period:
    t, s = np.asarray(t, dtype=float), np.asarray(signal, dtype=float)
    crossings = _upward_crossings(t, s - s.mean())
    if len(crossings) < 2:
        raise MeasureError("need at least 2 upward zero crossings")
    gaps = np.diff(crossings)
    return Period(float(gaps.mean()), len(gaps), float(gaps.std()))


def damping_rate(t: np.ndarray, signal: np.ndarray) -> float:
    """Decay rate gamma (1/s) of the peak envelope, assuming |peak| ~ exp(-gamma t)."""
    t, s = np.asarray(t, dtype=float), np.asarray(signal, dtype=float)
    a = np.abs(s)
    sign_changes = np.flatnonzero(np.sign(s[:-1]) != np.sign(s[1:]))
    peaks = []
    # Only whole half-cycles: a half-cycle cut by the series edge biases gamma.
    for lo, hi in zip(sign_changes[:-1] + 1, sign_changes[1:] + 1, strict=True):
        i = lo + int(np.argmax(a[lo:hi]))
        if a[i - 1] < a[i] > a[i + 1]:
            peaks.append(i)
    if len(peaks) < 3:
        raise MeasureError("need at least 3 peaks between zero crossings")
    slope, _ = np.polyfit(t[peaks], np.log(a[peaks]), 1)
    return float(-slope)


def amplitude_factor(theta0_rad: float) -> float:
    """T / T0 for a pendulum released at amplitude theta0 (series to theta0^4)."""
    return 1 + theta0_rad**2 / 16 + 11 * theta0_rad**4 / 3072


def g_from_pendulum(length_m: float, period_s: float, theta0_rad: float = 0.0) -> float:
    t0 = period_s / amplitude_factor(theta0_rad)
    return 4 * math.pi**2 * length_m / t0**2


def predict_period(period1_s: float, length1_m: float, length2_m: float) -> float:
    if min(period1_s, length1_m, length2_m) <= 0:
        raise ValueError("period and lengths must be positive")
    return period1_s * math.sqrt(length2_m / length1_m)
