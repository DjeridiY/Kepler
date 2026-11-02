"""Home experiments: where to put the cup for a ball rolling off a table, and a camera scale."""

import math

import numpy as np
from scipy.optimize import least_squares

from kepler.measure import period

G = 9.81


class ExperimentError(Exception):
    pass


def _travel(b: float, tau: np.ndarray) -> np.ndarray:
    """(1 - exp(-b tau)) / b, which tends to tau when b -> 0."""
    if b == 0:
        return tau
    return -np.expm1(-b * tau) / b


def fit_roll(t: np.ndarray, x: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares fit of x(t) = x0 + v0 (1 - exp(-b (t - t[0]))) / b with b >= 0: (t[0], x0, v0, b)."""
    t, x = np.asarray(t, dtype=float), np.asarray(x, dtype=float)
    if len(t) < 5:
        raise ExperimentError("need at least 5 points to fit a roll")
    tau = t - t[0]
    v_guess = max(float(np.polyfit(tau, x, 1)[0]), 1e-6)
    fit = least_squares(
        lambda p: p[0] + p[1] * _travel(p[2], tau) - x,
        [x[0], v_guess, 0.0],
        bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]),
    )
    x0, v0, b = fit.x
    return float(t[0]), float(x0), float(v0), float(b)


def edge_crossing(t0: float, x0: float, v0: float, b: float, x_edge: float) -> tuple[float, float]:
    """Absolute time and speed when the fitted roll reaches x_edge."""
    d = x_edge - x0
    if d < 0:
        raise ExperimentError("the edge is behind the starting point")
    # Under v' = -b v the speed drops linearly with distance: v = v0 - b d.
    v_edge = v0 - b * d
    if v0 <= 0 or v_edge <= 0:
        raise ExperimentError("the ball stops before the edge")
    dt = d / v0 if b == 0 else -math.log1p(-b * d / v0) / b
    return t0 + dt, v_edge


def landing_distance(v_edge: float, table_height_m: float, g: float = G) -> float:
    if table_height_m <= 0 or v_edge < 0:
        raise ValueError("need table_height_m > 0 and v_edge >= 0")
    return v_edge * math.sqrt(2 * table_height_m / g)


def spring_period(t: np.ndarray, z: np.ndarray) -> float:
    z = np.asarray(z, dtype=float)
    return period(t, z - z.mean()).period_s


def mass_from_periods(m_ref_kg: float, period_ref_s: float, period_obj_s: float) -> float:
    if period_ref_s <= 0 or period_obj_s <= 0:
        raise ValueError("periods must be > 0")
    return m_ref_kg * (period_obj_s / period_ref_s) ** 2
