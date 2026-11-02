import math

import numpy as np
import pytest

from kepler.measure import (
    MeasureError,
    amplitude_factor,
    angle_series,
    damping_rate,
    fit_pivot,
    g_from_pendulum,
    period,
    predict_period,
)


def test_period_on_non_uniform_timestamps():
    rng = np.random.default_rng(0)
    dt = (1 / 60) * (1 + rng.uniform(-0.25, 0.25, 1000))
    t = np.concatenate([[0.0], np.cumsum(dt)])
    t = t[t <= 10.0]
    p = period(t, np.cos(2 * np.pi * t / 1.234567))
    assert p.period_s == pytest.approx(1.234567, abs=1e-4)
    assert p.n_cycles >= 7
    assert p.std_s < 1e-3


def test_period_needs_two_upward_crossings():
    t = np.linspace(0, 1, 60)
    with pytest.raises(MeasureError, match="need at least 2 upward zero crossings"):
        period(t, np.cos(2 * np.pi * t / 1.3))


def test_damping_rate():
    t = np.arange(0, 20, 1 / 60)
    signal = np.exp(-0.05 * t) * np.cos(2 * np.pi * t / 1.3)
    assert damping_rate(t, signal) == pytest.approx(0.05, rel=0.02)


def test_damping_rate_needs_three_peaks():
    t = np.arange(0, 1.5, 1 / 60)
    with pytest.raises(MeasureError):
        damping_rate(t, np.cos(2 * np.pi * t / 1.3))


def _arc(scale: float = 1.0) -> np.ndarray:
    theta = np.linspace(-0.15, 0.15, 50)
    px, py, r = 0.2, 1.0, 0.45
    return scale * np.column_stack([px + r * np.sin(theta), py - r * np.cos(theta)])


def test_angle_series_recovers_pivot_length_and_angle():
    theta, pivot, length = angle_series(_arc())
    np.testing.assert_allclose(pivot, [0.2, 1.0], atol=1e-3)
    assert length == pytest.approx(0.45, abs=1e-3)
    np.testing.assert_allclose(theta, np.linspace(-0.15, 0.15, 50), atol=1e-9)


def test_angle_series_is_scale_invariant():
    theta1, _, length1 = angle_series(_arc())
    theta2, _, length2 = angle_series(_arc(2.0))
    np.testing.assert_allclose(theta2, theta1, atol=1e-9)
    assert length2 == pytest.approx(2 * length1)


def test_fit_pivot_needs_three_points():
    with pytest.raises(MeasureError):
        fit_pivot(np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_g_from_pendulum():
    t0 = 2 * math.pi * math.sqrt(0.45 / 9.81)
    assert g_from_pendulum(0.45, t0) == pytest.approx(9.81, abs=1e-9)
    t_wide = t0 * amplitude_factor(0.5)
    assert g_from_pendulum(0.45, t_wide, 0.5) == pytest.approx(9.81, abs=1e-9)


def test_predict_period():
    expected = 1.3 * math.sqrt(0.25 / 0.45)
    assert predict_period(1.3, 0.45, 0.25) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("args", [(1.3, 0.0, 0.25), (0.0, 0.45, 0.25), (1.3, 0.45, -1.0)])
def test_predict_period_rejects_non_positive(args):
    with pytest.raises(ValueError):
        predict_period(*args)
