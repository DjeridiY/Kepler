import math

import numpy as np
import pytest

from kepler.experiments import (
    ExperimentError,
    edge_crossing,
    fit_roll,
    landing_distance,
    mass_from_periods,
    spring_period,
)

X0, V0, B = 0.1, 1.6, 0.15


def _roll(noise_m: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, 0.8, 60)
    x = X0 + (V0 / B) * (1 - np.exp(-B * t))
    return t, x + np.random.default_rng(0).normal(0, noise_m, t.size)


def test_fit_roll_recovers_speed_and_drag():
    t0, _, v0, b = fit_roll(*_roll(0.0005))
    assert t0 == 0
    assert v0 == pytest.approx(V0, rel=0.01)
    assert abs(b - B) <= 0.05


def test_fit_roll_without_drag():
    t = np.linspace(0, 1, 20)
    _, x0, v0, b = fit_roll(t, 0.2 + 0.7 * t)
    assert (x0, v0, b) == pytest.approx((0.2, 0.7, 0.0), abs=1e-6)


def test_fit_roll_needs_five_points():
    with pytest.raises(ExperimentError):
        fit_roll(np.arange(4.0), np.arange(4.0))


def test_edge_crossing_speed_and_time():
    t, v = edge_crossing(0, 0, 1.6, 0.15, 0.35)
    assert v == pytest.approx(1.6 - 0.15 * 0.35, abs=1e-9)
    assert 0 + (1.6 / 0.15) * (1 - math.exp(-0.15 * t)) == pytest.approx(0.35, abs=1e-12)


def test_edge_crossing_without_drag_is_absolute_time():
    assert edge_crossing(2.0, 0.0, 0.5, 0.0, 1.0) == pytest.approx((4.0, 0.5))


def test_edge_crossing_ball_stops_before():
    with pytest.raises(ExperimentError, match="stops before the edge"):
        edge_crossing(0, 0, 0.1, 2.0, 1.0)


def test_landing_distance_value():
    assert landing_distance(1.583, 0.749) == pytest.approx(0.618589, abs=1e-4)
    assert landing_distance(0, 0.749) == 0


def test_cup_chain_lands_near_truth():
    edge, table = 0.35, 0.749
    truth = landing_distance(edge_crossing(0, X0, V0, B, edge)[1], table)
    _, v_edge = edge_crossing(*fit_roll(*_roll(0.0005)), edge)
    assert abs(landing_distance(v_edge, table) - truth) <= 0.005


def _spring(m: float, k: float = 35.0, gamma: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0, 4, 1 / 60)
    wd = math.sqrt(k / m - gamma**2)
    return t, 0.3 + 0.02 * np.exp(-gamma * t) * np.cos(wd * t)


def test_camera_scale_weighs_object():
    t1, t2 = spring_period(*_spring(0.5)), spring_period(*_spring(0.182))
    assert mass_from_periods(0.5, t1, t2) == pytest.approx(0.182, rel=0.01)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        mass_from_periods(0.5, 0, 1)
    with pytest.raises(ValueError):
        mass_from_periods(0.5, 1, -1)
    with pytest.raises(ValueError):
        landing_distance(1, 0)
    with pytest.raises(ValueError):
        landing_distance(-1, 1)
