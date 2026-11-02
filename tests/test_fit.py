import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from kepler.fit import fit_law, rank, simulate
from kepler.laws import BUILTIN_LAWS, load_law

LAWS = [load_law(src) for src in BUILTIN_LAWS.values()]


def _series(accel, y0, t, sigma):
    sol = solve_ivp(lambda _, s: [s[1], accel(s[0], s[1])], (t[0], t[-1]), [y0, 0.0], t_eval=t, rtol=1e-10, atol=1e-12)
    return sol.y[0] + np.random.default_rng(0).normal(0.0, sigma, t.size)


def _pendulum(t):
    return _series(lambda y, v: -21.8 * math.sin(y) - 2 * 0.05 * v, 0.14, t, 0.002)


def _uniform(duration):
    return np.arange(0.0, duration, 1 / 60)


def _check_damped_pendulum(t):
    results = rank([fit_law(law, t, _pendulum(t)) for law in LAWS])
    best = results[0]
    assert best.name == "damped pendulum (linear)"
    assert best.params["w2"] == pytest.approx(21.8, rel=0.005)
    assert best.params["gamma"] == pytest.approx(0.05, rel=0.1)
    assert best.rel_err_test < 0.01


def test_damped_pendulum_ranks_first():
    _check_damped_pendulum(_uniform(10.0))


def test_damped_pendulum_with_non_uniform_times():
    steps = np.random.default_rng(1).uniform(0.75, 1.25, 700) / 60
    t = np.concatenate([[0.0], np.cumsum(steps)])
    _check_damped_pendulum(t[t <= 10.0])


def test_quadratic_drag_beats_free_fall():
    t = _uniform(0.8)
    y = _series(lambda _, v: -9.81 - 0.8 * v * abs(v), 1.5, t, 0.001)
    names = [r.name for r in rank([fit_law(law, t, y) for law in LAWS])]
    assert names[0] == "fall with quadratic drag"
    assert names.index("fall with quadratic drag") < names.index("free fall")


@pytest.mark.parametrize("train_frac", [0.7, 0.55])
def test_test_part_never_influences_fit(train_frac):
    t = _uniform(10.0)
    y = _pendulum(t)
    law = load_law(BUILTIN_LAWS["damped pendulum (linear)"])
    split = math.floor(train_frac * t.size)
    shifted = y.copy()
    shifted[split:] += 1.0
    a = fit_law(law, t, y, train_frac)
    b = fit_law(law, t, shifted, train_frac)
    assert a.split_time == b.split_time == t[split]
    for key in a.params:
        assert b.params[key] == pytest.approx(a.params[key], abs=1e-9)
    assert (b.y0, b.v0, b.rmse_train) == pytest.approx((a.y0, a.v0, a.rmse_train), abs=1e-9)
    assert b.rmse_test > a.rmse_test


def test_too_few_training_points():
    law = load_law(BUILTIN_LAWS["free fall"])
    t = np.linspace(0.0, 1.0, 14)
    with pytest.raises(ValueError):
        fit_law(law, t, 1.0 - 4.9 * t**2)


def test_simulate_free_fall_matches_closed_form():
    law = load_law(BUILTIN_LAWS["free fall"])
    t = np.array([0.0, 0.1, 0.35, 0.4, 1.0])
    np.testing.assert_allclose(simulate(law, {"g": 9.8}, 2.0, 1.0, t), 2.0 + t - 4.9 * t**2, atol=1e-8)
