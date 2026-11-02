"""Fit a law on the start of a series and judge it on the unseen end."""

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from kepler.laws import Law

MIN_TRAIN_POINTS = 10


@dataclass(frozen=True)
class FitResult:
    name: str
    params: dict[str, float]
    y0: float
    v0: float
    rmse_train: float
    rmse_test: float
    rel_err_test: float  # rmse_test / (max(y) - min(y)) over the whole series
    split_time: float  # first time of the test part


def simulate(law: Law, params: dict[str, float], y0: float, v0: float, t: np.ndarray) -> np.ndarray:
    """Integrate the law from (y0, v0) at t[0]; returns y at each t (NaN after a solver failure)."""
    sol = solve_ivp(
        lambda ti, s: [s[1], law.accel(ti, s[0], s[1], params)],
        (t[0], t[-1]),
        [y0, v0],
        method="RK45",
        t_eval=t,
        rtol=1e-8,
        atol=1e-10,
    )
    y = np.full(t.size, np.nan)
    y[: sol.y.shape[1]] = sol.y[0]
    return y


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def fit_law(law: Law, t: np.ndarray, y: np.ndarray, train_frac: float = 0.7) -> FitResult:
    split = math.floor(train_frac * len(t))
    if split < MIN_TRAIN_POINTS:
        raise ValueError(f"need at least {MIN_TRAIN_POINTS} training points, got {split}")
    t_train, y_train = t[:split], y[:split]
    names = list(law.params)

    def unpack(x: np.ndarray) -> dict[str, float]:
        return dict(zip(names, map(float, x[2:]), strict=True))

    def residuals(x: np.ndarray) -> np.ndarray:
        r = simulate(law, unpack(x), x[0], x[1], t_train) - y_train
        # A diverging trajectory must still give the optimizer a finite, large cost.
        return np.nan_to_num(r, nan=1e6, posinf=1e6, neginf=-1e6)

    x0 = [y_train[0], np.polyfit(t_train[:3], y_train[:3], 1)[0], *(b[2] for b in law.params.values())]
    lower = [-np.inf, -np.inf, *(b[0] for b in law.params.values())]
    upper = [np.inf, np.inf, *(b[1] for b in law.params.values())]
    x = least_squares(residuals, x0, bounds=(lower, upper), max_nfev=2000).x
    params, y0, v0 = unpack(x), float(x[0]), float(x[1])
    rmse_test = _rmse(simulate(law, params, y0, v0, t)[split:], y[split:])
    return FitResult(
        name=law.name,
        params=params,
        y0=y0,
        v0=v0,
        rmse_train=_rmse(simulate(law, params, y0, v0, t_train), y_train),
        rmse_test=rmse_test,
        rel_err_test=rmse_test / float(np.max(y) - np.min(y)),
        split_time=float(t[split]),
    )


def rank(results: list[FitResult]) -> list[FitResult]:
    return sorted(results, key=lambda r: r.rmse_test)
