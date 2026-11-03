"""Quick home experiments, each giving a sealed number to call before the take: period, cup spot, mass."""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np

from kepler import experiments, measure, pipeline, video
from kepler.budget import Budget
from kepler.pipeline import RunError
from kepler.sam import Transport

FLAT_TOL_M = 0.004
MIN_FLAT_FRAMES = 8


@dataclass(frozen=True)
class Tracked:
    times: np.ndarray  # all frame times
    indices: np.ndarray  # tracked frame indices
    centers: np.ndarray  # (N, 2) pixel centers at those frames
    px_per_m: float | None  # from the A4 sheet, None if not found
    width: int
    height: int


def track_clip(
    path: str,
    prompt: str,
    *,
    transport: Transport,
    budget: Budget | None = None,
    a4_prompt: str = "white sheet of paper",
) -> Tracked:
    times = video.frame_times(path)
    image0 = next(video.read_frames(path)).image
    indices, centers, px_per_m = pipeline.track_object(
        path, times, image0, prompt, transport=transport, budget=budget, a4_prompt=a4_prompt
    )
    height, width = image0.shape[:2]
    return Tracked(times, indices, centers, px_per_m, width, height)


def seal(values: list[float]) -> tuple[str, str]:
    """sha256 of the values rounded to 6 decimals as little-endian float64, and the UTC time."""
    rounded = np.round(np.asarray(values, dtype=float), 6).astype("<f8")
    return hashlib.sha256(rounded.tobytes()).hexdigest(), datetime.now(UTC).isoformat()


def _sealed(result: dict, value: float) -> dict:
    sha, computed = seal([value])
    return {**result, "seal_sha256": sha, "computed": computed}


def bet(
    clip_a: str,
    length_a_m: float,
    length_b_m: float,
    *,
    prompt: str,
    transport: Transport,
    budget: Budget | None = None,
) -> dict:
    tr = track_clip(clip_a, prompt, transport=transport, budget=budget)
    theta, _, _ = measure.angle_series(tr.centers * [1, -1])
    period_a = measure.period(tr.times[tr.indices], theta).period_s
    period_b = measure.predict_period(period_a, length_a_m, length_b_m)
    result = {"period_a_s": period_a, "period_b_pred_s": period_b, "length_a_m": length_a_m, "length_b_m": length_b_m}
    return _sealed(result, period_b)


def _flat_run(indices: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[int, int] | None:
    """Last run of consecutive frames rolling right on a flat level: (first, last) positions."""

    def flat(s: int, e: int) -> bool:
        ys = y[s : e + 1]
        return bool(
            np.all(np.diff(indices[s : e + 1]) == 1)
            and np.all(np.diff(x[s : e + 1]) > 0)
            and np.all(np.abs(ys - np.median(ys)) < FLAT_TOL_M)
        )

    found, s = None, 0
    for e in range(len(y)):
        while not flat(s, e):
            s += 1
        if e - s + 1 >= MIN_FLAT_FRAMES:
            found = (s, e)
    return found


def _cup_run(clip: str, tr: Tracked, table_height_m: float, cup_height_m: float, ball_diameter_m: float) -> dict:
    if tr.px_per_m is None:
        raise RunError("cup needs a scale: show the A4 sheet")
    t = tr.times[tr.indices]
    x, y = tr.centers[:, 0] / tr.px_per_m, -tr.centers[:, 1] / tr.px_per_m
    run = _flat_run(tr.indices, x, y)
    if run is None:
        raise RunError(f"no rolling on the table found in {clip}")
    s, e = run
    level = float(np.median(y[s : e + 1]))
    below = e + 1 + np.flatnonzero(y[e + 1 :] < level - FLAT_TOL_M)
    if below.size == 0:
        raise RunError(f"the ball never leaves the table in {clip}")
    k = int(below[0])
    x_edge = float(x[k])
    try:
        _, v_edge = experiments.edge_crossing(*experiments.fit_roll(t[s : e + 1], x[s : e + 1]), x_edge)
    except experiments.ExperimentError as err:
        raise RunError(f"{err} in {clip}") from err
    d_pred = experiments.landing_distance(v_edge, table_height_m + ball_diameter_m / 2 - cup_height_m)

    # Height of the ball center above the floor: on the table it is one table height plus a radius.
    r = ball_diameter_m / 2
    h = y - (level - table_height_m - r)
    ground = k + np.flatnonzero(h[k:] <= r + FLAT_TOL_M)
    d_observed = None
    if ground.size:
        j = int(ground[0])
        x_ground = x[j - 1] + (r - h[j - 1]) * (x[j] - x[j - 1]) / (h[j] - h[j - 1])
        d_observed = float(x_ground - x_edge)
    return {"clip": clip, "v_edge_ms": v_edge, "d_pred_m": d_pred, "d_observed_m": d_observed}


def cup(
    clips: list[str],
    table_height_m: float,
    *,
    prompt: str,
    transport: Transport,
    cup_height_m: float = 0.0,
    ball_diameter_m: float = 0.04,
    budget: Budget | None = None,
) -> dict:
    runs = []
    for clip in clips:
        tr = track_clip(clip, prompt, transport=transport, budget=budget)
        runs.append(_cup_run(clip, tr, table_height_m, cup_height_m, ball_diameter_m))
    d = [r["d_pred_m"] for r in runs]
    result = {
        "runs": runs,
        "d_pred_m": float(np.mean(d)),
        "spread_m": max(d) - min(d),
        "table_height_m": table_height_m,
        "cup_height_m": cup_height_m,
    }
    return _sealed(result, result["d_pred_m"])


def weigh(
    ref_clip: str,
    obj_clip: str,
    ref_mass_kg: float,
    *,
    prompt_ref: str,
    prompt_obj: str,
    transport: Transport,
    budget: Budget | None = None,
) -> dict:
    periods = []
    for clip, prompt in ((ref_clip, prompt_ref), (obj_clip, prompt_obj)):
        tr = track_clip(clip, prompt, transport=transport, budget=budget)
        periods.append(experiments.spring_period(tr.times[tr.indices], -tr.centers[:, 1]))
    mass = experiments.mass_from_periods(ref_mass_kg, periods[0], periods[1])
    result = {"period_ref_s": periods[0], "period_obj_s": periods[1], "ref_mass_kg": ref_mass_kg, "mass_kg": mass}
    return _sealed(result, mass)
