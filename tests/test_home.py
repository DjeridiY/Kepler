import hashlib
import math

import cv2
import numpy as np
import pytest

from kepler import measure, synth
from kepler.experiments import landing_distance
from kepler.home import bet, cup, seal, weigh
from kepler.pipeline import RunError
from kepler.synth import render_fall, render_pendulum

G = 9.81
FPS = 60.0


def _box(cx: float, cy: float, r: int, frame: int, obj: int) -> str:
    x1, y1, x2, y2 = round(cx - r), round(cy - r), round(cx + r), round(cy + r)
    return f"<{frame}f>{obj}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w=1280;h=720|>"


class FakeTransport:
    """Answers like SAM would from the ground truth of each clip: the object, its track, the A4 sheet."""

    def __init__(self, clips: list[str], truths: dict[str, dict], radius: int = synth.BOB_RADIUS_PX):
        self.order = list(clips)  # a still image carries no path: clips are tracked in this order
        self.truths = truths
        self.radius = radius
        self.current: dict = {}

    def segment_image(self, jpeg: bytes, prompt: str) -> str:
        if prompt == "white sheet of paper":
            if self.current["a4_box_px"] is None:
                return ""
            x1, y1, x2, y2 = self.current["a4_box_px"]
            return f"<0f>0<|box;x1={x1};y1={y1};x2={x2};y2={y2};w=1280;h=720|>"
        self.current = self.truths[self.order.pop(0)]
        x, y = self.current["bob_px"][0]
        return _box(x, y, self.radius, 0, 0)

    def segment_video(self, path: str, prompt: str) -> tuple[str, int]:
        # SAM receives a constant-rate copy, not the clip itself: answer for the clip whose frame 0 was just detected
        bobs = self.current["bob_px"]
        return "\n".join(_box(x, y, self.radius, i, 1) for i, (x, y) in enumerate(bobs)), len(bobs)


def _render(path: str, times: np.ndarray, centers: list[tuple[float, float]], radius: int, a4: list[int]) -> None:
    background = synth._background((1280, 720), a4)
    frames = []
    for c in centers:
        img = background.copy()
        cv2.circle(img, synth._fixed(c), radius << synth._SHIFT, (0, 0, 255), cv2.FILLED, cv2.LINE_8, synth._SHIFT)
        frames.append(img)
    synth._encode(path, frames, times, FPS)


# --- bet: the shortened pendulum ---


def test_bet_predicts_the_short_pendulum_period(tmp_path):
    long_clip, short_clip = str(tmp_path / "long.mp4"), str(tmp_path / "short.mp4")
    truth = render_pendulum(long_clip, length_m=0.45, theta0_deg=25, vfr=True, duration_s=10)
    short = render_pendulum(short_clip, length_m=0.30, theta0_deg=25, vfr=True, duration_s=10)
    result = bet(long_clip, 0.45, 0.30, prompt="pendulum bob", transport=FakeTransport([long_clip], {long_clip: truth}))
    true_b = measure.period(np.array(short["times"]), np.array(short["theta"])).period_s
    assert result["period_b_pred_s"] == pytest.approx(true_b, rel=0.005)
    assert (result["length_a_m"], result["length_b_m"]) == (0.45, 0.30)
    assert result["seal_sha256"] == seal([result["period_b_pred_s"]])[0]


# --- cup: a ball rolling off a table ---

PX_PER_M = 600.0
FLOOR_ROW = 690.0  # pixel row of the floor
LEFT_COL = 40.0  # pixel column of x = 0
TABLE_M = 0.75
BALL_M = 0.04
EDGE_M = 0.8
DRAG_B = 0.15
V_EDGE = 1.2


def _rolling_ball(x0: float, phase: float) -> dict:
    """Ball center (x right, y up, metres above the floor) rolling with drag, then falling and stopping."""
    r = BALL_M / 2
    v0 = V_EDGE + DRAG_B * (EDGE_M - x0)
    t_edge = -math.log1p(-DRAG_B * (EDGE_M - x0) / v0) / DRAG_B
    t_land = t_edge + math.sqrt(2 * TABLE_M / G)
    times = np.arange(math.floor((t_land + 0.3) * FPS)) / FPS
    xs, ys = [], []
    for t in times + phase:
        if t < t_edge:
            xs.append(x0 + v0 * (1 - math.exp(-DRAG_B * t)) / DRAG_B)
            ys.append(TABLE_M + r)
        elif t < t_land:
            xs.append(EDGE_M + V_EDGE * (t - t_edge))
            ys.append(TABLE_M + r - G * (t - t_edge) ** 2 / 2)
        else:
            xs.append(EDGE_M + V_EDGE * (t_land - t_edge))
            ys.append(r)
    x, y = np.array(xs), np.array(ys)
    # The edge as the code sees it: first frame more than 4 mm below the table level.
    seen_edge = x[np.flatnonzero(y < TABLE_M + r - 0.004)[0]]
    return {
        "times": times,
        "bob_px": [(LEFT_COL + a * PX_PER_M, FLOOR_ROW - b * PX_PER_M) for a, b in zip(x, y, strict=True)],
        "a4_box_px": [1060, 40, 1060 + round(0.297 * PX_PER_M), 40 + round(0.210 * PX_PER_M)],
        "d_observed_m": EDGE_M + V_EDGE * (t_land - t_edge) - seen_edge,
    }


def _cup_clips(tmp_path, starts: list[tuple[float, float]]) -> tuple[list[str], dict[str, dict]]:
    radius = round(BALL_M / 2 * PX_PER_M)
    clips, truths = [], {}
    for k, (x0, phase) in enumerate(starts):
        clip = str(tmp_path / f"roll{k}.mp4")
        truth = _rolling_ball(x0, phase)
        _render(clip, truth["times"], truth["bob_px"], radius, truth["a4_box_px"])
        clips.append(clip)
        truths[clip] = truth
    return clips, truths


def test_cup_predicts_the_landing_spot_over_three_runs(tmp_path):
    clips, truths = _cup_clips(tmp_path, [(0.10, 0.0), (0.14, 0.004), (0.18, 0.011)])
    transport = FakeTransport(clips, truths, radius=round(BALL_M / 2 * PX_PER_M))
    result = cup(clips, TABLE_M, prompt="ball", transport=transport)
    truth_d = landing_distance(V_EDGE, TABLE_M + BALL_M / 2)
    assert abs(result["d_pred_m"] - truth_d) < 0.01
    assert result["spread_m"] < 0.01
    assert len(result["runs"]) == 3
    for run in result["runs"]:
        assert abs(run["d_pred_m"] - truth_d) < 0.01
        assert abs(run["d_observed_m"] - truths[run["clip"]]["d_observed_m"]) < 0.003
    assert result["seal_sha256"] == seal([result["d_pred_m"]])[0]
    assert (result["table_height_m"], result["cup_height_m"]) == (TABLE_M, 0.0)


def test_cup_needs_a_ball_rolling_on_the_table(tmp_path):
    clip = str(tmp_path / "fall.mp4")
    truth = render_fall(clip, size=(1280, 720), height_m=0.6, duration_s=0.3)
    with pytest.raises(RunError, match="no rolling on the table"):
        cup([clip], TABLE_M, prompt="ball", transport=FakeTransport([clip], {clip: truth}))


def test_cup_needs_a_scale(tmp_path):
    clips, truths = _cup_clips(tmp_path, [(0.10, 0.0)])
    truths[clips[0]]["a4_box_px"] = None
    with pytest.raises(RunError, match="cup needs a scale"):
        cup(clips, TABLE_M, prompt="ball", transport=FakeTransport(clips, truths))


# --- weigh: two masses on the same spring ---


def _spring(path: str, mass: float, k: float = 35.0, gamma: float = 0.25) -> dict:
    times = np.arange(240) / FPS
    wd = math.sqrt(k / mass - gamma**2)
    rows = 360 + 40 * np.exp(-gamma * times) * np.cos(wd * times)
    truth = {"times": times, "bob_px": [(640.0, float(y)) for y in rows], "a4_box_px": [40, 40, 278, 208]}
    _render(path, times, truth["bob_px"], synth.BOB_RADIUS_PX, truth["a4_box_px"])
    return truth


def test_weigh_finds_the_mass_from_two_periods(tmp_path):
    ref, obj = str(tmp_path / "ref.mp4"), str(tmp_path / "obj.mp4")
    truths = {ref: _spring(ref, 0.5), obj: _spring(obj, 0.182)}
    transport = FakeTransport([ref, obj], truths)
    result = weigh(ref, obj, 0.5, prompt_ref="bottle", prompt_obj="hanging object", transport=transport)
    assert result["mass_kg"] == pytest.approx(0.182, rel=0.02)
    assert result["period_obj_s"] < result["period_ref_s"]
    assert result["seal_sha256"] == seal([result["mass_kg"]])[0]


# --- the seal ---


def test_seal_is_sha256_of_rounded_values_with_utc_time():
    sha, computed = seal([1.108])
    assert sha == hashlib.sha256(np.array([1.108], dtype="<f8").tobytes()).hexdigest()
    assert seal([1.1080000004])[0] == sha
    assert computed.endswith("+00:00")
