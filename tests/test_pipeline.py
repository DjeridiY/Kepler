import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

from kepler import measure
from kepler.agent import Setup
from kepler.budget import Budget, BudgetExceeded
from kepler.fit import fit_law, simulate
from kepler.laws import load_law
from kepler.pipeline import RunError, run
from kepler.synth import BOB_RADIUS_PX, render_fall, render_pendulum
from kepler.video import frame_times

PENDULUM_SETUP = '{"experiment":"pendulum","object_prompt":"red ball","signal":"angle","description":"%s"}'
FALL_SETUP = Setup("fall", "red ball", "height", "a ball dropped from rest")
NO_LAWS = '{"laws": []}'


class FakeLLM:
    def __init__(self, replies: list[str]):
        self.replies = replies
        self.calls = 0

    def complete(self, system: str, user: str, images: list[bytes]) -> tuple[str, int, int]:
        self.calls += 1
        return self.replies[min(self.calls, len(self.replies)) - 1], 1000, 200


def _box(cx: float, cy: float, *, frame: int = 0, obj: int = 1) -> str:
    r = BOB_RADIUS_PX
    x1, y1, x2, y2 = round(cx - r), round(cy - r), round(cx + r), round(cy + r)
    return f"<{frame}f>{obj}<|box;x1={x1};y1={y1};x2={x2};y2={y2};w=1280;h=720|>"


class FakeTransport:
    """Answers like SAM would on the synthetic clip: the bob, its track, and the A4 sheet."""

    def __init__(self, truth: dict, a4_prompt: str = "white sheet of paper"):
        self.truth = truth
        self.a4_prompt = a4_prompt
        self.calls: list[str] = []

    def segment_image(self, jpeg: bytes, prompt: str) -> str:
        self.calls.append("image")
        if prompt == self.a4_prompt:
            x1, y1, x2, y2 = self.truth["a4_box_px"]
            return f"<0f>0<|box;x1={x1};y1={y1};x2={x2};y2={y2};w=1280;h=720|>"
        x, y = self.truth["bob_px"][0]
        return _box(x, y, obj=0)

    def segment_video(self, path: str, prompt: str) -> tuple[str, int]:
        self.calls.append("video")
        bobs = self.truth["bob_px"]
        return "\n".join(_box(x, y, frame=i) for i, (x, y) in enumerate(bobs)), len(bobs)


@pytest.fixture(scope="module")
def pendulum(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pendulum")
    clip = str(tmp / "clip.mp4")
    truth = render_pendulum(clip, vfr=True, duration_s=8.0, length_m=0.45, gamma=0.03, theta0_deg=20.0)
    return clip, truth


def _run_pendulum(clip, truth, out, description="a red ball hangs on a string", **kwargs):
    llm = FakeLLM([PENDULUM_SETUP % description, NO_LAWS])
    return run(clip, str(out), transport=FakeTransport(truth), llm=llm, **kwargs)


@pytest.fixture(scope="module")
def pendulum_run(pendulum, tmp_path_factory):
    clip, truth = pendulum
    out = tmp_path_factory.mktemp("out")
    budget = Budget(str(out / "budget.json"))
    return _run_pendulum(clip, truth, out, budget=budget), out


def test_paid_sam_calls_are_charged(pendulum_run):
    _, out = pendulum_run
    labels = [e["label"] for e in json.loads((out / "budget.json").read_text())["entries"]]
    assert labels.count("sam_image") == 2
    assert labels.count("sam_video") == 1


def test_run_keeps_raw_series_for_the_demo(pendulum_run):
    result, out = pendulum_run
    track = np.load(out / "track.npz")
    keys = {"times", "indices", "centers", "y", "pred", "pred_px", "split_time", "px_per_m", "pivot_px", "length_px"}
    assert set(track.files) == keys
    n = len(track["times"])
    assert len(track["pred"]) == n
    assert track["pred_px"].shape == (n, 2)
    assert track["centers"].shape == (len(track["indices"]), 2)
    assert len(track["y"]) == len(track["indices"])
    assert float(track["split_time"]) == pytest.approx(result.split_time)
    assert float(track["px_per_m"]) == pytest.approx(result.px_per_m)
    assert track["pivot_px"].shape == (2,)
    assert np.isfinite(track["length_px"])


def test_pendulum_run_finds_law_period_and_length(pendulum, pendulum_run):
    clip, truth = pendulum
    result, out = pendulum_run
    true_period = measure.period(np.array(truth["times"]), np.array(truth["theta"])).period_s
    assert result.experiment == "pendulum"
    assert result.coverage == 1.0
    assert result.law == "damped pendulum (linear)"
    assert result.period_s == pytest.approx(true_period, rel=0.005)
    assert result.length_m == pytest.approx(0.45, rel=0.02)
    assert result.g_ms2 is not None
    assert len(frame_times(str(out / "kepler.mp4"))) == len(frame_times(clip))


def test_outputs_match_result_and_seal(pendulum_run):
    result, out = pendulum_run
    report = json.loads((out / "report.json").read_text())
    assert report == asdict(result)
    assert all(e["source"] for e in report["entries"])

    accepted = next(e for e in result.entries if e.status == "accepted")
    assert accepted.name == result.law
    seal_lines = (out / "seal.txt").read_text().splitlines()
    assert seal_lines[0] == f"sha256 {result.seal_sha256}"
    assert seal_lines[1].startswith("computed ")
    assert seal_lines[1].endswith(("Z", "+00:00"))

    rows = [line for line in (out / "notebook.md").read_text().splitlines() if line.startswith("|")]
    assert len(rows) == 2 + len(result.entries)
    assert "a red ball hangs on a string" in (out / "notebook.md").read_text()


def test_seal_is_hash_of_rounded_prediction(pendulum, pendulum_run):
    clip, truth = pendulum
    result, out = pendulum_run
    times = frame_times(clip)
    # Same centers as the fake SAM boxes, then the same fit and prediction as the run.
    r = BOB_RADIUS_PX
    centers = np.array(
        [((round(x - r) + round(x + r)) / 2, (round(y - r) + round(y + r)) / 2) for x, y in truth["bob_px"]]
    )
    theta, _, _ = measure.angle_series(centers * [1, -1])
    law = load_law(next(e.source for e in result.entries if e.status == "accepted"))
    fit = fit_law(law, times, theta)
    pred = simulate(law, fit.params, fit.y0, fit.v0, times)
    expected = hashlib.sha256(np.round(pred[times >= fit.split_time], 6).astype("<f8").tobytes()).hexdigest()
    assert (out / "seal.txt").read_text().splitlines()[0] == f"sha256 {expected}"


def test_wrong_number_in_description_is_withheld(pendulum, tmp_path):
    clip, truth = pendulum
    _run_pendulum(clip, truth, tmp_path, description="period about 2.5 s", px_per_m=truth["px_per_m"])
    notebook = (tmp_path / "notebook.md").read_text()
    assert "description withheld: unverified numbers" in notebook
    assert "2.5" not in notebook


def test_budget_too_small_stops_before_any_sam_call(pendulum, tmp_path):
    clip, truth = pendulum
    transport = FakeTransport(truth)
    budget = Budget(str(tmp_path / "budget.json"), cap_usd=0.001)
    llm = FakeLLM([PENDULUM_SETUP % "x", NO_LAWS])
    with pytest.raises(BudgetExceeded):
        run(clip, str(tmp_path / "out"), transport=transport, llm=llm, budget=budget)
    assert transport.calls == []


@pytest.fixture(scope="module")
def fall(tmp_path_factory):
    clip = str(tmp_path_factory.mktemp("fall") / "fall.mp4")
    truth = render_fall(clip, drag="quadratic")
    x, y = truth["bob_px"][0]
    box0 = (round(x) - 10, round(y) - 10, round(x) + 10, round(y) + 10)
    return clip, truth, box0


def test_fall_with_scale_finds_quadratic_drag(fall, tmp_path):
    clip, truth, box0 = fall
    llm = FakeLLM([NO_LAWS])
    result = run(
        clip, str(tmp_path), transport=None, llm=llm, tracker="color", box0=box0,
        px_per_m=truth["px_per_m"], setup=FALL_SETUP,
    )  # fmt: skip
    assert result.law == "fall with quadratic drag"
    assert result.period_s is None
    assert (tmp_path / "kepler.mp4").is_file()


def test_fall_without_scale_needs_a_scale(fall, tmp_path):
    clip, _, box0 = fall
    with pytest.raises(RunError, match="needs a scale"):
        run(clip, str(tmp_path), transport=None, llm=FakeLLM([NO_LAWS]), tracker="color", box0=box0, setup=FALL_SETUP)


def test_color_tracker_needs_a_box(fall, tmp_path):
    clip, _, _ = fall
    with pytest.raises(ValueError):
        run(clip, str(tmp_path), transport=None, llm=FakeLLM([NO_LAWS]), tracker="color", setup=FALL_SETUP)


def test_low_coverage_is_reported(pendulum, tmp_path):
    clip, truth = pendulum
    n = len(truth["bob_px"])
    half = {"bob_px": truth["bob_px"][: n // 3], "a4_box_px": truth["a4_box_px"]}
    llm = FakeLLM([PENDULUM_SETUP % "x", NO_LAWS])
    with pytest.raises(RunError, match=f"{n // 3}/{n}"):
        run(clip, str(tmp_path), transport=FakeTransport(half), llm=llm)
