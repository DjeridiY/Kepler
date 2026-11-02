"""One run from a clip to a sealed prediction: track, scale, find the law, predict, render, report."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from kepler import agent, colortrack, measure, render, sam, scale, video
from kepler.agent import LLM, Entry, Setup
from kepler.budget import Budget, sam_video_cost
from kepler.fit import FitResult, simulate
from kepler.laws import load_law
from kepler.sam import Transport

MIN_COVERAGE = 0.5


class RunError(Exception):
    pass


@dataclass(frozen=True)
class RunResult:
    experiment: str
    law: str
    params: dict[str, float]
    rel_err_test: float
    split_time: float
    coverage: float  # tracked frames / total frames
    px_per_m: float | None
    period_s: float | None  # pendulum only
    length_m: float | None  # pendulum with a scale only
    g_ms2: float | None  # pendulum with a scale only
    seal_sha256: str
    entries: list[Entry]


@dataclass(frozen=True)
class _Signal:
    t: np.ndarray
    y: np.ndarray
    pivot_px: tuple[float, float] | None  # angle only
    length_px: float | None  # angle only
    x_px: float | None  # height only: median tracked x


def _jpeg(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RunError("cannot encode frame as JPEG")
    return buf.tobytes()


def _frames(path: str, n: int, setup: Setup | None) -> dict[int, np.ndarray]:
    if setup is not None:
        return {0: next(video.read_frames(path)).image}
    wanted = {0, n // 2, n - 1}
    return {f.index: f.image for f in video.read_frames(path) if f.index in wanted}


def _track(
    path: str,
    tracker: str,
    *,
    transport: Transport | None,
    prompt: str,
    image0: np.ndarray,
    box0: tuple[int, int, int, int] | None,
    on_cost: Callable[[str, float], None] | None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        if tracker == "sam":
            if transport is None:
                raise ValueError("the sam tracker needs a transport")
            track = sam.track(transport, path, prompt, frame0_bgr=image0, on_cost=on_cost)
            return track.indices, track.centers
        if tracker == "color":
            if box0 is None:
                raise ValueError("the color tracker needs a start box")
            return colortrack.color_track(path, box0)
    except (sam.SamError, colortrack.TrackError) as e:
        raise RunError(str(e)) from e
    raise ValueError(f"unknown tracker: {tracker!r}")


def _scale(
    transport: Transport | None, image0: np.ndarray, prompt: str, on_cost: Callable[[str, float], None] | None
) -> float | None:
    if transport is None:
        return None
    boxes = sam.detect(transport, image0, prompt)
    if on_cost:
        on_cost("sam_image", sam.IMAGE_COST)
    if not boxes:
        return None
    b = boxes[0]
    try:
        return scale.px_per_meter_from_box((b.x1, b.y1, b.x2, b.y2))
    except scale.ScaleError:
        return None


def track_object(
    path: str,
    times: np.ndarray,
    image0: np.ndarray,
    prompt: str,
    *,
    transport: Transport | None,
    budget: Budget | None = None,
    tracker: str = "sam",
    box0: tuple[int, int, int, int] | None = None,
    px_per_m: float | None = None,
    a4_prompt: str = "white sheet of paper",
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """Tracked frame indices, pixel centers and px/m (given, else from the A4 sheet, else None)."""
    on_cost = budget.charge if budget else None
    if budget and tracker == "sam":
        budget.reserve("sam", sam.IMAGE_COST * 2 + sam_video_cost(len(times)))
    indices, centers = _track(
        path, tracker, transport=transport, prompt=prompt, image0=image0, box0=box0, on_cost=on_cost
    )
    if len(indices) < MIN_COVERAGE * len(times):
        raise RunError(f"object tracked on {len(indices)}/{len(times)} frames")
    if px_per_m is None:
        px_per_m = _scale(transport, image0, a4_prompt, on_cost)
    return indices, centers, px_per_m


def _signal(setup: Setup, t: np.ndarray, centers: np.ndarray, px_per_m: float | None) -> _Signal:
    if setup.signal == "angle":
        theta, pivot_up, length_px = measure.angle_series(centers * [1, -1])
        return _Signal(t, theta, (float(pivot_up[0]), -float(pivot_up[1])), length_px, None)
    if px_per_m is None:
        raise RunError("height needs a scale: show the A4 sheet")
    return _Signal(t, -centers[:, 1] / px_per_m, None, None, float(np.median(centers[:, 0])))


def _predict(source: str, fit: FitResult, times: np.ndarray, t0: float) -> np.ndarray:
    after = times >= t0
    pred = simulate(load_law(source), fit.params, fit.y0, fit.v0, times[after])
    # Frames before the first tracked one hold the first predicted value.
    return np.concatenate((np.full(int(np.sum(~after)), pred[0]), pred))


def _measures(sig: _Signal, px_per_m: float | None) -> tuple[float | None, float | None, float | None]:
    if sig.length_px is None:
        return None, None, None
    period_s = measure.period(sig.t, sig.y).period_s
    if px_per_m is None:
        return period_s, None, None
    length_m = sig.length_px / px_per_m
    return period_s, length_m, measure.g_from_pendulum(length_m, period_s, float(np.max(np.abs(sig.y))))


def _labels(r: RunResult) -> list[str]:
    lines = [f"law: {r.law}", f"test error: {r.rel_err_test * 100:.2f} %"]
    if r.period_s is not None:
        lines.append(f"period: {r.period_s:.3f} s")
    if r.length_m is not None:
        lines.append(f"length: {r.length_m:.3f} m")
    if r.g_ms2 is not None:
        lines.append(f"g: {r.g_ms2:.2f} m/s^2")
    return lines


def _numbers(r: RunResult) -> list[float]:
    values = [r.rel_err_test, r.split_time, r.coverage, r.px_per_m, r.period_s, r.length_m, r.g_ms2]
    return [v for v in [*values, *r.params.values()] if v is not None]


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _notebook(r: RunResult, description: str) -> str:
    lines = ["| round | name | status | error % | reason |", "|---|---|---|---|---|"]
    for e in r.entries:
        err = "" if e.rel_err_test is None else f"{e.rel_err_test * 100:.2f}"
        lines.append(f"| {e.round} | {_cell(e.name)} | {e.status} | {err} | {_cell(e.reason)} |")
    verified = not agent.check_numbers(description, _numbers(r))
    lines += ["", description if verified else "description withheld: unverified numbers"]
    return "\n".join(lines) + "\n"


def _pixels(sig: _Signal, pred: np.ndarray, px_per_m: float | None) -> np.ndarray:
    if sig.pivot_px is not None and sig.length_px is not None:
        return render.pendulum_to_px(pred, sig.pivot_px, sig.length_px)
    assert px_per_m is not None and sig.x_px is not None
    return np.column_stack((np.full(pred.size, sig.x_px), -pred * px_per_m))


def run(
    path: str,
    out_dir: str,
    *,
    transport: Transport | None,
    llm: LLM,
    budget: Budget | None = None,
    tracker: str = "sam",
    box0: tuple[int, int, int, int] | None = None,
    px_per_m: float | None = None,
    setup: Setup | None = None,
    a4_prompt: str = "white sheet of paper",
) -> RunResult:
    times = video.frame_times(path)
    frames = _frames(path, len(times), setup)
    image0 = frames[0]
    if setup is None:
        setup = agent.describe_setup(llm, [_jpeg(frames[k]) for k in sorted(frames)])
    indices, centers, px_per_m = track_object(
        path, times, image0, setup.object_prompt, transport=transport, budget=budget,
        tracker=tracker, box0=box0, px_per_m=px_per_m, a4_prompt=a4_prompt,
    )  # fmt: skip
    sig = _signal(setup, times[indices], centers, px_per_m)

    on_cost = budget.charge if budget else None
    entries, fit = agent.run_lab(llm, sig.t, sig.y, setup, on_cost=on_cost)
    source = next(e.source for e in entries if e.status == "accepted")
    pred = _predict(source, fit, times, float(sig.t[0]))
    period_s, length_m, g_ms2 = _measures(sig, px_per_m)
    seal = hashlib.sha256(np.round(pred[times >= fit.split_time], 6).astype("<f8").tobytes()).hexdigest()
    result = RunResult(
        experiment=setup.experiment,
        law=fit.name,
        params=dict(fit.params),
        rel_err_test=float(fit.rel_err_test),
        split_time=fit.split_time,
        coverage=len(indices) / len(times),
        px_per_m=px_per_m,
        period_s=period_s,
        length_m=length_m,
        g_ms2=g_ms2,
        seal_sha256=seal,
        entries=entries,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    band_px = fit.rmse_test * (sig.length_px if sig.length_px is not None else px_per_m or 0.0)
    pred_px = _pixels(sig, pred, px_per_m)
    np.savez(
        out / "track.npz", times=times, indices=indices, centers=centers, y=sig.y, pred=pred, pred_px=pred_px,
        split_time=fit.split_time, px_per_m=np.nan if px_per_m is None else px_per_m,
        pivot_px=np.array(sig.pivot_px if sig.pivot_px is not None else (np.nan, np.nan)),
        length_px=np.nan if sig.length_px is None else sig.length_px,
    )  # fmt: skip
    overlay = render.Overlay(times, pred_px, band_px, fit.split_time, _labels(result))
    render.render_overlay(path, str(out / "kepler.mp4"), overlay)
    (out / "report.json").write_text(json.dumps(asdict(result), indent=2))
    (out / "notebook.md").write_text(_notebook(result, setup.description))
    (out / "seal.txt").write_text(f"sha256 {seal}\ncomputed {datetime.now(UTC).isoformat()}\n")
    return result
