"""Synthetic experiment videos (pendulum, falling ball) with exact known ground truth."""

import math
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
from scipy.integrate import solve_ivp

A4_LONG_M = 0.297
A4_SHORT_M = 0.210

BOB_RADIUS_PX = 18
_RED = (0, 0, 255)
_BLACK = (0, 0, 0)
_GRAY = (200, 200, 200)
_SHIFT = 4  # cv2 fixed-point bits, so the bob is drawn at sub-pixel positions


def frame_times(fps: float, duration_s: float, vfr: bool, seed: int) -> np.ndarray:
    """Timestamps of every frame with t < duration_s, starting at 0."""
    if not vfr:
        n = math.ceil(duration_s * fps) + 1
        times = np.arange(n) / fps
    else:
        n = math.ceil(duration_s * fps / 0.75) + 2
        steps = (1 / fps) * (1 + np.random.default_rng(seed).uniform(-0.25, 0.25, n))
        times = np.concatenate(([0.0], np.cumsum(steps)))
    return times[times < duration_s]


def _integrate(rhs: Callable, y0: list[float], times: np.ndarray) -> np.ndarray:
    sol = solve_ivp(rhs, (0.0, float(times[-1])), y0, t_eval=times, rtol=1e-10, atol=1e-12)
    return sol.y


def _a4_box(px_per_m: float, top_left: tuple[int, int]) -> list[int]:
    x1, y1 = top_left
    return [x1, y1, x1 + round(A4_LONG_M * px_per_m), y1 + round(A4_SHORT_M * px_per_m)]


def _background(size: tuple[int, int], box: list[int]) -> np.ndarray:
    img = np.full((size[1], size[0], 3), 255, np.uint8)
    corners = (box[0], box[1]), (box[2] - 1, box[3] - 1)
    cv2.rectangle(img, *corners, _GRAY, cv2.FILLED)
    cv2.rectangle(img, *corners, _BLACK, 1)
    return img


def _fixed(point: tuple[float, float]) -> tuple[int, int]:
    return round(point[0] * (1 << _SHIFT)), round(point[1] * (1 << _SHIFT))


def _encode(path: str, frames: list[np.ndarray], times: np.ndarray, fps: float) -> None:
    durations = [*np.diff(times), 1 / fps]
    with tempfile.TemporaryDirectory() as tmp:
        lines = ["ffconcat version 1.0"]
        for k, (img, d) in enumerate(zip(frames, durations, strict=True)):
            name = Path(tmp) / f"{k:06d}.bmp"
            cv2.imwrite(str(name), img)
            # Without the framerate option, ffmpeg 8 rounds each image duration to 1/25 s.
            lines += [f"file '{name}'", "option framerate 90000", f"duration {d:.9f}"]
        listing = Path(tmp) / "list.ffconcat"
        listing.write_text("\n".join(lines) + "\n")
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(listing)]
        cmd += ["-fps_mode", "passthrough", "-bf", "0", "-c:v", "libx264", "-pix_fmt", "yuv420p"]
        cmd += ["-crf", "18", "-video_track_timescale", "90000", path]
        subprocess.run(cmd, check=True)


def render_pendulum(
    path: str,
    *,
    length_m: float = 0.45,
    theta0_deg: float = 8.0,
    gamma: float = 0.0,
    g: float = 9.81,
    px_per_m: float = 800.0,
    fps: float = 60.0,
    duration_s: float = 10.0,
    size: tuple[int, int] = (1280, 720),
    vfr: bool = False,
    seed: int = 0,
) -> dict:
    """Render a damped pendulum to an MP4 and return its ground truth."""
    times = frame_times(fps, duration_s, vfr, seed)
    theta = _integrate(
        lambda _t, s: [s[1], -(g / length_m) * math.sin(s[0]) - 2 * gamma * s[1]],
        [math.radians(theta0_deg), 0.0],
        times,
    )[0]
    pivot = (size[0] // 2, 60)
    r = length_m * px_per_m
    bobs = [(pivot[0] + r * math.sin(a), pivot[1] + r * math.cos(a)) for a in theta]
    box = _a4_box(px_per_m, (40, size[1] - 40 - round(A4_SHORT_M * px_per_m)))
    background = _background(size, box)
    frames = []
    for bob in bobs:
        img = background.copy()
        cv2.line(img, _fixed(pivot), _fixed(bob), _BLACK, 2, cv2.LINE_8, _SHIFT)
        cv2.circle(img, _fixed(bob), BOB_RADIUS_PX << _SHIFT, _RED, cv2.FILLED, cv2.LINE_8, _SHIFT)
        frames.append(img)
    _encode(path, frames, times, fps)
    return {
        "times": times.tolist(),
        "bob_px": [list(b) for b in bobs],
        "a4_box_px": box,
        "px_per_m": px_per_m,
        "theta": theta.tolist(),
        "pivot_px": list(pivot),
        "length_m": length_m,
        "period_small_angle_s": 2 * math.pi * math.sqrt(length_m / g),
    }


def render_fall(
    path: str,
    *,
    drag: str = "quadratic",
    coeff: float = 0.8,
    height_m: float = 1.5,
    g: float = 9.81,
    px_per_m: float = 400.0,
    fps: float = 60.0,
    duration_s: float = 0.8,
    size: tuple[int, int] = (720, 1280),
    vfr: bool = False,
    seed: int = 0,
) -> dict:
    """Render a ball dropped from rest (y up, metres) to an MP4 and return its ground truth."""
    drag_accel = {"none": lambda v: 0.0, "linear": lambda v: coeff * v, "quadratic": lambda v: coeff * v * abs(v)}
    if drag not in drag_accel:
        raise ValueError(f"unknown drag model: {drag!r}")
    times = frame_times(fps, duration_s, vfr, seed)
    y_m = _integrate(lambda _t, s: [s[1], -g - drag_accel[drag](s[1])], [height_m, 0.0], times)[0]
    bobs = [(float(size[0] // 2), size[1] - 40 - y * px_per_m) for y in y_m]
    box = _a4_box(px_per_m, (40, 40))
    background = _background(size, box)
    frames = []
    for bob in bobs:
        img = background.copy()
        cv2.circle(img, _fixed(bob), BOB_RADIUS_PX << _SHIFT, _RED, cv2.FILLED, cv2.LINE_8, _SHIFT)
        frames.append(img)
    _encode(path, frames, times, fps)
    return {
        "times": times.tolist(),
        "bob_px": [list(b) for b in bobs],
        "a4_box_px": box,
        "px_per_m": px_per_m,
        "y_m": y_m.tolist(),
        "drag": drag,
        "coeff": coeff,
    }
