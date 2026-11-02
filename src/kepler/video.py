"""Per-frame timestamps and frame reading for phone videos (often variable frame rate)."""

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class VideoError(Exception):
    pass


@dataclass(frozen=True)
class Frame:
    index: int
    t: float  # seconds from the first frame
    image: np.ndarray  # BGR uint8


def _probe(path: str, entries: str) -> dict[str, Any]:
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entries, "-of", "json", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    info = json.loads(out)
    if not info.get("streams"):
        raise VideoError("no video stream")
    return info


def frame_times(path: str) -> np.ndarray:
    """Timestamp of each decoded frame of the first video stream, in seconds, the first being 0.0."""
    info = _probe(path, "stream=index:frame=best_effort_timestamp_time,pts_time")
    times = np.array(
        [float(f.get("best_effort_timestamp_time", f.get("pts_time"))) for f in info.get("frames", [])],
        dtype=np.float64,
    )
    if len(times) == 0:
        raise VideoError("no decodable frames")
    if np.any(np.diff(times) <= 0):
        raise VideoError("non-increasing timestamps")
    return times - times[0]


def nominal_fps(path: str) -> float:
    return float(Fraction(_probe(path, "stream=avg_frame_rate")["streams"][0]["avg_frame_rate"]))


def is_vfr(path: str, tol: float = 0.01) -> bool:
    dt = np.diff(frame_times(path))
    median = np.median(dt)
    return bool(np.max(np.abs(dt - median)) / median > tol)


def read_frames(path: str, step: int = 1) -> Iterator[Frame]:
    """Yield every `step`-th frame in decoding order, timed by its own timestamp."""
    times = frame_times(path)
    cap = cv2.VideoCapture(path)
    count = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            if count < len(times) and count % step == 0:
                yield Frame(count, float(times[count]), image)
            count += 1
    finally:
        cap.release()
    if count != len(times):
        raise VideoError(f"frame count mismatch: {count} vs {len(times)}")
