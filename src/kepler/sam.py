"""SAM 3.1 tracking: detect the object on a still frame 0, then track it in the video with a box prompt.

A text prompt on video does not always acquire the object, while the same prompt on a still image does;
a box prompt on frame 0 then tracks reliably.
"""

import base64
import re
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import cv2
import numpy as np
from openai import OpenAI

from kepler import video

IMAGE_COST = 0.0025
VIDEO_FRAME_COST = 0.0002
VIDEO_MIN_FRAMES = 50

_LINE = re.compile(r"<(\d+)f>(.*)")
# One object: id, its box tag, then any further tags (mask...); objects are separated by top-level commas.
_OBJECT = re.compile(r"(\d+)<\|box;([^|]*)\|>(?:<\|.*?\|>)*(?:,|$)")


class SamError(Exception):
    pass


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True)
class Track:
    indices: np.ndarray
    centers: np.ndarray
    boxes: list[Box]
    n_frames: int


class Transport(Protocol):
    def segment_image(self, jpeg: bytes, prompt: str) -> str: ...

    def segment_video(self, path: str, prompt: str) -> tuple[str, int]: ...


def _parse_line(line: str) -> tuple[int, dict[int, Box]]:
    head = _LINE.fullmatch(line)
    if head is None:
        raise SamError(f"bad record: {line[:80]}")
    rest, objects, pos = head.group(2), {}, 0
    while pos < len(rest):
        m = _OBJECT.match(rest, pos)
        if m is None:
            raise SamError(f"bad record: {line[:80]}")
        fields = dict(kv.split("=", 1) for kv in m.group(2).split(";"))
        objects[int(m.group(1))] = Box(*(int(fields[k]) for k in ("x1", "y1", "x2", "y2")))
        pos = m.end()
    return int(head.group(1)), objects


def parse_records(text: str) -> dict[int, dict[int, Box]]:
    records: dict[int, dict[int, Box]] = {}
    for line in map(str.strip, text.splitlines()):
        if line:
            frame, objects = _parse_line(line)
            records.setdefault(frame, {}).update(objects)
    return records


def box_prompt(box: Box, width: int, height: int, frame: int = 0, obj_id: int = 1) -> str:
    return f"<{frame}f>{obj_id}<|box;x1={box.x1};y1={box.y1};x2={box.x2};y2={box.y2};w={width};h={height}|>"


def detect(transport: Transport, image_bgr: np.ndarray, prompt: str) -> list[Box]:
    ok, jpeg = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise SamError("cannot encode frame as JPEG")
    boxes = parse_records(transport.segment_image(jpeg.tobytes(), prompt)).get(0, {})
    return sorted(boxes.values(), key=lambda b: b.area, reverse=True)


def constant_rate_copy(path: str, dst: str, fps: int = 60) -> int:
    """Re-encode `path` into `dst`, each decoded frame becoming one frame timed k / fps; return the frame count."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-map", "0:v:0", "-vf", f"setpts=N/({fps}*TB)"]
    cmd += ["-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "10", dst]
    subprocess.run(cmd, check=True)
    copied, source = len(video.frame_times(dst)), len(video.frame_times(path))
    if copied != source:
        raise SamError(f"constant-rate copy has {copied} frames, source has {source}")
    return copied


def track(
    transport: Transport,
    path: str,
    prompt: str,
    *,
    frame0_bgr: np.ndarray,
    on_cost: Callable[[str, float], None] | None = None,
) -> Track:
    boxes = detect(transport, frame0_bgr, prompt)
    if on_cost:
        on_cost("sam_image", IMAGE_COST)
    if not boxes:
        raise SamError(f"object not found on frame 0: {prompt}")
    height, width = frame0_bgr.shape[:2]
    # On a variable-rate upload SAM drops frames and its indices drift from the decoded ones.
    with tempfile.TemporaryDirectory() as tmp:
        copy = str(Path(tmp) / "cfr.mp4")
        constant_rate_copy(path, copy)
        text, n_frames = transport.segment_video(copy, box_prompt(boxes[0], width, height))
    if on_cost:
        on_cost("sam_video", max(n_frames, VIDEO_MIN_FRAMES) * VIDEO_FRAME_COST)
    found = sorted((i, objs[1]) for i, objs in parse_records(text).items() if 1 in objs)
    return Track(
        indices=np.array([i for i, _ in found], dtype=int),
        centers=np.array([b.center for _, b in found], dtype=float).reshape(-1, 2),
        boxes=[b for _, b in found],
        n_frames=n_frames,
    )


class MetaTransport:
    def __init__(self, api_key: str, base_url: str = "https://api.meta.ai/v1") -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def _run(self, prompt: str, media: dict[str, Any]) -> tuple[str, int]:
        stream = self._client.responses.create(
            model="sam-3.1",
            stream=True,
            metadata={"mask_encoding": "one_bit", "propagation_direction": "forward"},
            input=cast(Any, [{"role": "user", "content": [{"type": "input_text", "text": prompt}, media]}]),
        )
        parts: list[str] = []
        n_frames = 0
        for event in stream:
            if event.type == "response.output_text.delta":
                parts.append(event.delta)
            elif event.type in ("response.failed", "error"):
                raise SamError(f"SAM request failed: {event}")
            elif event.type == "response.completed":
                n_frames = int(getattr(event.response.usage, "video_frames_processed", 0) or 0)
        return "".join(parts), n_frames

    def segment_image(self, jpeg: bytes, prompt: str) -> str:
        url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
        return self._run(prompt, {"type": "input_image", "image_url": url})[0]

    def segment_video(self, path: str, prompt: str) -> tuple[str, int]:
        with open(path, "rb") as f:
            uploaded = self._client.files.create(file=f, purpose="user_data")
        return self._run(prompt, {"type": "input_video", "file_id": uploaded.id})
