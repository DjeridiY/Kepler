import subprocess
from pathlib import Path

import numpy as np
import pytest

from kepler.video import VideoError, frame_times, is_vfr, nominal_fps, read_frames


def _ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True)


@pytest.fixture(scope="module")
def cfr_clip(tmp_path_factory: pytest.TempPathFactory) -> str:
    path = tmp_path_factory.mktemp("cfr") / "cfr.mp4"
    _ffmpeg("-f", "lavfi", "-i", "testsrc=size=320x240:rate=30", "-t", "2", "-c:v", "libx264", str(path))
    return str(path)


def test_cfr_clip_times_are_k_over_30(cfr_clip: str) -> None:
    times = frame_times(cfr_clip)
    assert times.dtype == np.float64
    assert len(times) == 60
    np.testing.assert_allclose(times, np.arange(60) / 30, rtol=0, atol=1e-6)
    assert not is_vfr(cfr_clip)
    assert nominal_fps(cfr_clip) == 30.0


def test_vfr_clip_times_follow_frame_durations(tmp_path: Path) -> None:
    durations = [1 / 30, 1 / 20, 1 / 40] * 5
    lines = ["ffconcat version 1.0"]
    for i, d in enumerate(durations):
        png = tmp_path / f"f{i}.png"
        _ffmpeg("-f", "lavfi", "-i", f"color=c=0x{i * 16:02x}0000:size=64x48", "-frames:v", "1", str(png))
        lines += [f"file '{png.name}'", "option framerate 90000", f"duration {d:.9f}"]
    concat = tmp_path / "list.ffconcat"
    concat.write_text("\n".join(lines) + "\n")
    clip = tmp_path / "vfr.mp4"
    _ffmpeg(
        "-f", "concat", "-safe", "0", "-i", str(concat), "-fps_mode", "passthrough", "-bf", "0",
        "-video_track_timescale", "90000", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip),
    )  # fmt: skip

    times = frame_times(str(clip))
    expected = np.concatenate([[0.0], np.cumsum(durations)[:-1]])
    np.testing.assert_allclose(times, expected, rtol=0, atol=1e-3)
    assert is_vfr(str(clip))


def test_read_frames_with_step(cfr_clip: str) -> None:
    frames = list(read_frames(cfr_clip, step=10))
    assert [f.index for f in frames] == [0, 10, 20, 30, 40, 50]
    assert all(f.image.shape == (240, 320, 3) for f in frames)
    np.testing.assert_allclose([f.t for f in frames], np.arange(0, 60, 10) / 30, atol=1e-6)


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    missing = str(tmp_path / "nope.mp4")
    for call in (frame_times, nominal_fps, is_vfr, lambda p: next(read_frames(p))):
        with pytest.raises(FileNotFoundError):
            call(missing)


def test_audio_only_file_raises_no_video_stream(tmp_path: Path) -> None:
    audio = tmp_path / "audio.m4a"
    _ffmpeg("-f", "lavfi", "-i", "anullsrc", "-t", "1", str(audio))
    with pytest.raises(VideoError, match="no video stream"):
        frame_times(str(audio))
    with pytest.raises(VideoError, match="no video stream"):
        nominal_fps(str(audio))
