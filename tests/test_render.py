import cv2
import numpy as np
import pytest

from kepler.render import Overlay, pendulum_to_px, render_overlay
from kepler.synth import render_pendulum
from kepler.video import frame_times, is_vfr


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("render")
    src, dst = str(tmp / "src.mp4"), str(tmp / "dst.mp4")
    truth = render_pendulum(src, vfr=True, duration_s=2.0)
    times = frame_times(src)
    overlay = Overlay(times, np.array(truth["bob_px"]), 4.0, float(times[len(times) // 2]), ["model: pendulum"])
    render_overlay(src, dst, overlay)
    return src, dst, overlay


def test_output_keeps_source_timestamps(rendered):
    src, dst, _ = rendered
    t_src, t_dst = frame_times(src), frame_times(dst)
    assert len(t_dst) == len(t_src)
    assert np.max(np.abs(t_dst - t_src)) < 1e-3
    assert is_vfr(dst)


def test_ghost_color_switches_at_split_time(rendered):
    _, dst, overlay = rendered
    cap = cv2.VideoCapture(dst)
    for t, (x, y) in zip(overlay.times, overlay.predicted_px, strict=True):
        ok, img = cap.read()
        assert ok
        b, g, r = (int(c) for c in img[round(y), round(x + 18)])
        if t < overlay.split_time:
            assert b > 150 and g > 150 and r < 100, (t, b, g, r)
        else:
            assert b > 150 and r > 150 and g < 100, (t, b, g, r)
    cap.release()


def test_pendulum_to_px():
    px = pendulum_to_px(np.array([0.0, np.pi / 2]), (100, 50), 200)
    np.testing.assert_allclose(px, [[100, 250], [300, 50]], atol=1e-9)


def test_wrong_number_of_predictions_raises(rendered, tmp_path):
    src, _, overlay = rendered
    extra = np.vstack([overlay.predicted_px, overlay.predicted_px[-1]])
    bad = Overlay(overlay.times, extra, overlay.band_px, overlay.split_time, overlay.labels)
    with pytest.raises(ValueError):
        render_overlay(src, str(tmp_path / "bad.mp4"), bad)
