import cv2
import numpy as np
import pytest

from kepler.colortrack import TrackError, color_track, sample_color
from kepler.synth import _encode, render_pendulum
from kepler.video import read_frames


@pytest.fixture(scope="module")
def pendulum(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("ct") / "pendulum.mp4")
    truth = render_pendulum(path, duration_s=3.0)
    return path, truth


def _box_around(point, size=36):
    x, y = round(point[0]), round(point[1])
    half = size // 2
    return (x - half, y - half, x + half - 1, y + half - 1)


def test_sample_color_red_wraps_hue():
    img = np.zeros((20, 20, 3), np.uint8)
    img[:, :10] = (0, 0, 255)  # hue 0
    img[:, 10:] = (10, 0, 255)  # hue ~179
    low, high = sample_color(img, (0, 0, 19, 19))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    assert low[0] > high[0]
    assert all(h >= low[0] or h <= high[0] for h in hsv[0, :, 0])
    assert low[1] == 195 and high[1] == 255


def test_tracks_pendulum_bob(pendulum):
    path, truth = pendulum
    bobs = np.array(truth["bob_px"])
    indices, centers = color_track(path, _box_around(bobs[0]))
    assert indices.tolist() == list(range(len(bobs)))
    assert np.max(np.linalg.norm(centers - bobs, axis=1)) <= 1.0


def test_ignores_static_red_distractor(pendulum, tmp_path):
    path, truth = pendulum
    bobs = np.array(truth["bob_px"])
    distractor = (1150, 150)
    frames = []
    for frame in read_frames(path):
        img = frame.image.copy()
        cv2.circle(img, distractor, 30, (0, 0, 255), cv2.FILLED)
        frames.append(img)
    copy = str(tmp_path / "distractor.mp4")
    _encode(copy, frames, np.array(truth["times"]), 60.0)
    indices, centers = color_track(copy, _box_around(bobs[0]))
    assert indices.tolist() == list(range(len(bobs)))
    assert np.max(np.linalg.norm(centers - bobs, axis=1)) <= 1.0
    assert np.min(np.linalg.norm(centers - np.array(distractor), axis=1)) > 100


def test_empty_white_box_loses_object(pendulum):
    path, _ = pendulum
    with pytest.raises(TrackError, match="object lost on"):
        color_track(path, (1180, 40, 1215, 75))
