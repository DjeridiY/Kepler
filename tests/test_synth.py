import subprocess

import cv2
import numpy as np
import pytest

from kepler.synth import BOB_RADIUS_PX, render_fall, render_pendulum


def probe_times(path) -> np.ndarray:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time", "-of", "csv=p=0", path],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    t = np.array([float(line.strip(",")) for line in out.split()])
    return t - t[0]


def red_centers(path) -> list[tuple[float, float] | None]:
    cap = cv2.VideoCapture(str(path))
    centers = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        b, g, r = (img[..., i].astype(int) for i in range(3))
        ys, xs = np.nonzero((r > 150) & (g < 100) & (b < 100))
        centers.append((xs.mean(), ys.mean()) if len(xs) else None)
    return centers


def test_cfr_frame_count_and_timestamps(tmp_path):
    p = str(tmp_path / "p.mp4")
    truth = render_pendulum(p, duration_s=2.0)
    t = probe_times(p)
    assert len(truth["times"]) == 120
    assert len(t) == 120
    assert np.max(np.abs(t - truth["times"])) < 1e-3


def test_frame_count_includes_every_time_below_duration(tmp_path):
    truth = render_pendulum(str(tmp_path / "p.mp4"), duration_s=0.51)
    assert len(truth["times"]) == 31
    assert len(probe_times(str(tmp_path / "p.mp4"))) == 31


def test_vfr_timestamps_preserved(tmp_path):
    p = str(tmp_path / "p.mp4")
    truth = render_pendulum(p, duration_s=2.0, vfr=True, seed=3)
    t = probe_times(p)
    assert len(t) == len(truth["times"])
    assert np.max(np.abs(t - truth["times"])) < 2e-3
    assert np.std(np.diff(t)) > 0.5e-3


@pytest.mark.parametrize("vfr", [False, True])
def test_pendulum_bob_matches_ground_truth(tmp_path, vfr):
    p = tmp_path / "p.mp4"
    truth = render_pendulum(str(p), duration_s=2.0, theta0_deg=20.0, vfr=vfr)
    centers = red_centers(p)
    assert len(centers) == len(truth["bob_px"])
    for c, bob in zip(centers, truth["bob_px"], strict=True):
        assert c is not None
        assert np.hypot(c[0] - bob[0], c[1] - bob[1]) < 2


def test_fall_bob_matches_ground_truth_when_inside_frame(tmp_path):
    p = tmp_path / "f.mp4"
    truth = render_fall(str(p), drag="none")
    h = 1280
    inside = [k for k, (_, y) in enumerate(truth["bob_px"]) if BOB_RADIUS_PX <= y <= h - BOB_RADIUS_PX]
    assert len(inside) > 10
    centers = red_centers(p)
    for k in inside:
        c, bob = centers[k], truth["bob_px"][k]
        assert c is not None
        assert np.hypot(c[0] - bob[0], c[1] - bob[1]) < 2


def test_times_depend_only_on_seed(tmp_path):
    a = render_pendulum(str(tmp_path / "a.mp4"), duration_s=0.5, vfr=True, seed=1)["times"]
    b = render_pendulum(str(tmp_path / "b.mp4"), duration_s=0.5, vfr=True, seed=1)["times"]
    c = render_pendulum(str(tmp_path / "c.mp4"), duration_s=0.5, vfr=True, seed=2)["times"]
    assert a == b
    assert a != c


def test_quadratic_drag_slows_the_fall(tmp_path):
    def final_speed(truth):
        return abs((truth["y_m"][-1] - truth["y_m"][-2]) / (truth["times"][-1] - truth["times"][-2]))

    quad = render_fall(str(tmp_path / "q.mp4"), drag="quadratic", coeff=0.8)
    free = render_fall(str(tmp_path / "n.mp4"), drag="none")
    assert quad["y_m"][0] == 1.5
    assert final_speed(quad) < final_speed(free)


def test_a4_sheet_drawn_where_reported(tmp_path):
    p = tmp_path / "p.mp4"
    x1, y1, x2, y2 = render_pendulum(str(p), duration_s=0.1)["a4_box_px"]
    assert (x2 - x1, y2 - y1) == (238, 168)
    ok, img = cv2.VideoCapture(str(p)).read()
    assert ok
    inner = img[y1 + 3 : y2 - 3, x1 + 3 : x2 - 3].astype(int)
    assert np.all(np.abs(inner - 200) < 15)
    assert np.all(img[y1 - 4, x1:x2] > 240)
