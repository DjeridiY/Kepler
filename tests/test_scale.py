import cv2
import numpy as np
import pytest

from kepler.scale import ScaleError, px_per_meter_from_box, px_per_meter_from_mask, to_meters

BOX = (100.0, 100.0, 694.0, 520.0)


def test_box_a4_gives_px_per_meter() -> None:
    assert px_per_meter_from_box(BOX) == pytest.approx(2000.0, abs=1e-9)


def test_declared_reference_size_scales_distances() -> None:
    s = px_per_meter_from_box(BOX, long_m=0.594, short_m=0.420)
    assert s == pytest.approx(1000.0, abs=1e-9)
    pts = np.array([[0.0, 0.0], [300.0, 400.0]])
    d_small = np.linalg.norm(np.diff(to_meters(pts, 2000.0), axis=0))
    d_big = np.linalg.norm(np.diff(to_meters(pts, s), axis=0))
    assert d_big == 2 * d_small


def test_square_box_is_not_face_on() -> None:
    with pytest.raises(ScaleError, match="not face-on"):
        px_per_meter_from_box((0.0, 0.0, 400.0, 400.0))


def test_rotated_mask_gives_px_per_meter() -> None:
    rect = ((600.0, 600.0), (594.0, 420.0), 30.0)
    corners = cv2.boxPoints(rect).round().astype(np.int32)
    img = np.zeros((1200, 1200), np.uint8)
    cv2.fillPoly(img, [corners], 1)
    assert px_per_meter_from_mask(img.astype(bool)) == pytest.approx(2000.0, rel=0.01)


def test_empty_mask_raises() -> None:
    with pytest.raises(ScaleError, match="empty reference mask"):
        px_per_meter_from_mask(np.zeros((10, 10), bool))


def test_to_meters_flips_y_up() -> None:
    out = to_meters(np.array([[10.0, 20.0]]), 100.0, origin_px=(0.0, 120.0))
    np.testing.assert_allclose(out, [[0.1, 1.0]])


def test_to_meters_rejects_non_positive_scale() -> None:
    with pytest.raises(ValueError):
        to_meters(np.array([[1.0, 2.0]]), 0.0)
