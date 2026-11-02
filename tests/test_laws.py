import math

import pytest

from kepler.laws import BUILTIN_LAWS, LawError, load_law

VALID = """
NAME = "custom"
PARAMS = {"a": (0.0, 2.0, 1.0)}
def accel(t, y, v, p):
    return -p["a"] * math.cos(y) * np.sign(v)
"""


def test_loads_law_using_math_and_numpy():
    law = load_law(VALID)
    assert law.name == "custom"
    assert law.params == {"a": (0.0, 2.0, 1.0)}
    assert law.accel(0.0, 0.0, -1.0, {"a": 2.0}) == pytest.approx(2.0)
    assert law.source == VALID


@pytest.mark.parametrize(
    "line",
    [
        "import os",
        "from os import path",
        "x = __import__('os')",
        "x = ().__class__",
        "x = open('x')",
        "x = eval('1')",
        "x = getattr(math, 'pi')",
        "_x = 1",
    ],
)
def test_rejects_unsafe_code(line):
    with pytest.raises(LawError):
        load_law(line + "\n" + VALID)


@pytest.mark.parametrize(
    ("source", "missing"),
    [
        ('PARAMS = {"a": (0.0, 1.0, 0.5)}\ndef accel(t, y, v, p):\n    return 0.0\n', "NAME"),
        ('NAME = "x"\ndef accel(t, y, v, p):\n    return 0.0\n', "PARAMS"),
        ('NAME = "x"\nPARAMS = {}\ndef accel(t, y, v, p):\n    return 0.0\n', "PARAMS"),
        ('NAME = "x"\nPARAMS = {"a": (0.0, 1.0, 2.0)}\ndef accel(t, y, v, p):\n    return 0.0\n', "PARAMS"),
        ('NAME = "x"\nPARAMS = {"a": (0.0, 1.0, 0.5)}\n', "accel"),
        ('NAME = "x"\nPARAMS = {"a": (0.0, 1.0, 0.5)}\ndef accel(t, y, v):\n    return 0.0\n', "accel"),
    ],
)
def test_reports_missing_or_invalid_element(source, missing):
    with pytest.raises(LawError, match=missing):
        load_law(source)


def test_builtin_laws():
    assert set(BUILTIN_LAWS) == {
        "pendulum",
        "damped pendulum (linear)",
        "damped pendulum (quadratic)",
        "free fall",
        "fall with linear drag",
        "fall with quadratic drag",
    }
    laws = {name: load_law(src) for name, src in BUILTIN_LAWS.items()}
    assert all(law.name == name for name, law in laws.items())
    assert laws["free fall"].params["g"] == (8.0, 11.5, 9.8)
    y, v = 0.5, -2.0
    expected = {
        "pendulum": -20 * math.sin(y),
        "damped pendulum (linear)": -20 * math.sin(y) - 2 * 0.3 * v,
        "damped pendulum (quadratic)": -20 * math.sin(y) - 0.4 * v * abs(v),
        "free fall": -9.8,
        "fall with linear drag": -9.8 - 0.7 * v,
        "fall with quadratic drag": -9.8 - 0.6 * v * abs(v),
    }
    p = {"w2": 20.0, "gamma": 0.3, "c": 0.4, "g": 9.8, "b": 0.7, "k": 0.6}
    for name, law in laws.items():
        assert law.accel(0.0, y, v, {k: p[k] for k in law.params}) == pytest.approx(expected[name])
