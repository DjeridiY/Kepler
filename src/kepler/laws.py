"""Candidate physical laws written as small, sandboxed Python sources."""

import ast
import math
import types
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class LawError(Exception):
    """A law source is unsafe or does not follow the law contract."""


@dataclass(frozen=True)
class Law:
    name: str
    params: dict[str, tuple[float, float, float]]
    accel: Callable[[float, float, float, dict[str, float]], float]
    source: str


_FORBIDDEN_NAMES = {"eval", "exec", "open", "compile", "globals", "locals", "getattr", "setattr"}
_BUILTINS = {f.__name__: f for f in (abs, min, max, float, int, pow, range, len, sum, round)}


def _check_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            raise LawError("imports are not allowed")
        name = None
        if isinstance(node, ast.Name):
            name = node.id
        elif isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.FunctionDef | ast.arg):
            name = node.name if isinstance(node, ast.FunctionDef) else node.arg
        if name is not None and (name.startswith("_") or name in _FORBIDDEN_NAMES):
            raise LawError(f"forbidden name: {name}")


def _check_params(params: object) -> dict[str, tuple[float, float, float]]:
    if not isinstance(params, dict) or not params:
        raise LawError("PARAMS must be a non-empty dict")
    checked = {}
    for key, bounds in params.items():
        if not isinstance(key, str) or not isinstance(bounds, tuple) or len(bounds) != 3:
            raise LawError(f"PARAMS[{key!r}] must be a (low, high, initial) tuple")
        low, high, initial = (float(b) for b in bounds)
        if not low <= initial <= high:
            raise LawError(f"PARAMS[{key!r}] needs low <= initial <= high")
        checked[key] = (low, high, initial)
    return checked


def load_law(source: str) -> Law:
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise LawError(f"invalid syntax: {e}") from e
    _check_ast(tree)
    namespace: dict = {"__builtins__": dict(_BUILTINS), "math": math, "np": np}
    try:
        exec(compile(tree, "<law>", "exec"), namespace)
    except Exception as e:
        raise LawError(f"law source failed to run: {e}") from e
    name = namespace.get("NAME")
    if not isinstance(name, str):
        raise LawError("NAME must be a str")
    accel = namespace.get("accel")
    if not isinstance(accel, types.FunctionType) or accel.__code__.co_argcount != 4:
        raise LawError("accel must be a function of (t, y, v, p)")
    return Law(name, _check_params(namespace.get("PARAMS")), accel, source)


def _law(name: str, params: str, expr: str) -> str:
    return f'NAME = "{name}"\nPARAMS = {{{params}}}\ndef accel(t, y, v, p):\n    return {expr}\n'


_W2 = '"w2": (1.0, 400.0, 20.0)'
_G = '"g": (8.0, 11.5, 9.8)'

BUILTIN_LAWS: dict[str, str] = {
    "pendulum": _law("pendulum", _W2, '-p["w2"] * math.sin(y)'),
    "damped pendulum (linear)": _law(
        "damped pendulum (linear)", f'{_W2}, "gamma": (0.0, 5.0, 0.1)', '-p["w2"] * math.sin(y) - 2 * p["gamma"] * v'
    ),
    "damped pendulum (quadratic)": _law(
        "damped pendulum (quadratic)", f'{_W2}, "c": (0.0, 5.0, 0.1)', '-p["w2"] * math.sin(y) - p["c"] * v * abs(v)'
    ),
    "free fall": _law("free fall", _G, '-p["g"]'),
    "fall with linear drag": _law("fall with linear drag", f'{_G}, "b": (0.0, 10.0, 0.1)', '-p["g"] - p["b"] * v'),
    "fall with quadratic drag": _law(
        "fall with quadratic drag", f'{_G}, "k": (0.0, 10.0, 0.1)', '-p["g"] - p["k"] * v * abs(v)'
    ),
}
