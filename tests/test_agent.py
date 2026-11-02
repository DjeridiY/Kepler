import json
import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from kepler.agent import (
    AgentError,
    Entry,
    OpenAICompatLLM,
    Setup,
    check_numbers,
    describe_setup,
    extract_json,
    propose_laws,
    run_lab,
)

SETUP_JSON = '{"experiment":"pendulum","object_prompt":"red nut","signal":"angle","description":"x"}'
SETUP = Setup("pendulum", "red nut", "angle", "x")
INVALID_LAW = 'import os\nNAME = "bad"\nPARAMS = {"a": (0.0, 1.0, 0.5)}\ndef accel(t, y, v, p):\n    return 0.0\n'
VALID_LAW = (
    'NAME = "small-angle pendulum"\nPARAMS = {"w2": (1.0, 400.0, 20.0)}\n'
    'def accel(t, y, v, p):\n    return -p["w2"] * y\n'
)


class FakeLLM:
    """Replies with the scripted texts in order, repeating the last one."""

    def __init__(self, replies: list[str], tokens: tuple[int, int] = (1000, 200)):
        self.replies = replies
        self.tokens = tokens
        self.calls: list[tuple[str, str, list[bytes]]] = []

    def complete(self, system: str, user: str, images: list[bytes]) -> tuple[str, int, int]:
        self.calls.append((system, user, images))
        text = self.replies[min(len(self.calls), len(self.replies)) - 1]
        return text, *self.tokens


@pytest.fixture(scope="module")
def pendulum() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(0.0, 10.0, 1 / 60)
    sol = solve_ivp(
        lambda _, s: [s[1], -21.8 * math.sin(s[0]) - 2 * 0.05 * s[1]],
        (t[0], t[-1]),
        [0.14, 0.0],
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
    )
    return t, sol.y[0] + np.random.default_rng(0).normal(0.0, 0.002, t.size)


def test_extract_json_reads_fenced_block():
    assert extract_json('Sure:\n```json\n{"a": {"b": 1}}\n```\nthen {"c": 2}') == {"a": {"b": 1}}


def test_extract_json_without_object_fails():
    with pytest.raises(AgentError, match="no JSON object in reply"):
        extract_json("no braces here, {not json}")


def test_describe_setup_retries_after_reply_without_json():
    llm = FakeLLM(["It looks like a pendulum.", SETUP_JSON])
    assert describe_setup(llm, [b"jpeg"]) == SETUP
    assert len(llm.calls) == 2
    assert llm.calls[0][2] == [b"jpeg"]


def test_describe_setup_fails_after_retries():
    llm = FakeLLM(["no json"] * 3)
    with pytest.raises(AgentError):
        describe_setup(llm, [b"jpeg"])
    assert len(llm.calls) == 3


def test_describe_setup_rejects_values_outside_lists():
    bad = SETUP_JSON.replace('"angle"', '"speed"')
    with pytest.raises(AgentError):
        describe_setup(FakeLLM([bad]), [], retries=1)


def test_propose_laws_shows_history_and_caps_count():
    llm = FakeLLM([json.dumps({"laws": ["a", "b", "c", "d"]})])
    history = [Entry(0, "pendulum", "src", "rejected", 0.0123, ""), Entry(1, "?", "src", "invalid", None, "boom")]
    assert propose_laws(llm, SETUP, history, n=3) == ["a", "b", "c"]
    prompt = llm.calls[0][1]
    assert "pendulum: rejected, test error 1.23 %" in prompt
    assert "invalid" in prompt
    assert "boom" in prompt
    assert "def accel(t, y, v, p)" in llm.calls[0][0]


def test_propose_laws_retries_on_bad_shape():
    llm = FakeLLM(['{"laws": "x"}', "nothing", '{"laws": ["a"]}'])
    assert propose_laws(llm, SETUP, [], retries=2) == ["a"]
    with pytest.raises(AgentError):
        propose_laws(FakeLLM(['{"laws": 3}']), SETUP, [], retries=1)


def test_check_numbers_example():
    text = "period 1.35 s, error 0.7 %, length 0.45 m, g 12.0"
    assert check_numbers(text, [1.347, 0.0070, 0.4519]) == ["12.0"]


def test_check_numbers_rules():
    allowed = [1.347, 0.0070, 0.4519, -2.0]
    # Small integers pass; small decimals, wrong sign, and a percentage of a raw value do not.
    assert check_numbers("3 laws, 2 s, 2.5 s", allowed) == ["2.5"]
    assert check_numbers("v = -2.0 or 2.0", allowed) == ["2.0"]
    assert check_numbers("0.45m (1.35) 1,35 0,7%", allowed) == []
    assert check_numbers("0.0070 %, 134.7 %, 0.7", allowed) == ["0.0070 %", "0.7"]
    assert check_numbers("g 12.0 and 13.5 and 11", allowed) == ["12.0", "13.5", "11"]


def test_lab_marks_invalid_rejected_and_accepted(pendulum):
    t, y = pendulum
    llm = FakeLLM([json.dumps({"laws": [INVALID_LAW, VALID_LAW]})])
    entries, best = run_lab(llm, t, y, SETUP, rounds=1, target=0.001)
    assert len(llm.calls) == 1
    by_status = {e.name: e.status for e in entries}
    assert len(entries) == 8
    invalid = [e for e in entries if e.status == "invalid"]
    assert len(invalid) == 1
    assert invalid[0].round == 1
    assert "imports are not allowed" in invalid[0].reason
    assert invalid[0].rel_err_test is None
    assert best.name == "damped pendulum (linear)"
    assert by_status["damped pendulum (linear)"] == "accepted"
    assert by_status["small-angle pendulum"] == "rejected"
    assert [e.status for e in entries].count("accepted") == 1
    assert [e.status for e in entries].count("rejected") == 6


def test_lab_stops_when_builtin_law_is_good_enough(pendulum):
    t, y = pendulum
    llm = FakeLLM([json.dumps({"laws": [VALID_LAW]})])
    entries, best = run_lab(llm, t, y, SETUP, rounds=3, target=0.05)
    assert llm.calls == []
    assert len(entries) == 6
    assert best.rel_err_test < 0.05


def test_lab_reports_cost_of_every_llm_call(pendulum):
    t, y = pendulum
    llm = FakeLLM(["no json", json.dumps({"laws": [INVALID_LAW]})], tokens=(1000, 200))
    costs: list[tuple[str, float]] = []
    run_lab(llm, t, y, SETUP, rounds=2, target=0.0, on_cost=lambda kind, usd: costs.append((kind, usd)))
    assert len(llm.calls) == 3
    assert costs == [("llm", pytest.approx(1.25e-6 * 1000 + 4.25e-6 * 200))] * 3


def test_unknown_provider_is_rejected():
    with pytest.raises(AgentError):
        OpenAICompatLLM("acme", "key")
