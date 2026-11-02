"""A lab agent: a language model describes the setup and proposes laws; computation judges them."""

import base64
import importlib
import json
import math
import re
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np

from kepler.fit import FitResult, fit_law
from kepler.laws import BUILTIN_LAWS, LawError, load_law


class AgentError(Exception):
    """The language model did not give a usable answer."""


class LLM(Protocol):
    def complete(self, system: str, user: str, images: list[bytes]) -> tuple[str, int, int]: ...


@dataclass(frozen=True)
class Setup:
    experiment: str  # "pendulum" | "fall" | "other"
    object_prompt: str  # short noun phrase for SAM, e.g. "red nut"
    signal: str  # "angle" | "height"
    description: str


@dataclass(frozen=True)
class Entry:
    round: int
    name: str
    source: str
    status: str  # "accepted" | "rejected" | "invalid"
    rel_err_test: float | None
    reason: str


EXPERIMENTS = ("pendulum", "fall", "other")
SIGNALS = ("angle", "height")


def extract_json(text: str) -> dict:
    decoder = json.JSONDecoder()
    for start in (m.start() for m in re.finditer(r"\{", text)):
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise AgentError("no JSON object in reply")


def _ask[T](llm: LLM, system: str, user: str, images: list[bytes], parse: Callable[[dict], T], *, retries: int) -> T:
    error = AgentError("no attempt made")
    for _ in range(retries + 1):
        text, _, _ = llm.complete(system, user, images)
        try:
            return parse(extract_json(text))
        except AgentError as e:
            error = e
    raise error


_SETUP_SYSTEM = (
    "You are an experimental physicist looking at frames of a phone video of a lab experiment. "
    'Reply with one JSON object only: {"experiment": one of ' + json.dumps(EXPERIMENTS) + ', "object_prompt": '
    'a short noun phrase naming the moving object (e.g. "red nut"), "signal": one of ' + json.dumps(SIGNALS) + ", "
    '"description": one sentence describing the setup}. Do not write any measured value.'
)


def _parse_setup(data: dict) -> Setup:
    keys = ("experiment", "object_prompt", "signal", "description")
    fields = [f for k in keys if isinstance(f := data.get(k), str) and f]
    if len(fields) != len(keys):
        raise AgentError("setup fields must be non-empty strings")
    setup = Setup(*fields)
    if setup.experiment not in EXPERIMENTS or setup.signal not in SIGNALS:
        raise AgentError(f"unknown experiment or signal: {setup.experiment!r}, {setup.signal!r}")
    return setup


def describe_setup(llm: LLM, frames: list[bytes], *, retries: int = 2) -> Setup:
    return _ask(llm, _SETUP_SYSTEM, "Describe this experiment.", frames, _parse_setup, retries=retries)


_LAWS_SYSTEM = (
    "You are a physicist proposing laws of motion as Python source. Each law is a module that defines "
    "NAME (a str), PARAMS (a dict mapping each parameter name to a (low, high, initial) tuple of floats) and "
    "accel(t, y, v, p) returning the second derivative of the signal y, with p the dict of parameter values. "
    "Only math and np are available; no imports, no names starting with an underscore. Example:\n"
    + BUILTIN_LAWS["damped pendulum (linear)"]
    + '\nReply with one JSON object only: {"laws": [<source>, ...]}.'
)


def _history_line(e: Entry) -> str:
    if e.rel_err_test is None:
        return f"- {e.name}: {e.status} ({e.reason})"
    return f"- {e.name}: {e.status}, test error {e.rel_err_test * 100:.2f} %"


def propose_laws(llm: LLM, setup: Setup, history: list[Entry], *, n: int = 3, retries: int = 2) -> list[str]:
    tried = "\n".join(map(_history_line, history)) or "- none yet"
    user = (
        f"Experiment: {setup.experiment}; signal: {setup.signal}; {setup.description}\n"
        "Each law is fitted on the first 70 % of the series and judged on the unseen rest "
        "(test error = test RMSE / signal range).\n"
        f"Laws tried so far:\n{tried}\nPropose up to {n} new laws that could beat them."
    )

    def parse(data: dict) -> list[str]:
        laws = data.get("laws")
        if not isinstance(laws, list) or not all(isinstance(s, str) for s in laws):
            raise AgentError('"laws" must be a list of source strings')
        return laws[:n]

    return _ask(llm, _LAWS_SYSTEM, user, [], parse, retries=retries)


_NUMBER = re.compile(r"(?<![\w.,])([-+]?\d+(?:[.,]\d+)?)(\s*%)?")


def check_numbers(text: str, allowed: list[float], rel_tol: float = 0.02) -> list[str]:
    unjustified = []
    for m in _NUMBER.finditer(text):
        digits, percent = m.group(1), m.group(2)
        value = float(digits.replace(",", "."))
        if re.fullmatch(r"\d+", digits) and value <= 10:
            continue
        targets = [a * 100 if percent else a for a in allowed]
        if not any(math.isclose(value, a, rel_tol=rel_tol) for a in targets):
            unjustified.append(m.group(0))
    return unjustified


def _llm_cost(input_tokens: int, output_tokens: int) -> float:
    try:
        budget = importlib.import_module("kepler.budget")
    except ModuleNotFoundError:
        return 1.25e-6 * input_tokens + 4.25e-6 * output_tokens
    return budget.llm_cost(input_tokens, output_tokens)


@dataclass
class _Metered:
    llm: LLM
    on_cost: Callable[[str, float], None] | None

    def complete(self, system: str, user: str, images: list[bytes]) -> tuple[str, int, int]:
        text, tokens_in, tokens_out = self.llm.complete(system, user, images)
        if self.on_cost:
            self.on_cost("llm", _llm_cost(tokens_in, tokens_out))
        return text, tokens_in, tokens_out


def _judge(round_: int, source: str, t: np.ndarray, y: np.ndarray) -> tuple[Entry, FitResult | None]:
    try:
        law = load_law(source)
    except LawError as e:
        return Entry(round_, "(unnamed)", source, "invalid", None, str(e)), None
    try:
        fit = fit_law(law, t, y)
    except Exception as e:
        # Builtin laws already fitted these data, so a failure here comes from the proposed law's own code.
        return Entry(round_, law.name, source, "invalid", None, f"law failed while fitting: {e}"), None
    return Entry(round_, fit.name, source, "rejected", fit.rel_err_test, ""), fit


def _key(fit: FitResult) -> float:
    return fit.rmse_test if math.isfinite(fit.rmse_test) else math.inf


def run_lab(
    llm: LLM,
    t: np.ndarray,
    y: np.ndarray,
    setup: Setup,
    *,
    rounds: int = 3,
    target: float = 0.01,
    on_cost: Callable[[str, float], None] | None = None,
) -> tuple[list[Entry], FitResult]:
    entries: list[Entry] = []
    fits: dict[int, FitResult] = {}  # entry index -> fit, for valid laws only
    for source in BUILTIN_LAWS.values():
        fits[len(entries)] = fit = fit_law(load_law(source), t, y)
        entries.append(Entry(0, fit.name, source, "rejected", fit.rel_err_test, ""))
    metered = _Metered(llm, on_cost)

    def best() -> int:
        return min(fits, key=lambda i: _key(fits[i]))

    for round_ in range(1, rounds + 1):
        if fits[best()].rel_err_test < target:
            break
        for source in propose_laws(metered, setup, entries):
            entry, fit = _judge(round_, source, t, y)
            if fit is not None:
                fits[len(entries)] = fit
            entries.append(entry)
    i = best()
    entries[i] = replace(entries[i], status="accepted")
    return entries, fits[i]


_PROVIDERS = {
    "meta": ("muse-spark-1.3", "https://api.meta.ai/v1"),
    "baseten": ("meta-models/muse-spark-open-1026", "https://inference.baseten.co/v1"),
}


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


class OpenAICompatLLM:
    """Meta (responses API) or Baseten (chat.completions) over HTTPS."""

    def __init__(self, provider: str, api_key: str):
        if provider not in _PROVIDERS:
            raise AgentError(f"unknown provider: {provider!r}")
        self.provider = provider
        self.model, self.base_url = _PROVIDERS[provider]
        self._api_key = api_key

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)

    def complete(self, system: str, user: str, images: list[bytes]) -> tuple[str, int, int]:
        if self.provider == "meta":
            content = [{"type": "input_text", "text": user}]
            content += [{"type": "input_image", "image_url": _data_url(img)} for img in images]
            reply = self._post(
                "/responses",
                {"model": self.model, "instructions": system, "input": [{"role": "user", "content": content}]},
            )
            text = "".join(
                part["text"]
                for item in reply["output"]
                if item.get("type") == "message"
                for part in item["content"]
                if part.get("type") == "output_text"
            )
            return text, reply["usage"]["input_tokens"], reply["usage"]["output_tokens"]
        content = [{"type": "text", "text": user}]
        content += [{"type": "image_url", "image_url": {"url": _data_url(img)}} for img in images]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        reply = self._post("/chat/completions", {"model": self.model, "messages": messages})
        usage = reply["usage"]
        return reply["choices"][0]["message"]["content"], usage["prompt_tokens"], usage["completion_tokens"]
