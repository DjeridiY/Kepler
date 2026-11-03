import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from kepler import cli, home
from kepler.budget import BudgetExceeded
from kepler.pipeline import RunError

KEPLER = str(Path(sys.executable).parent / "kepler")


def _kepler(*args: str, **env: str) -> subprocess.CompletedProcess:
    base = {k: v for k, v in os.environ.items() if k not in ("META_API_KEY", "BASETEN_API_KEY")}
    return subprocess.run([KEPLER, *args], capture_output=True, text=True, env={**base, **env}, check=False)


def test_help_exits_zero():
    done = _kepler("--help")
    assert done.returncode == 0
    assert "run" in done.stdout


def test_run_without_meta_key_exits_2_without_traceback(tmp_path):
    done = _kepler("run", str(tmp_path / "clip.mp4"), HOME=str(tmp_path))
    assert done.returncode == 2
    assert "META_API_KEY" in done.stderr
    assert "Traceback" not in done.stderr
    assert not (tmp_path / ".kepler").exists()


def test_sam_tracker_needs_meta_key_even_with_baseten(tmp_path):
    done = _kepler("run", "clip.mp4", "--provider", "baseten", BASETEN_API_KEY="k", HOME=str(tmp_path))
    assert done.returncode == 2
    assert "META_API_KEY" in done.stderr


HOME_ARGS = {
    "bet": ["a.mp4", "--length-a", "0.45", "--length-b", "0.30"],
    "cup": ["r1.mp4", "r2.mp4", "r3.mp4", "--table-height", "0.75"],
    "weigh": ["ref.mp4", "obj.mp4", "--ref-mass", "0.5"],
}


@pytest.mark.parametrize("command", HOME_ARGS)
def test_home_command_help_and_missing_key(command, tmp_path):
    assert _kepler(command, "--help").returncode == 0
    done = _kepler(command, *HOME_ARGS[command], HOME=str(tmp_path))
    assert done.returncode == 2
    assert "META_API_KEY" in done.stderr
    assert "Traceback" not in done.stderr


def _sealed(result: dict, value: float) -> dict:
    sha, computed = home.seal([value])
    return {**result, "seal_sha256": sha, "computed": computed}


FAKE_RESULTS = {
    "bet": (_sealed({"period_a_s": 1.357, "period_b_pred_s": 1.108}, 1.108), "PERIOD B  1.108 s"),
    "cup": (
        _sealed({"runs": [{}, {}, {}], "d_pred_m": 0.6191, "spread_m": 0.008}, 0.6191),
        "PUT THE CUP AT  61.9 cm  (±0.4 cm over 3 runs)",
    ),
    "weigh": (_sealed({"period_ref_s": 0.75, "mass_kg": 0.1812}, 0.1812), "MASS  181 g"),
}


@pytest.fixture
def no_network(monkeypatch, tmp_path):
    monkeypatch.setenv("META_API_KEY", "k")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(cli, "MetaTransport", lambda key: object())


@pytest.mark.parametrize("command", HOME_ARGS)
def test_home_command_writes_result_and_seal_then_calls_the_number(command, no_network, monkeypatch, tmp_path, capsys):
    result, headline = FAKE_RESULTS[command]
    monkeypatch.setattr(home, command, lambda *a, **k: result)
    out = tmp_path / "out"
    assert cli.main([command, *HOME_ARGS[command], "--out", str(out)]) == 0
    assert capsys.readouterr().out.splitlines()[0] == headline
    assert json.loads((out / "result.json").read_text()) == result
    seal_lines = (out / "seal.txt").read_text().splitlines()
    assert seal_lines == [f"sha256 {result['seal_sha256']}", f"computed {result['computed']}"]


@pytest.mark.parametrize(("error", "code"), [(BudgetExceeded("cap"), 3), (RunError("no rolling on the table"), 4)])
def test_home_command_exit_codes(error, code, no_network, monkeypatch, tmp_path):
    def fail(*a, **k):
        raise error

    monkeypatch.setattr(home, "cup", fail)
    assert cli.main(["cup", "r.mp4", "--table-height", "0.75", "--out", str(tmp_path / "out")]) == code
    assert not (tmp_path / "out").exists()
