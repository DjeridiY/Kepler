import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from kepler.budget import Budget, BudgetExceeded, llm_cost, sam_video_cost


def test_costs() -> None:
    assert sam_video_cost(20) == pytest.approx(0.01, abs=1e-12)
    assert sam_video_cost(600) == pytest.approx(0.12, abs=1e-12)
    assert llm_cost(1000, 2000) == pytest.approx(0.00975, abs=1e-12)


def test_reserve_refuses_past_cap_without_writing(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    budget = Budget(str(path), cap_usd=1.0)
    budget.charge("a", 0.6)
    before = path.read_bytes()
    with pytest.raises(BudgetExceeded) as exc:
        budget.reserve("b", 0.5)
    assert "budget cap 1.00 USD reached" in str(exc.value)
    budget.reserve("b", 0.4)
    assert path.read_bytes() == before


def test_reserve_up_to_exact_cap_passes(tmp_path: Path) -> None:
    budget = Budget(str(tmp_path / "budget.json"), cap_usd=1.0)
    budget.charge("a", 0.5)
    budget.reserve("b", 0.5)


def test_reserve_does_not_create_file(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    Budget(str(path), cap_usd=1.0).reserve("a", 0.1)
    assert not path.exists()


def test_second_instance_sees_spend(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    Budget(str(path), cap_usd=1.0).charge("a", 0.6)
    other = Budget(str(path))
    assert other.spent == pytest.approx(0.6)
    data = json.loads(path.read_text())
    assert data["cap_usd"] == 1.0
    assert [e["label"] for e in data["entries"]] == ["a"]
    assert datetime.fromisoformat(data["entries"][0]["time"]).utcoffset() == timedelta(0)


def test_negative_charge_rejected_file_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    budget = Budget(str(path))
    with pytest.raises(ValueError):
        budget.charge("x", -1)
    assert not path.exists()
    budget.charge("a", 0.1)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        budget.charge("x", -1)
    assert path.read_bytes() == before
