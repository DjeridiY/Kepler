"""Spend tracker that refuses paid API calls past a USD cap."""

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path


class BudgetExceeded(Exception):
    pass


PRICES = {
    "sam_video_frame": 0.0002,  # USD per video frame, minimum 50 frames per request
    "sam_image": 0.0025,  # USD per still image
    "llm_input_token": 1.25e-6,  # USD per input token (muse-spark-1.3)
    "llm_output_token": 4.25e-6,  # USD per output token, reasoning included
}


def sam_video_cost(n_frames: int) -> float:
    return max(n_frames, 50) * PRICES["sam_video_frame"]


def llm_cost(input_tokens: int, output_tokens: int) -> float:
    return input_tokens * PRICES["llm_input_token"] + output_tokens * PRICES["llm_output_token"]


class Budget:
    def __init__(self, path: str, cap_usd: float = 20.0) -> None:
        self.path = Path(path)
        self.cap_usd = cap_usd

    def _load(self) -> dict:
        if not self.path.exists():
            return {"cap_usd": self.cap_usd, "spent_usd": 0.0, "entries": []}
        return json.loads(self.path.read_text())

    @property
    def spent(self) -> float:
        return float(self._load()["spent_usd"])

    def reserve(self, label: str, usd: float) -> None:
        """Raise BudgetExceeded if spending `usd` more would pass the cap; call before a paid call."""
        spent = self.spent
        if spent + usd > self.cap_usd:
            raise BudgetExceeded(f"budget cap {self.cap_usd:.2f} USD reached: spent {spent:.2f}, need {usd:.4f}")

    def charge(self, label: str, usd: float) -> None:
        """Record the actual cost of a paid call that already happened."""
        if usd < 0:
            raise ValueError(f"cost must be non-negative, got {usd}")
        data = self._load()
        data["cap_usd"] = self.cap_usd
        data["spent_usd"] = data["spent_usd"] + usd
        data["entries"].append({"label": label, "usd": usd, "time": datetime.now(UTC).isoformat()})
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, self.path)
