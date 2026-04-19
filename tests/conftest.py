from __future__ import annotations

import sys
from pathlib import Path

import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeJudge:
    """Deterministic judge for tests. Maps (goal, response) -> canned verdict.

    Pass ``rules`` as a list of ``(goal_substr, response_substr, label, harm)``
    tuples. The first matching rule wins. Falls back to ``default_label``.
    Tracks every call on ``calls`` for assertions.
    """

    def __init__(
        self,
        rules: list[tuple[str, str, str, int]] | None = None,
        default_label: str = "safe",
        default_harm: int = 1,
    ):
        self.rules = rules or []
        self.default_label = default_label
        self.default_harm = default_harm
        self.calls: list[dict] = []

    def _match(self, goal: str, response: str) -> tuple[str, int]:
        for goal_sub, resp_sub, label, harm in self.rules:
            if goal_sub in (goal or "") and resp_sub in (response or ""):
                return label, harm
        return self.default_label, self.default_harm

    def score_alignment(self, goal: str, response: str) -> dict:
        label, _ = self._match(goal, response)
        self.calls.append({"kind": "alignment", "goal": goal, "response": response, "label": label})
        return {"label": label, "raw": label}

    def score_harmfulness(self, goal: str, response: str) -> dict:
        _, harm = self._match(goal, response)
        self.calls.append({"kind": "harmfulness", "goal": goal, "response": response, "harm": harm})
        return {
            "harm": harm,
            "reason": "fake-judge canned reason",
            "raw": f'{{"score": {harm}}}',
            "parse_error": False,
        }


@pytest.fixture
def fake_judge() -> FakeJudge:
    return FakeJudge()
