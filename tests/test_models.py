from __future__ import annotations

import pytest

from fitd_repro.models import build_model


def test_openai_model_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("FITD_ALLOW_OPENAI", raising=False)

    with pytest.raises(RuntimeError, match="disabled by default"):
        build_model(backend="openai", model_name="gpt-4o-mini")
