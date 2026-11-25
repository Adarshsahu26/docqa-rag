"""Fixtures shared by all tests."""
from __future__ import annotations

import os

import pytest

from tests.helpers import FakeLLM, make_pipeline


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    """Tests must not depend on the developer's own keys or DOCQA_* settings."""
    for name in list(os.environ):
        if name.startswith("DOCQA_") or name in {"ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"}:
            monkeypatch.delenv(name)
    # Local fake servers must be reached directly, even on machines that route traffic through a proxy.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture()
def pipeline(fake_llm):
    return make_pipeline(fake_llm)
