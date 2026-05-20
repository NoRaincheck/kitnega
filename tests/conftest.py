import os

import pytest


@pytest.fixture(autouse=True)
def _clean_cody_state(monkeypatch):
    """Reset cached state in cody modules before each test."""
    monkeypatch.delenv("KN_API", raising=False)  # disable real API calls by default


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """Override HOME so session files land under tmp_path."""
    os.environ["HOME"] = str(tmp_path)
    return tmp_path
