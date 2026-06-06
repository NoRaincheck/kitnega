import os

import pytest
from lib.config import reload_config


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset config cache before each test so tests start with a clean slate."""
    reload_config()


@pytest.fixture()
def fake_home(tmp_path, monkeypatch):
    """Override HOME so session files land under tmp_path."""
    os.environ["HOME"] = str(tmp_path)
    return tmp_path
