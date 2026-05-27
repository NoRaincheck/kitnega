"""Shared configuration loader for kitnega tools.

Reads settings from ``~/.kitnega/config.json``. Creates the file with
defaults on first use if it doesn't exist.  All modules should call
``get_config()`` (once per process) rather than reading environment
variables directly — this keeps every setting in one place.

Example config::

    {
        "api": "http://localhost:1234/v1/responses",
        "model": "qwen3.6-35b-a3b",
        "api_key": "",
        "stream": true,
        "max_steps": 20,
        "approve_all": false,
        "read_limit": 30,
        "turn_cap": 100,
        "bash_mode": "auto",
        "bash_allow": []
    }

Migration: if the file contains old ``kn_api`` / ``kn_model`` keys they are
renamed to ``api`` / ``model`` on write.
"""

import json
import os

_CONFIG_DIR = os.path.expanduser("~/.kitnega")
_CONFIG_PATH = os.path.join(_CONFIG_DIR, "config.json")

_DEFAULTS = {
    "api": "http://localhost:1234/v1/responses",
    "model": "qwen3.6-35b-a3b",
    "api_key": "",
    "stream": True,
    "max_steps": 20,
    "approve_all": False,
    "read_limit": 30,
    "turn_cap": 100,
    "bash_mode": "auto",
    "bash_allow": [],
}

_cache = None


def _ensure_file():
    """Create config file with defaults if it does not exist."""
    os.makedirs(_CONFIG_DIR, exist_ok=True)
    if not os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH, "w") as f:
            json.dump(_DEFAULTS, f, indent=2)


def get_config():
    """Return a dict of all kitnega configuration.

    Results are cached per-process after the first call.
    """
    global _cache
    if _cache is not None:
        return _cache
    _ensure_file()
    try:
        with open(_CONFIG_PATH) as f:
            cfg = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cfg = dict(_DEFAULTS)

    # Normalise old env-var-style keys to new names.
    rename_map = {
        "kn_api": "api",
        "kn_model": "model",
        "kn_api_key": "api_key",
        "kn_stream": "stream",
        "kn_max_steps": "max_steps",
        "kn_approve_all": "approve_all",
        "kn_read_limit": "read_limit",
        "kn_turn_cap": "turn_cap",
        "kn_bash_mode": "bash_mode",
        "kn_bash_allow": "bash_allow",
    }
    for old, new in rename_map.items():
        if old in cfg and new not in cfg:
            cfg[new] = cfg.pop(old)

    # Fill missing keys with defaults.
    for key, val in _DEFAULTS.items():
        cfg.setdefault(key, val)

    _cache = cfg
    return cfg


def reload_config():
    """Discard the cached config and re-read from disk."""
    global _cache
    _cache = None
    return get_config()


def save_config(cfg=None):
    """Write the current (or given) config dict to disk.

    Also updates the in-process cache so subsequent ``get_config()`` calls
    see the new values immediately.
    """
    global _cache
    if cfg is None:
        cfg = get_config()
    _ensure_file()
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
    _cache = dict(cfg)
