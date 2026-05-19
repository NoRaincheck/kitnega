import json
import os
import time

from ._shared import CWD, _color

SESSIONS = os.path.expanduser("~/.cody_sessions.json")
_MAX_SESSIONS = 50

_cache = None
_dirty = False


def _read():
    global _cache, _dirty
    if _cache is not None:
        return _cache
    try:
        with open(SESSIONS) as f:
            _cache = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        _cache = []
    _dirty = False
    return _cache


def _flush():
    global _dirty
    if not _dirty or _cache is None:
        return
    if len(_cache) > _MAX_SESSIONS:
        _cache[:] = _cache[-_MAX_SESSIONS:]
    with open(SESSIONS, "w") as f:
        json.dump(_cache, f)
    _dirty = False


def load_sessions():
    return _read()


def save_session(response_id, label):
    sessions = _read()
    sessions[:] = [s for s in sessions if not (s["label"] == label and s["cwd"] == CWD)]
    sessions.append({"id": response_id, "label": label[:80], "cwd": CWD, "ts": int(time.time())})
    global _dirty
    _dirty = True
    _flush()


def list_sessions():
    sessions = [s for s in _read() if s["cwd"] == CWD][-10:]
    if not sessions:
        print(_color(90, "no sessions in this directory"))
        return
    for s in reversed(sessions):
        age = int(time.time()) - s["ts"]
        age_label = f"{age // 60}m" if age < 3600 else f"{age // 3600}h" if age < 86400 else f"{age // 86400}d"
        print(f"{_color(36, s['id'][:12])}  {s['label']}  {_color(90, age_label + ' ago')}")
