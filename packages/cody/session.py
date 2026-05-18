import json
import os
import time

from ._shared import CWD, _color

SESSIONS = os.path.expanduser("~/.cody_sessions.json")


def load_sessions():
    try:
        return json.load(open(SESSIONS))
    except FileNotFoundError, json.JSONDecodeError:
        return []


def save_session(response_id, label):
    sessions = load_sessions()
    sessions = [s for s in sessions if not (s["label"] == label and s["cwd"] == CWD)]
    sessions.append({"id": response_id, "label": label[:80], "cwd": CWD, "ts": int(time.time())})
    json.dump(sessions[-50:], open(SESSIONS, "w"))


def list_sessions():
    sessions = [s for s in load_sessions() if s["cwd"] == CWD][-10:]
    if not sessions:
        print(_color(90, "no sessions in this directory"))
        return
    for s in reversed(sessions):
        age = int(time.time()) - s["ts"]
        age_label = f"{age // 60}m" if age < 3600 else f"{age // 3600}h" if age < 86400 else f"{age // 86400}d"
        print(f"{_color(36, s['id'][:12])}  {s['label']}  {_color(90, age_label + ' ago')}")
