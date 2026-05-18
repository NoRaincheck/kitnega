import json
import os
import sys
import threading
from urllib.request import Request, urlopen

from ._shared import _TTY, _color

API = os.getenv("CODY_API", "http://localhost:1234/v1/responses")
MODEL = os.getenv("CODY_MODEL", "qwen3.6-35b-a3b")


def api_key():
    return os.getenv("CODY_API_KEY") or ""


def _spinner(done, frames="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
    index = 0
    while not done.wait(0.1):
        print(f"\r{_color(90, frames[index % len(frames)] + ' thinking')}", end="", file=sys.stderr, flush=True)
        index += 1
    print("\r             \r", end="", file=sys.stderr, flush=True)


def respond(payload, system, tools, previous=None):
    body = {"model": MODEL, "instructions": system, "tools": tools, "input": payload}
    if previous:
        body["previous_response_id"] = previous
    headers = {"Content-Type": "application/json"}
    key = api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    spinner_done = threading.Event() if _TTY else None
    spinner_thread = threading.Thread(target=_spinner, args=(spinner_done,), daemon=True) if spinner_done else None
    if spinner_thread:
        spinner_thread.start()
    try:
        with urlopen(Request(API, json.dumps(body).encode(), headers=headers)) as r:
            return json.load(r)
    finally:
        if spinner_thread and (sp := spinner_done):
            sp.set()
            spinner_thread.join()


def text(response):
    return "".join(
        part.get("text", "")
        for item in response.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
