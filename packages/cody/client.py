import json
import os
import sys
import threading
from urllib.request import Request, urlopen

from ._shared import _TTY, _color

API = os.getenv("CODY_API", "http://localhost:1234/v1/responses")
MODEL = os.getenv("CODY_MODEL", "qwen3.6-35b-a3b")
USE_STREAM = os.getenv("CODY_STREAM", "1") in ("1", "true", "yes", "")


def api_key():
    return os.getenv("CODY_API_KEY") or ""


_THINK_PHRASES = [
    "Expanding Horizons...",
    "Unloading Loading Screens...",
    "Mediating Modifiers...",
    "Reticulating 4-D Splines...",
    "Ascending Maslow's Hierarchy...",
    "Mapping the Llama Genome...",
    "Tabulating Traits...",
    "Calibrating Social Distance...",
    "Threading Fabric Compositors...",
    "Texture-Compositing Teddy Bears...",
]


def _spinner(done, step=0, frames="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
    index = 0
    phrase = _THINK_PHRASES[step % len(_THINK_PHRASES)]
    while not done.wait(0.1):
        print(f"\r{_color(90, frames[index % len(frames)] + ' ' + phrase)}", end="", file=sys.stderr, flush=True)
        index += 1
    # don't clear — next output (spinner or text) overwrites via \r


def _build_body(payload, system, tools, previous, stream):
    body = {"model": MODEL, "instructions": system, "tools": tools, "input": payload}
    if previous:
        body["previous_response_id"] = previous
    if stream:
        body["stream"] = True
    return body


def _headers():
    headers = {"Content-Type": "application/json"}
    key = api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _request(body, headers):
    return urlopen(Request(API, json.dumps(body).encode(), headers=headers))


def _sse_events(response):
    buf = b""
    while True:
        chunk = response.read(8192)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            raw, buf = buf.split(b"\n\n", 1)
            event_type = ""
            data_str = ""
            for line in raw.decode().split("\n"):
                if line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data_str = line[6:]
            if data_str:
                yield event_type, json.loads(data_str)


def _stream_respond(body, headers, spinner_done=None):
    had_content = False
    with _request(body, headers) as r:
        first_delta = True
        for event_type, data in _sse_events(r):
            if event_type == "response.output_text.delta":
                delta = data.get("delta", "")
                if delta:
                    had_content = True
                    if first_delta:
                        if spinner_done:
                            spinner_done.set()
                        print(file=sys.stderr, flush=True)
                        first_delta = False
                    print(_color(90, delta), end="", file=sys.stderr, flush=True)
            elif event_type in ("response.done", "response.completed"):
                return had_content, data.get("response", data)
    return had_content, body


def respond(payload, system, tools, previous=None, step=0):
    stream = USE_STREAM
    body = _build_body(payload, system, tools, previous, stream)
    headers = _headers()
    spinner_done = threading.Event() if _TTY else None
    spinner_thread = threading.Thread(target=_spinner, args=(spinner_done, step), daemon=True) if spinner_done else None
    if spinner_thread:
        spinner_thread.start()
    had_content = False
    try:
        if stream:
            had_content, resp = _stream_respond(body, headers, spinner_done)
            return resp
        with _request(body, headers) as r:
            return json.load(r)
    finally:
        if spinner_thread and (sp := spinner_done):
            sp.set()
            spinner_thread.join()
        if had_content:
            print(file=sys.stderr, flush=True)


def text(response):
    return "".join(
        part.get("text", "")
        for item in response.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
