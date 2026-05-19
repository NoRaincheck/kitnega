import json, os, sys, threading
from urllib.request import Request, urlopen
from ._shared import _TTY, _color

API = os.getenv("CODY_API", "http://localhost:1234/v1/responses")
MODEL = os.getenv("CODY_MODEL", "qwen3.6-35b-a3b")
USE_STREAM = os.getenv("CODY_STREAM", "1") in ("1", "true", "yes", "")


def api_key():
    return os.getenv("CODY_API_KEY") or ""

# Single consistent spinner text for multi-round sessions — no phase cycling.
_SPINNER_TEXT = "Working..."


def _spinner(done, frames="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
    index = 0
    while not done.wait(0.1):
        print(f"\r{_color(90, frames[index % len(frames)] + ' ' + _SPINNER_TEXT)}", end="", file=sys.stderr, flush=True)
        index += 1


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
                        spinner_done and spinner_done.set()
                        print(file=sys.stderr, flush=True)
                        first_delta = False
                    print(_color(90, delta), end="", file=sys.stderr, flush=True)
            elif event_type in ("response.done", "response.completed"):
                return had_content, data.get("response", data)
    return had_content, body


def respond(payload, system, tools, previous=None):
    stream = USE_STREAM
    body = _build_body(payload, system, tools, previous, stream)
    headers = _headers()
    spinner_done = threading.Event() if _TTY else None
    thread_args = (spinner_done,) if spinner_done else ()
    spinner_thread = threading.Thread(target=_spinner, args=thread_args, daemon=True) if spinner_done else None
    if spinner_thread:
        spinner_thread.start()
    had_content = False
    try:
        return _stream_respond(body, headers, spinner_done)[1] if stream \
            else json.load(_request(body, headers))
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
