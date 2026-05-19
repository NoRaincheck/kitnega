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


_SPINNER_TEXT = "Working..."


def _spinner(done, frames="⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"):
    idx = 0
    while not done.wait(0.1):
        print(f"\r{_color(90, frames[idx % len(frames)] + ' ' + _SPINNER_TEXT)}", end="", file=sys.stderr, flush=True)
        idx += 1


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


# Reasoning delta events — different providers name them differently.
_REASONING_EVENTS = frozenset(
    (
        "response.reasoning.delta",
        "reasoner_part.delta",
    )
)


def _stream_respond(body, headers, spinner_done=None):
    """Stream an SSE response, returning (had_content, response_dict).

    Reasoning deltas are printed to stdout in grey (once per contiguous
    reasoning block).  All other SSE events are accumulated into a response
    dict matching the non-streaming JSON shape.
    """
    had_content = False
    response = {}
    output_items = []
    item_map = {}
    in_reasoning = False
    started_output = False

    with _request(body, headers) as r:
        for event_type, data in _sse_events(r):
            delta_str = str(data.get("delta", ""))
            is_reasoning_event = event_type in _REASONING_EVENTS

            if delta_str:
                had_content = True

            # Reasoning → print in grey, prefixed with "> " once per block
            if is_reasoning_event and delta_str:
                if not in_reasoning:
                    if spinner_done:
                        spinner_done.set()
                    sys.stdout.write(_color(90, "> "))
                    in_reasoning = True
                sys.stdout.write(_color(90, delta_str))
                sys.stdout.flush()
                continue
            elif in_reasoning:
                in_reasoning = False

            # Accumulate the response object from SSE events
            if event_type == "response.created":
                response = data.get("response", {})
            elif event_type == "response.output_item.added":
                item = data.get("item", {})
                item_id = item.get("id")
                if item_id:
                    item_map[item_id] = item
                output_items.append(item)
            elif event_type == "response.content_part.added":
                part = data.get("part", {})
                item_id = data.get("item_id")
                if item_id in item_map:
                    item_map[item_id].setdefault("content", []).append(part)
            elif event_type == "response.output_text.delta":
                item_id = data.get("item_id")
                content_index = data.get("content_index", 0)
                delta = data.get("delta", "")
                if item_id in item_map:
                    content = item_map[item_id].setdefault("content", [])
                    while len(content) <= content_index:
                        content.append({"type": "output_text", "text": "", "annotations": []})
                    content[content_index]["text"] += delta
                if delta:
                    if not started_output:
                        if spinner_done:
                            spinner_done.set()
                        started_output = True
                    sys.stdout.write(delta)
                    sys.stdout.flush()
            elif event_type == "response.function_call_arguments.delta":
                item_id = data.get("item_id")
                delta = data.get("delta", "")
                if item_id in item_map:
                    item_map[item_id].setdefault("arguments", "")
                    item_map[item_id]["arguments"] += delta
            elif event_type == "response.function_call_arguments.done":
                item_id = data.get("item_id")
                arguments = data.get("arguments", "")
                if item_id in item_map:
                    item_map[item_id]["arguments"] = arguments
            elif event_type in ("response.completed", "response.done"):
                completed_resp = data.get("response", {})
                if completed_resp:
                    response = completed_resp
                break

    sys.stdout.write("\n")
    sys.stdout.flush()

    if not response.get("output"):
        response["output"] = output_items

    return had_content, response


def respond(payload, system, tools, previous=None):
    stream = USE_STREAM
    body = _build_body(payload, system, tools, previous, stream)
    headers = _headers()
    spinner_done = threading.Event() if _TTY else None
    spinner_thread = threading.Thread(target=_spinner, args=(spinner_done,), daemon=True) if spinner_done else None
    if spinner_thread:
        spinner_thread.start()
    try:
        return _stream_respond(body, headers, spinner_done)[1] if stream else json.load(_request(body, headers))
    finally:
        if spinner_thread:
            spinner_done.set()
            spinner_thread.join()


def text(response):
    """Extract visible output_text from a response's message chain."""
    return "".join(
        part.get("text", "")
        for item in response.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") == "output_text"
    )
