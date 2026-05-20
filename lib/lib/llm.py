import json
import os
from urllib.request import Request, urlopen

API = os.getenv("CODY_API", "http://localhost:1234/v1/responses")
MODEL = os.getenv("CODY_MODEL", "qwen3.6-35b-a3b")


def _api_key():
    return os.getenv("CODY_API_KEY") or ""


def _headers():
    headers = {"Content-Type": "application/json"}
    key = _api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return headers


def prompt(input_text, system="", model=None, timeout=300):
    body = {
        "model": model or MODEL,
        "instructions": system,
        "input": input_text,
    }
    resp = urlopen(
        Request(API, json.dumps(body).encode(), _headers()),
        timeout=timeout,
    )
    data = json.load(resp)
    return _text(data)


def _text(data):
    text = ""
    for item in data.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text += part.get("text", "")
    return text
