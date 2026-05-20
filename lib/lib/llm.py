"""Prompt an LLM via an OpenAI-compatible /v1/responses API.

This module provides a simple ``prompt()`` function and a CLI entry point
for sending prompts to an LLM and retrieving the text response.

Example
-------
    >>> import os
    >>> os.environ["CODY_API"] = "http://localhost:1234/v1/responses"
    >>> from lib.llm import prompt
    >>> result = prompt("What is 2+2?")
    >>> print(result)
    4

Or from the command line:

    $ echo "What is 2+2?" | python -m lib.llm
    4
"""

import argparse
import json
import os
import sys
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
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Prompt an LLM and print the response")
    parser.add_argument("--system", "-s", default="You are a helpful assistant.", help="System prompt / instructions")
    parser.add_argument("--model", "-m", default=None, help="Model override")
    parser.add_argument("--timeout", "-t", type=int, default=300, help="Request timeout in seconds")
    parser.add_argument("input", nargs="*", help="Input text (read from stdin if omitted)")
    args = parser.parse_args()

    input_text = " ".join(args.input) if args.input else sys.stdin.read().strip()
    if not input_text:
        parser.print_usage()
        sys.exit(1)

    try:
        result = prompt(input_text, system=args.system, model=args.model, timeout=args.timeout)
        print(result)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
