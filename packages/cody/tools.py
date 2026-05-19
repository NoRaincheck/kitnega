import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from ._shared import CWD, _color
from .client import respond, text
from .handlers import _handle_bash, _handle_edit, _handle_find, _handle_grep, _handle_ls, _handle_read, _handle_write
from .system_prompt import build_system_prompt

MAX_STEPS = int(os.getenv("CODY_MAX_STEPS", "200"))
APPROVE_ALL = os.getenv("CODY_APPROVE", "all").lower() == "all"
_PARALLEL_TOOLS = APPROVE_ALL


def _tool_def(name, desc, props, required):
    return {
        "type": "function",
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required, "additionalProperties": False},
    }


TOOLS = [
    _tool_def(
        "read",
        "Read file contents with optional line range.",
        {
            "path": {"type": "string", "description": "Path to the file"},
            "offset": {"type": "integer", "description": "Starting line (1-indexed), default 1"},
            "limit": {"type": "integer", "description": "Maximum lines to read"},
        },
        ["path"],
    ),
    _tool_def(
        "write",
        "Create or overwrite a file with content.",
        {
            "path": {"type": "string", "description": "Path to the file"},
            "content": {"type": "string", "description": "Content to write"},
        },
        ["path", "content"],
    ),
    _tool_def(
        "edit",
        "Replace text in a file using exact string matching.",
        {
            "path": {"type": "string", "description": "Path to the file"},
            "old_string": {"type": "string", "description": "Exact text to find and replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
        },
        ["path", "old_string", "new_string"],
    ),
    _tool_def(
        "bash",
        "Run a shell command with inherited environment.",
        {
            "command": {"type": "string"},
            "description": {"type": "string", "description": "Why this command is useful, 5-10 words."},
            "cwd": {"type": ["string", "null"]},
            "timeout": {"type": "integer"},
            "env": {"type": "object", "additionalProperties": {"type": "string"}},
        },
        ["command", "description"],
    ),
    _tool_def(
        "grep",
        "Search file contents using a regex pattern.",
        {
            "pattern": {"type": "string", "description": "Regex pattern to search for"},
            "include": {"type": "string", "description": "File glob pattern (e.g., *.py, *.txt)"},
            "path": {"type": "string", "description": "Directory to search in"},
        },
        ["pattern"],
    ),
    _tool_def(
        "find",
        "Find files matching a glob pattern.",
        {
            "pattern": {"type": "string", "description": "Glob pattern (e.g., **/*.py)"},
            "path": {"type": "string", "description": "Directory to search in"},
        },
        ["pattern"],
    ),
    _tool_def(
        "ls",
        "List files and directories in a path.",
        {
            "path": {"type": "string", "description": "Directory to list"},
        },
        [],
    ),
]


TOOL_SNIPPETS = {t["name"]: t["description"].rstrip(".") for t in TOOLS}


def approve(args, requires_approval):
    if not requires_approval:
        return True
    global APPROVE_ALL
    desc = args.get("description", "")
    if desc:
        print(f"\n{_color(90, '# ' + desc)}", file=sys.stderr)
    display_args = {k: v for k, v in args.items() if k != "description" and v is not None}
    for key, value in display_args.items():
        val = value if len(str(value)) < 80 else str(value)[:77] + "..."
        print(f"{_color(32, key)}: {_color(90, val)}", file=sys.stderr)
    if APPROVE_ALL:
        return True
    try:
        choice = (
            input(f"Approve? {_color(32, '[y] Approve')}  {_color(33, '[a] Approve All')}  {_color(31, '[n] Deny')}: ")
            .strip()
            .lower()
        )
    except EOFError:
        return False
    if choice in ("a", "all"):
        APPROVE_ALL = True
        return True
    return choice in ("y", "yes")


NEEDS_APPROVAL = {"write", "edit", "bash"}

TOOL_HANDLERS = {
    "read": _handle_read,
    "write": _handle_write,
    "edit": _handle_edit,
    "bash": _handle_bash,
    "grep": _handle_grep,
    "find": _handle_find,
    "ls": _handle_ls,
}


def tool_output(call):
    handler = TOOL_HANDLERS.get(call["name"])
    if not handler:
        result = "unknown tool"
    else:
        try:
            args = json.loads(call.get("arguments") or "{}")
        except json.JSONDecodeError as error:
            result = f"bad arguments: {error}"
        else:
            if call["name"] == "bash":
                words = args.get("description", "").split()
                if not 5 <= len(words) <= 10:
                    result = "bad arguments: description must be 5-10 words"
                    return {"type": "function_call_output", "call_id": call["call_id"], "output": result}
            result = handler(args) if approve(args, call["name"] in NEEDS_APPROVAL) else _color(31, "denied by user")
    return {"type": "function_call_output", "call_id": call["call_id"], "output": result}


def _execute_calls(calls):
    if _PARALLEL_TOOLS and len(calls) > 1:
        by_id = {}
        with ThreadPoolExecutor() as pool:
            fut_map = {pool.submit(tool_output, c): c["call_id"] for c in calls}
            for fut in as_completed(fut_map):
                by_id[fut_map[fut]] = fut.result()
        return [by_id[c["call_id"]] for c in calls]
    return [tool_output(call) for call in calls]


def _session_id(response):
    sid = response.get("id")
    return sid if isinstance(sid, str) and sid else None


def run(prompt, previous=None):
    tool_names = [t["name"] for t in TOOLS]
    system = build_system_prompt(cwd=CWD, selected_tools=tool_names, tool_snippets=TOOL_SNIPPETS)
    response = respond(prompt, system, TOOLS, previous, step=0)
    for step in range(1, MAX_STEPS + 1):
        calls = [x for x in response.get("output", []) if x.get("type") == "function_call"]
        if not calls:
            return text(response), _session_id(response)
        response = respond(_execute_calls(calls), system, TOOLS, _session_id(response), step=step)
    return "stopped: too many tool calls", _session_id(response)
