import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from ._shared import CWD, _color
from .checkpoint import create_checkpoint
from .client import USE_STREAM, respond, text
from .extra_tools import handle_glob
from .handlers import _handle_bash, _handle_edit, _handle_find, _handle_grep, _handle_ls, _handle_read, _handle_write
from .permission_gate import gate_command
from .quality_monitor import assess_response, build_correction_message, phrase_for_user
from .read_guard import trim_result
from .system_prompt import build_system_prompt
from .turn_cap import check_turn_cap, get_turn_cap
from .write_guard import guard_edit, guard_write

MAX_STEPS = int(os.getenv("KN_MAX_STEPS", "20"))
APPROVE_ALL = os.getenv("KN_APPROVE", "all").lower() == "all"
_TURN_CAP = get_turn_cap()
_READONLY_TOOLS = frozenset({"read", "grep", "find", "ls", "glob"})
_RESULT_CACHE = {}

# Track recent tool calls for quality monitoring
_recent_tool_calls = []


def _tool_def(name, desc, props, required):
    return {
        "type": "function",
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required, "additionalProperties": False},
    }


TOOL_SNIPPETS = {
    "read": "Read a file's content (truncated to 30 lines by default — use Grep first for large files)",
    "write": "Create or overwrite an entire file with given text",
    "edit": "Replace one exact old_string match in a file with new_string",
    "bash": "Run a shell command and capture output (with optional timeout/cwd/env; default 30s timeout)",
    "grep": "Search files for lines matching a regex pattern, optionally filtering by glob",
    "find": "Find files/dirs matching a glob pattern within a directory tree",
    "ls": "List sorted entries in a directory, marking subdirs with /",
    "glob": "Glob file search with heavy-dir pruning (node_modules, .git, __pycache__ excluded)",
}

_PROP_DESC = {
    "path": "Target file or directory path",
    "content": "Full file content to write",
    "old_string": "Exact string to match (must appear exactly once)",
    "new_string": "Replacement string",
    "command": "Shell command to execute",
    "description": "Short human-readable summary of what the command does",
    "cwd": "Working directory (defaults to project root)",
    "timeout": "Timeout in seconds (default 60)",
    "pattern": "Search pattern",
    "include": "Glob filter (e.g. *.py)",
    "offset": "Starting line number, 1-indexed",
    "limit": "Maximum lines to return",
}


def _prop(name, t):
    return {**{"type": t}, "description": _PROP_DESC[name]}


TOOLS = [
    _tool_def(
        "read",
        TOOL_SNIPPETS["read"] + ".",
        {"path": _prop("path", "string"), "offset": _prop("offset", "integer"), "limit": _prop("limit", "integer")},
        ["path"],
    ),
    _tool_def(
        "write",
        TOOL_SNIPPETS["write"] + ".",
        {"path": _prop("path", "string"), "content": _prop("content", "string")},
        ["path", "content"],
    ),
    _tool_def(
        "edit",
        TOOL_SNIPPETS["edit"] + ".",
        {
            "path": _prop("path", "string"),
            "old_string": _prop("old_string", "string"),
            "new_string": _prop("new_string", "string"),
        },
        ["path", "old_string", "new_string"],
    ),
    _tool_def(
        "bash",
        TOOL_SNIPPETS["bash"] + ".",
        {
            "command": _prop("command", "string"),
            "description": _prop("description", "string"),
            "cwd": _prop("cwd", ["string", "null"]),
            "timeout": _prop("timeout", "integer"),
        },
        ["command", "description"],
    ),
    _tool_def(
        "grep",
        TOOL_SNIPPETS["grep"] + ".",
        {
            "pattern": _prop("pattern", "string"),
            "include": _prop("include", "string"),
            "path": _prop("path", "string"),
        },
        ["pattern"],
    ),
    _tool_def(
        "find",
        TOOL_SNIPPETS["find"] + ".",
        {"pattern": _prop("pattern", "string"), "path": _prop("path", "string")},
        ["pattern"],
    ),
    _tool_def("ls", TOOL_SNIPPETS["ls"] + ".", {"path": _prop("path", "string")}, []),
    _tool_def(
        "glob",
        TOOL_SNIPPETS["glob"] + ".",
        {"pattern": _prop("pattern", "string"), "path": _prop("path", "string")},
        ["pattern"],
    ),
]


def approve(args, requires_approval):
    if not requires_approval:
        return True
    global APPROVE_ALL
    for k, v in {k: v for k, v in args.items() if k != "description" and v is not None}.items():
        val = v if len(str(v)) < 80 else str(v)[:77] + "..."
        print(f"{_color(32, k)}: {_color(90, val)}", file=sys.stderr)
    if APPROVE_ALL:
        return True
    try:
        choice = input("Approve? [y]yes  [a]all  [n]no: ").strip().lower()
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
    "glob": handle_glob,
}


def _cache_key(call):
    args = json.loads(call.get("arguments") or "{}")
    return (call["name"], json.dumps(args, sort_keys=True))


def tool_output(call):
    handler = TOOL_HANDLERS.get(call["name"])
    if not handler:
        sys.stderr.write(_color(31, f"[{call['name']}] unknown tool\n"))
        return {"type": "function_call_output", "call_id": call["call_id"], "output": "unknown"}

    is_readonly = call["name"] in _READONLY_TOOLS
    if is_readonly:
        key = _cache_key(call)
        cached = _RESULT_CACHE.get(key)
        if cached is not None:
            return {"type": "function_call_output", "call_id": call["call_id"], "output": cached}

    args = {}
    key = None
    try:
        args = json.loads(call.get("arguments") or "{}")
    except json.JSONDecodeError as e:
        sys.stderr.write(_color(31, f"[{call['name']}] bad arguments: {e}\n"))
        result = f"bad arguments: {e}"
    else:
        # Permission gate for bash
        if call["name"] == "bash":
            words = args.get("description", "").split()
            if len(words) == 0:
                sys.stderr.write(_color(31, f"[{call['name']}] description must not be empty\n"))
                return {
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": "bad arguments: description must not be empty",
                }
            block = gate_command(args.get("command", ""))
            if block:
                return {"type": "function_call_output", "call_id": call["call_id"], "output": block}

        # Write guard for write operations (normalize + refuse existing)
        if call["name"] == "write":
            safe_path, err = guard_write(args)
            if err:
                return {"type": "function_call_output", "call_id": call["call_id"], "output": err}
            args["path"] = safe_path

        # Write guard for edit operations (normalize + verify exists)
        if call["name"] == "edit":
            safe_path, err = guard_edit(args)
            if err:
                return {"type": "function_call_output", "call_id": call["call_id"], "output": err}
            args["path"] = safe_path

        # Checkpoint before modifications
        if call["name"] in ("write", "edit") and isinstance(args.get("path"), str):
            create_checkpoint(args["path"])

        result = handler(args) if approve(args, call["name"] in NEEDS_APPROVAL) else _color(31, "denied by user")

    # Read guard: trim oversized results
    if is_readonly and isinstance(result, str) and key:
        path_arg = args.get("path", "")
        result = trim_result(result, path_arg)

    if is_readonly and key:
        _RESULT_CACHE[key] = result
    else:
        _RESULT_CACHE.clear()

    return {"type": "function_call_output", "call_id": call["call_id"], "output": result}


def _execute_calls(calls):
    groups = {}
    for c in calls:
        key = _cache_key(c)
        groups.setdefault(key, []).append(c)

    uniques = [g[0] for g in groups.values()]
    if APPROVE_ALL and len(uniques) > 1:
        by_id = {}
        with ThreadPoolExecutor() as pool:
            fmap = {pool.submit(tool_output, c): c["call_id"] for c in uniques}
            for f in as_completed(fmap):
                by_id[fmap[f]] = f.result()
        first_results = by_id
    else:
        first_results = {c["call_id"]: tool_output(c) for c in uniques}

    result_by_key = {}
    for c in uniques:
        result_by_key[_cache_key(c)] = first_results[c["call_id"]]

    return [{**result_by_key[_cache_key(c)], "call_id": c["call_id"]} for c in calls]


REFINE_SYSTEM = (
    "You are a prompt refinement engine. Your only job is to take a user's "
    "request and expand it into a clear, detailed set of instructions for a "
    "coding agent. Be specific. Include what files might need to be read, "
    "what commands might need to be run, and what changes might need to be made. "
    "Output only the refined instructions, no preamble or commentary."
)


def refine_prompt(prompt):
    response = respond(prompt, REFINE_SYSTEM, [], previous=None, stream=False)
    return text(response)


def run(prompt, previous=None):
    global _recent_tool_calls
    tool_names = [t["name"] for t in TOOLS]
    system = build_system_prompt(cwd=CWD, selected_tools=tool_names, tool_snippets=TOOL_SNIPPETS)
    response = respond(prompt, system, TOOLS, previous)
    recent_text = ""

    for turn_idx in range(MAX_STEPS):
        # Turn cap check
        if _TURN_CAP and check_turn_cap(turn_idx, _TURN_CAP):
            sys.stderr.write(_color(31, f"\nTurn limit ({_TURN_CAP}) reached. Stopping.\n"))
            return "", response.get("id", "")

        calls = [x for x in response.get("output", []) if x.get("type") == "function_call"]
        if not calls:
            recent_text = text(response)
            if USE_STREAM:
                return "", response.get("id", "")
            return text(response), response.get("id", "")

        # Quality monitoring: assess before executing
        known_tools = set(tool_names)
        quality = assess_response(recent_text, calls, _recent_tool_calls, known_tools)
        if not quality["ok"]:
            correction = build_correction_message(quality["reason"])
            sys.stderr.write(_color(31, f"\nharness intervention: {phrase_for_user(quality['reason'])}\n"))
            # Send correction back to model as a system message
            response = respond(
                None,
                system + "\n\nCorrection: " + correction,
                TOOLS,
                response.get("id"),
            )
            continue  # don't execute tool calls, just let the model try again

        sys.stderr.write("\r\033[K")
        for call in calls:
            args = json.loads(call.get("arguments") or "{}")
            params = ", ".join(
                f"{k}={str(v) if len(str(v)) < 60 else str(v)[:57] + '...'}" for k, v in args.items() if v is not None
            )
            sys.stderr.write(_color(90, f"» {call['name']}({params})\n"))
        sys.stderr.flush()
        sys.stdout.flush()

        # Track tool calls for quality monitoring
        _recent_tool_calls = [
            {"name": c["name"], "arguments": c.get("arguments", "{}")} for c in calls
        ]

        payload = _execute_calls(calls) if calls else None
        response = respond(payload, system, TOOLS, response.get("id"))
