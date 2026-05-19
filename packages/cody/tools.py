import json, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from ._shared import CWD, _color
from .client import respond, text
from .handlers import (_handle_bash, _handle_edit, _handle_find, _handle_grep,
                       _handle_ls, _handle_read, _handle_write)
from .system_prompt import build_system_prompt

MAX_STEPS = int(os.getenv("CODY_MAX_STEPS", "200"))
APPROVE_ALL = os.getenv("CODY_APPROVE", "all").lower() == "all"


def _tool_def(name, desc, props, required):
    return {
        "type": "function",
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": required,
                       "additionalProperties": False},
    }


TOOL_SNIPPETS = {
    "read": "Read a file's content, optionally limiting lines (1-indexed)",
    "write": "Create or overwrite an entire file with given text",
    "edit": "Replace one exact old_string match in a file with new_string",
    "bash": "Run a shell command and capture output (with optional timeout/cwd/env)",
    "grep": "Search files for lines matching a regex pattern, optionally filtering by glob",
    "find": "Find files/dirs matching a glob pattern within a directory tree",
    "ls":  "List sorted entries in a directory, marking subdirs with /",
}

TOOLS = [
    _tool_def("read", TOOL_SNIPPETS["read"] + ".",
              {"path": {"type": "string"},
               "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"]),
    _tool_def("write", TOOL_SNIPPETS["write"] + ".",
              {"path": {"type": "string"}, "content": {"type": "string"}}, ["path","content"]),
    _tool_def("edit", TOOL_SNIPPETS["edit"] + ".",
              {"path":{"type":"string"},"old_string":{"type":"string"},"new_string":{"type":"string"}},
              ["path","old_string","new_string"]),
    _tool_def("bash", TOOL_SNIPPETS["bash"] + ".",
              {"command": {"type":"string"}, "description": {"type":"string"},
               "cwd": {"type":["string","null"]}, "timeout":{"type":"integer"}},
              ["command","description"]),
    _tool_def("grep", TOOL_SNIPPETS["grep"] + ".",
              {"pattern":{"type":"string"},"include":{"type":"string"},"path":{"type":"string"}},
              ["pattern"]),
    _tool_def("find", TOOL_SNIPPETS["find"] + ".",
              {"pattern":{"type":"string"},"path":{"type":"string"}}, ["pattern"]),
    _tool_def("ls", TOOL_SNIPPETS["ls"] + ".",
              {"path": {"type":"string"}}, []),
]


def approve(args, requires_approval):
    if not requires_approval: return True
    global APPROVE_ALL
    desc = args.get("description","")
    if desc: print(f"\n{_color(90,'# '+desc)}",file=sys.stderr)
    for k,v in {k:v for k,v in args.items() if k!="description" and v is not None}.items():
        val=v if len(str(v))<80 else str(v)[:77]+"..."
        print(f"{_color(32,k)}: {_color(90,val)}",file=sys.stderr)
    if APPROVE_ALL: return True
    try: choice=input("Approve? [y]yes  [a]all  [n]no: ").strip().lower()
    except EOFError: return False
    if choice in ("a","all"): APPROVE_ALL=True; return True
    return choice in ("y","yes")


NEEDS_APPROVAL = {"write","edit","bash"}

TOOL_HANDLERS={"read":_handle_read,"write":_handle_write,"edit":_handle_edit,
               "bash":_handle_bash,"grep":_handle_grep,"find":_handle_find,"ls":_handle_ls}


def tool_output(call):
    handler = TOOL_HANDLERS.get(call["name"])
    if not handler: return {"type":"function_call_output","call_id":call["call_id"],"output":"unknown"}
    try: args=json.loads(call.get("arguments") or "{}")
    except json.JSONDecodeError as e: result=f"bad arguments: {e}"
    else:
        if call["name"]=="bash":
            words=args.get("description","").split()
            if not 5<=len(words)<=10: return {"type":"function_call_output",
                "call_id":call["call_id"],"output":"bad arguments: description must be 5-10 words"}
        result=handler(args) if approve(args,call["name"] in NEEDS_APPROVAL)\
            else _color(31,"denied by user")
    return {"type":"function_call_output","call_id":call["call_id"],"output":result}


def _execute_calls(calls):
    if APPROVE_ALL and len(calls)>1:
        by_id={}
        with ThreadPoolExecutor() as pool:
            fmap={pool.submit(tool_output,c):c["call_id"] for c in calls}
            for f in as_completed(fmap): by_id[fmap[f]]=f.result()
        return [by_id[c["call_id"]]for c in calls]
    return[tool_output(call)for call in calls]


def run(prompt, previous=None):
    tool_names=[t["name"] for t in TOOLS]
    system=build_system_prompt(cwd=CWD,selected_tools=tool_names,tool_snippets=TOOL_SNIPPETS)
    response = respond(prompt,system,TOOLS,previous)
    for _round in range(MAX_STEPS):
        calls=[x for x in response.get("output",[])if x.get("type")=="function_call"]
        if not calls: return text(response),response.get("id","")
        payload=_execute_calls(calls) if calls else None
        response = respond(payload, system, TOOLS, response.get("id"))

