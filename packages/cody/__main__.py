import argparse
import sys

from ._shared import CWD, _color
from .session import list_sessions, load_sessions, save_session
from .tools import run


def repl(previous=None, label=None):
    print(_color(1, "cody") + " repl " + _color(90, "(:quit, :reset, :load)"))
    while True:
        try:
            prompt = input(_color(36, "cody > ")).strip()
        except EOFError, KeyboardInterrupt:
            print()
            return
        if not prompt:
            continue
        if prompt.lower() in (":q", ":quit"):
            return
        if prompt.lower() in (":reset", "reset"):
            previous, label = None, None
            print(_color(90, "reset"))
            continue
        if prompt.lower() == ":load":
            list_sessions()
            continue
        if prompt.lower().startswith(":load "):
            target_id = prompt[6:].strip()
            sessions = [s for s in load_sessions() if s.get("cwd") == CWD]
            match = next((s for s in sessions if s["id"].startswith(target_id)), None)
            if not match:
                print(_color(31, f"no session matching '{target_id}'"))
                list_sessions()
                continue
            previous, label = match["id"], match["label"]
            print(_color(90, f"loaded: {label}"))
            continue
        answer, previous = run(prompt, previous)
        if not label:
            label = prompt
        save_session(previous, label)
        print(answer)


def main():
    parser = argparse.ArgumentParser(description="cody — a minimal coding agent")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-s", "--sessions", action="store_true", help="list available sessions and start repl")
    group.add_argument(
        "-c",
        "--continue",
        dest="cont",
        action="store_true",
        help="load last session in current directory, then run prompt or enter repl",
    )
    parser.add_argument("prompt", nargs="*", help="command to execute (omitted for interactive mode)")

    args = parser.parse_args()
    previous, label = None, None

    if args.sessions:
        print(_color(90, "use :load <id> in the repl to resume a session"))
        list_sessions()
    elif args.cont:
        sessions = [s for s in load_sessions() if s.get("cwd") == CWD]
        if not sessions:
            sys.exit("no sessions in this directory")
        previous, label = sessions[-1]["id"], sessions[-1]["label"]
        print(_color(90, f"continuing: {label}"))

    prompt_text = " ".join(args.prompt)
    if prompt_text:
        answer, response_id = run(prompt_text, previous)
        save_session(response_id, label or prompt_text)
        print(answer)
    else:
        repl(previous, label)


if __name__ == "__main__":
    main()
