import sys

from ._shared import CWD, _color


def _get_run():
    from .tools import run

    return run


def _get_session():
    from .session import list_sessions, load_sessions, save_session

    return list_sessions, load_sessions, save_session


def repl(previous=None, label=None):
    from .session import list_sessions, load_sessions, save_session
    from .tools import run

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
        if previous:
            save_session(previous, label)
        if answer:
            print(answer)


def main():
    import argparse

    from .checkpoint import set_session_id

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
    parser.add_argument("--seed", action="store_true", help="use prompt as-is without refinement")
    parser.add_argument("prompt", nargs="*", help="command to execute (omitted for interactive mode)")

    args = parser.parse_args()
    previous, label = None, None

    if args.sessions:
        list_sessions, _, _ = _get_session()
        print(_color(90, "use :load <id> in the repl to resume a session"))
        list_sessions()
        return

    if args.cont:
        _, load_sessions, _ = _get_session()
        sessions = [s for s in load_sessions() if s.get("cwd") == CWD]
        if not sessions:
            sys.exit("no sessions in this directory")
        previous, label = sessions[-1]["id"], sessions[-1]["label"]
        print(_color(90, f"continuing: {label}"))

    prompt_text = " ".join(args.prompt)
    if prompt_text:
        from .tools import refine_prompt

        run = _get_run()
        _, _, save_session = _get_session()
        if not args.seed:
            print(_color(90, "refining prompt..."), file=sys.stderr)
            prompt_text = refine_prompt(prompt_text)
        from .checkpoint import set_session_id

        answer, response_id = run(prompt_text, previous)
        if response_id:
            set_session_id(response_id)
            save_session(response_id, label or prompt_text)
        if answer:
            sys.stdout.write(f"{answer}\n")
            sys.stdout.flush()
    else:
        repl(previous, label)


if __name__ == "__main__":
    main()
