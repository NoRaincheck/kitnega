import os
import sys

TTY = sys.stderr.isatty()


def ansi(n: int) -> str:
    return f"\033[{n}m" if TTY else ""


def cwd() -> str:
    return os.getcwd()
