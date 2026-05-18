import os
import sys

_TTY = sys.stderr.isatty()
CWD = os.getcwd()


def _color(code, text):
    return f"\033[{code}m{text}\033[0m" if _TTY else text
