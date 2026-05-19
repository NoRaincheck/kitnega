import fnmatch
import glob
import os
import re
import subprocess
from pathlib import Path

from ._shared import CWD

SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", "venv", ".ruff_cache"}
_GITIGNORE_CACHE = {}
_GITIGNORE_MTIME = {}


def _load_dir_skip(root):
    """Return a set of directory names to skip — built-in SKIP_DIRS plus .gitignore rules.

    For each `.gitignore` found (root + any parent up to ~), extract the
    bare directory-name patterns (e.g. `build/`, `dist`, `.next/`) so callers
    can skip them during `os.walk` directory pruning.
    """
    ignored = set()
    start = Path(root).resolve()
    for parent in [start, *start.parents]:
        gi = parent / ".gitignore"
        try:
            mtime = gi.stat().st_mtime
        except OSError:
            continue
        if _GITIGNORE_CACHE.get(gi) is not None and _GITIGNORE_MTIME.get(gi) == mtime:
            ignored.update(_GITIGNORE_CACHE[gi])
            continue
        patterns = set()
        try:
            for line in gi.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("!"):
                    continue
                name = line.rstrip("/")
                patterns.add(name)
        except Exception:
            pass
        _GITIGNORE_CACHE[gi] = patterns
        _GITIGNORE_MTIME[gi] = mtime
        ignored.update(patterns)
    return SKIP_DIRS | ignored


def _handle_read(args):
    path = args["path"]
    if not os.path.isfile(path):
        return f"error: file not found: {path}"
    offset = args.get("offset", 1)
    limit = args.get("limit")
    try:
        with open(path) as f:
            lines = f.readlines()
    except Exception as e:
        return f"error: {e}"
    total = len(lines)
    start = max(0, (offset or 1) - 1)
    end = min(total, start + (limit or total))
    selected = lines[start:end]
    result = "".join(selected)
    if limit and end < total:
        result += f"\n... ({total - end} more lines)"
    return f"--- {path} ({start + 1}-{end}/{total} lines) ---\n{result}"


def _handle_write(args):
    path = args["path"]
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(args["content"])
        return f"wrote {len(args['content'])} bytes to {path}"
    except Exception as e:
        return f"error: {e}"


def _handle_edit(args):
    path = args["path"]
    old = args["old_string"]
    new = args["new_string"]
    if not os.path.isfile(path):
        return f"error: file not found: {path}"
    try:
        with open(path) as f:
            content = f.read()
        count = content.count(old)
        if count == 0:
            return f"error: old_string not found in {path}"
        if count > 1:
            return f"error: {count} matches found in {path}, need exactly 1"
        with open(path, "w") as f:
            f.write(content.replace(old, new, 1))
        return f"edited {path}: replaced {len(old)} chars with {len(new)} chars"
    except Exception as e:
        return f"error: {e}"


def _handle_bash(args):
    command = args["command"]
    cwd = args.get("cwd")
    timeout = args.get("timeout", 60)
    env = args.get("env")
    run_env = {**os.environ, **(env or {})}
    try:
        process = subprocess.run(
            command,
            shell=True,
            cwd=os.path.abspath(cwd or CWD),
            env=run_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        return f"$ {command}\nexit {process.returncode}\n{process.stdout}"[-12000:]
    except subprocess.TimeoutExpired as error:
        return f"$ {command}\ntimeout after {timeout}s\n{error.stdout or ''}"[-12000:]
    except Exception as error:
        return f"{type(error).__name__}: {error}"


def _handle_grep(args):
    pattern = args["pattern"]
    include = args.get("include")
    root = args.get("path") or CWD
    if not os.path.isdir(root):
        return f"error: directory not found: {root}"
    skip = _load_dir_skip(root)
    matches = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in skip]
        for file in files:
            if include and not fnmatch.fnmatch(file, include):
                continue
            filepath = os.path.join(base, file)
            try:
                with open(filepath, errors="replace") as f:
                    for i, line in enumerate(f, 1):
                        if re.search(pattern, line):
                            rel = os.path.relpath(filepath, root)
                            matches.append(f"{rel}:{i}:{line.rstrip()}")
            except Exception:
                continue
            if len(matches) >= 200:
                break
        if len(matches) >= 200:
            break
    if not matches:
        return f"grep: no matches for {pattern} in {root}"
    return "\n".join(matches)


def _handle_find(args):
    pattern = args["pattern"]
    root = args.get("path") or CWD
    if not os.path.isdir(root):
        return f"error: directory not found: {root}"
    skip = _load_dir_skip(root)
    matches = []
    for entry in glob.glob(pattern, root_dir=root, recursive=True):
        parts = entry.replace(os.sep, "/").split("/")
        if any(p in skip for p in parts):
            continue
        matches.append(entry)
        if len(matches) >= 200:
            break
    if not matches:
        return f"find: no files matching {pattern} in {root}"
    return "\n".join(sorted(matches))


def _handle_ls(args):
    path = args.get("path") or CWD
    if not os.path.isdir(path):
        return f"error: directory not found: {path}"
    try:
        entries = os.listdir(path)
    except Exception as e:
        return f"error: {e}"
    if not entries:
        return f"{path}: empty"
    result = []
    for entry in sorted(entries):
        full = os.path.join(path, entry)
        suffix = "/" if os.path.isdir(full) else ""
        result.append(f"{entry}{suffix}")
    return "\n".join(result)
