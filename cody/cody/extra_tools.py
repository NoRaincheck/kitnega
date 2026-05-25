"""Extra tools: glob file search with heavy-dir pruning.

Cody ships grep/find but not glob — a more intuitive pattern for small models.
This module provides a bounded glob walk that prunes node_modules, .git, and
other large directories by default.
"""

import fnmatch
import os

DEFAULT_IGNORE = frozenset([
    "node_modules", ".git", ".svn", ".hg", ".DS_Store",
    "__pycache__", ".tox", ".nox", ".venv", "venv",
    "dist", "build", ".next", ".nuxt", ".output",
])


def glob_walk(pattern, path=None, max_depth=20):
    """Walk a directory tree and return files matching a glob pattern.

    Args:
        pattern: Glob pattern (e.g., '*.py', 'src/**/*.ts')
        path: Directory to search in (defaults to cwd)
        max_depth: Maximum recursion depth

    Returns:
        Sorted list of relative file paths matching the pattern.
    """
    root = os.path.abspath(path or os.getcwd())
    if not os.path.isdir(root):
        return []

    results = []

    def _walk(current_dir, depth):
        if depth > max_depth:
            return
        try:
            entries = sorted(os.listdir(current_dir))
        except OSError:
            return

        for entry in entries:
            full_path = os.path.join(current_dir, entry)
            is_dir = os.path.isdir(full_path)

            # Prune ignored directories
            if is_dir and entry in DEFAULT_IGNORE:
                continue

            if is_dir:
                _walk(full_path, depth + 1)
            else:
                rel = os.path.relpath(full_path, root)
                if fnmatch.fnmatch(entry, pattern):
                    results.append(rel)

    _walk(root, 0)
    return sorted(results)


def handle_glob(args):
    """Handle a glob tool call. Returns formatted result string."""
    pattern = args.get("pattern", "")
    path = args.get("path")

    if not pattern:
        return "error: pattern is required"

    results = glob_walk(pattern, path)
    if not results:
        return f"glob: no files matching '{pattern}' in {path or '.'}"

    output = "\n".join(results)
    count = len(results)
    suffix = "" if count < 200 else " (truncated)"
    return f"{output}\n\n({count} file(s)){suffix}"
