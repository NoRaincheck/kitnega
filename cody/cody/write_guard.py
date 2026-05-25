"""Write guard: refuse writes on existing files and normalize bare paths.

Small models frequently try to overwrite existing files instead of using Edit.
This module intercepts write operations to (1) normalize root-bare paths like
`/foo.md` into relative paths, and (2) refuse writes when the target file
already exists — suggesting Edit with exact old_string/new_string instead.
"""

import os


def normalize_path(path):
    """Strip leading / so '/foo.md' becomes '<cwd>/foo.md'.

    Small models often write bare paths like '/foo.md' meaning 'file foo.md in
    cwd'. We detect this by checking if the absolute path doesn't exist — if it
    doesn't, we treat it as a bare path and normalize to cwd-relative.
    """
    if not isinstance(path, str) or not path:
        return path
    # Relative paths starting with . — leave alone
    if path[0] == ".":
        return path
    # Paths starting with / — check if it's a real absolute path
    if path.startswith("/"):
        if os.path.isabs(path) and os.path.exists(path):
            return path  # real filesystem path, leave alone
        # Bare path (/foo.md) that doesn't exist on disk — normalize to cwd
        cleaned = path.lstrip("/")
        return os.path.join(os.getcwd(), cleaned)
    # Relative paths without leading . — pass through unchanged
    return path


def guard_write(args):
    """Validate a write call. Returns (safe_path, error_or_None).

    If the file already exists, returns an Edit-suggestion error instead of
    allowing the overwrite. Also normalizes bare paths.
    """
    path = args.get("path", "")
    if not isinstance(path, str) or not path:
        return None, "error: empty path"

    # Normalize root-bare paths
    safe_path = normalize_path(path)

    if os.path.isfile(safe_path):
        return (
            None,
            f'Write refuses on existing file "{path}". '
            f"Use Edit with exact old_string / new_string to modify it. "
            f"Read the file first for line numbers and precision.",
        )

    return safe_path, None


def guard_edit(args):
    """Validate an edit call path (normalize bare paths). Returns (safe_path, error_or_None)."""
    path = args.get("path", "")
    if not isinstance(path, str) or not path:
        return None, "error: empty path"

    safe_path = normalize_path(path)

    if not os.path.isfile(safe_path):
        return None, f"error: file not found: {safe_path}"

    return safe_path, None
