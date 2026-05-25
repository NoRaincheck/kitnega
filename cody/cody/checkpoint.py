"""Checkpoint: backup files before Write/Edit operations.

Small models can make mistakes. This module backs up files to a session-scoped
checkpoint directory (~/.kitnega/checkpoints/) before any modification, providing
a best-effort undo safety net.
"""

import os
import shutil
import time

CHECKPOINT_DIR = os.path.expanduser("~/.kitnega/checkpoints")
_checkpoint_session = None


def _get_session_dir():
    """Get (or create) the session-scoped checkpoint directory."""
    global _checkpoint_session
    if _checkpoint_session is not None:
        return _checkpoint_session

    # Use timestamp + CWD hash for uniqueness
    cwd_hash = hex(abs(hash(os.getcwd())))[-8:]
    ts = time.strftime("%Y%m%d_%H%M%S")
    session_dir = os.path.join(CHECKPOINT_DIR, f"{ts}_{cwd_hash}")
    try:
        os.makedirs(session_dir, exist_ok=True)
    except OSError:
        return None

    _checkpoint_session = session_dir
    return session_dir


def create_checkpoint(file_path):
    """Back up a file before modification. Best-effort — never raises."""
    if not isinstance(file_path, str) or not file_path:
        return

    # Normalize bare paths
    safe_path = file_path.lstrip("/")
    full_path = os.path.abspath(safe_path)

    if not os.path.isfile(full_path):
        return

    try:
        session_dir = _get_session_dir()
        if session_dir is None:
            return

        # Sanitize filename for filesystem safety
        safe_name = os.path.basename(full_path).replace(" ", "_")
        checkpoint_path = os.path.join(session_dir, f"{safe_name}.bak")

        # Create subdirectory structure within checkpoint dir
        rel = os.path.relpath(full_path, os.getcwd())
        subdir = os.path.dirname(rel)
        if subdir:
            checkpoint_path = os.path.join(session_dir, subdir, f"{safe_name}.bak")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        shutil.copy2(full_path, checkpoint_path)
    except Exception:
        # Best-effort — don't fail the operation if checkpointing fails
        pass


def set_session_id(session_id):
    """Set a session identifier for checkpoint scoping."""
    global _checkpoint_session
    try:
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        session_dir = os.path.join(CHECKPOINT_DIR, f"session_{session_id}")
        os.makedirs(session_dir, exist_ok=True)
        _checkpoint_session = session_dir
    except OSError:
        _checkpoint_session = None
