"""Test that example.py runs without error."""

import os
import tempfile


def test_example_runs():
    import importlib.util

    example_path = os.path.join(os.path.dirname(__file__), "..", "example.py")
    spec = importlib.util.spec_from_file_location("example", example_path)
    mod = importlib.util.module_from_spec(spec)

    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            spec.loader.exec_module(mod)
            mod.main()
        finally:
            os.chdir(old_cwd)
