"""Tests for the rollup script that produces pktpy-compatible output."""

import ast
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_SCRIPTS = _PROJECT / "scripts"
_SOURCE_DIR = _PROJECT / "duncan" / "duncan"


class TestRollup:
    def _run_merge(self, output_path, **kwargs):
        sys.path.insert(0, str(_SCRIPTS))
        try:
            from rollup import merge_scripts

            merge_scripts(_SOURCE_DIR, output_path, namespace_prefixes=["TN.", "TA.", "TS."], **kwargs)
        finally:
            sys.path.pop(0)

    def test_output_is_valid_python(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            output_path = Path(f.name)
        try:
            self._run_merge(output_path)
            code = output_path.read_text()
            tree = ast.parse(code)
            assert isinstance(tree, ast.Module)
        finally:
            output_path.unlink()

    def test_no_hashlib_import(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            output_path = Path(f.name)
        try:
            self._run_merge(output_path)
            code = output_path.read_text()
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                    if "hashlib" in names:
                        raise AssertionError("hashlib import found in output")
        finally:
            output_path.unlink()

    def test_no_argparse_import(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            output_path = Path(f.name)
        try:
            self._run_merge(output_path)
            code = output_path.read_text()
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                    if "argparse" in names:
                        raise AssertionError("argparse import found in output")
        finally:
            output_path.unlink()

    def test_no_relative_imports(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            output_path = Path(f.name)
        try:
            self._run_merge(output_path)
            code = output_path.read_text()
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level and node.level > 0:
                    raise AssertionError(f"relative import found at line {node.lineno}")
        finally:
            output_path.unlink()

    def test_no_lib_imports(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            output_path = Path(f.name)
        try:
            self._run_merge(output_path)
            code = output_path.read_text()
            assert "from lib" not in code
        finally:
            output_path.unlink()

    def test_no_generator_expressions(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            output_path = Path(f.name)
        try:
            self._run_merge(output_path)
            code = output_path.read_text()
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.GeneratorExp):
                    raise AssertionError(f"generator expression found at line {node.lineno}")
        finally:
            output_path.unlink()

    def test_contains_hash_str(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            output_path = Path(f.name)
        try:
            self._run_merge(output_path)
            code = output_path.read_text()
            assert "def _hash_str" in code
        finally:
            output_path.unlink()

    def test_contains_oracles_dict(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            output_path = Path(f.name)
        try:
            self._run_merge(output_path)
            code = output_path.read_text()
            assert "ORACLES" in code
        finally:
            output_path.unlink()

    def test_contains_dice_class(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            output_path = Path(f.name)
        try:
            self._run_merge(output_path)
            code = output_path.read_text()
            assert "class Dice:" in code
        finally:
            output_path.unlink()

    def test_no_namespace_prefixes(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            output_path = Path(f.name)
        try:
            self._run_merge(output_path)
            code = output_path.read_text()
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id in ("TN", "TA", "TS"):
                        raise AssertionError(f"namespace prefix {node.value.id}. found at line {node.lineno}")
        finally:
            output_path.unlink()

    def test_no_hashlib_sha256_call(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            output_path = Path(f.name)
        try:
            self._run_merge(output_path)
            code = output_path.read_text()
            assert "hashlib.sha256" not in code
        finally:
            output_path.unlink()

    def test_uses_print_instead_of_stdout_write(self):
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            output_path = Path(f.name)
        try:
            self._run_merge(output_path)
            code = output_path.read_text()
            assert "sys.stdout.write" not in code
        finally:
            output_path.unlink()
