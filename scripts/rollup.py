"""
Combines Python source modules into a single self-contained script
compatible with uv run pktpy.

Usage:
  python scripts/rollup.py --source-dir duncan/duncan --output out.py --strip-prefix TN. --strip-prefix TA.
  python scripts/rollup.py --source-dir cody/cody
"""

import argparse
import ast
import subprocess
import sys
from pathlib import Path

_INCOMPATIBLE_MODULES = frozenset(
    {
        "re",
        "pathlib",
        "itertools",
        "argparse",
        "hashlib",
        "subprocess",
        "threading",
        "glob",
        "shutil",
        "copy",
        "types",
        "struct",
        "ctypes",
        "urllib",
        "socket",
        "sqlite3",
        "importlib",
        "contextlib",
        "abc",
        "warnings",
        "weakref",
        "signal",
        "mmap",
        "configparser",
        "inspect",
        "csv",
        "fractions",
        "decimal",
        "statistics",
        "tempfile",
    }
)


def _compat_header() -> str:
    return """\
def _hash_str(s: str) -> int:
    h = 0
    for c in s:
        h = (h * 31 + ord(c)) & 0xFFFFFFFF
    return h
"""


class _GenExprToComp(ast.NodeTransformer):
    def visit_GeneratorExp(self, node):
        return ast.ListComp(
            elt=self.visit(node.elt),
            generators=[self.visit(g) for g in node.generators],
        )


def _is_name_eq_main(node: ast.If) -> bool:
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def transform_ast(code: str) -> tuple[list[str], str]:
    tree = ast.parse(code)
    imports = []
    new_body = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
            if any(name in _INCOMPATIBLE_MODULES for name in names):
                continue
            imports.append(ast.unparse(node))
            continue
        if isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            parts = (node.module or "").split(".")
            if parts[0] in _INCOMPATIBLE_MODULES or parts[0] == "lib":
                continue
            imports.append(ast.unparse(node))
            continue
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            continue
        if isinstance(node, ast.If) and _is_name_eq_main(node):
            continue
        new_body.append(node)
    tree.body = new_body
    _GenExprToComp().visit(tree)
    ast.fix_missing_locations(tree)
    return imports, ast.unparse(tree)


def transform_text(code: str, namespace_prefixes: list[str] | None = None) -> tuple[str, bool]:
    had_hash = "hashlib.sha256(seed.encode()).hexdigest()" in code
    code = code.replace(
        "hashlib.sha256(seed.encode()).hexdigest()",
        "str(_hash_str(seed))",
    )
    code = code.replace("sys.stdout.write(", "print(")
    code = code.replace("sys.stderr.isatty()", "False")
    if namespace_prefixes:
        for prefix in namespace_prefixes:
            code = code.replace(prefix, "")
    return code, had_hash


def merge_scripts(
    source_dir: Path,
    output_path: Path,
    namespace_prefixes: list[str] | None = None,
    exclude: list[str] | None = None,
) -> None:
    import fnmatch

    py_files = sorted(p for p in source_dir.iterdir() if p.suffix == ".py" and p.name != "__init__.py")
    if exclude:
        py_files = [p for p in py_files if not any(fnmatch.fnmatch(p.name, pat) for pat in exclude)]
    main_file = next((p for p in py_files if p.name == "__main__.py"), None)
    if main_file:
        py_files = [p for p in py_files if p.name != "__main__.py"] + [main_file]
    all_imports = []
    parts = []
    needs_hash = False
    for filepath in py_files:
        code = filepath.read_text()
        imports, body = transform_ast(code)
        body, had_hash = transform_text(body, namespace_prefixes)
        if had_hash:
            needs_hash = True
        all_imports.extend(imports)
        parts.append(body)
    sections = []
    if all_imports:
        sections.append("\n".join(all_imports))
    if needs_hash:
        sections.append(_compat_header().rstrip("\n"))
    sections.append("\n\n".join(parts))
    result = "\n\n".join(sections)
    output_path.write_text(result)


def main():
    parser = argparse.ArgumentParser(
        description="Roll Python source modules into a single pktpy-compatible script",
    )
    parser.add_argument("--source-dir", required=True, help="Source directory with .py files")
    parser.add_argument("--output", "-o", default="out.py", help="Output file path")
    parser.add_argument(
        "--strip-prefix",
        action="append",
        default=None,
        help="Namespace prefix to strip from attribute access (repeatable, e.g. --strip-prefix TN.)",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=None,
        help="Glob pattern for files to exclude (repeatable, e.g. --exclude 'test_*')",
    )
    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        print(f"Source directory not found: {source_dir}", file=sys.stderr)
        sys.exit(1)
    merge_scripts(
        source_dir,
        Path(args.output),
        namespace_prefixes=args.strip_prefix,
        exclude=args.exclude,
    )
    print(f"Wrote {args.output}", file=sys.stderr)
    subprocess.run(["ruff", "format", args.output], capture_output=True)
    result = subprocess.run(["ruff", "check", args.output, "--fix"], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: ruff check found issues in {args.output}.", file=sys.stderr)
        names = set()
        for line in result.stdout.splitlines():
            if "F821" in line and "Undefined name" in line:
                name = line.split("Undefined name")[-1].strip().strip("`'\"")
                if name:
                    names.add(name)
        if names:
            prefixes = " ".join((f"--strip-prefix {n}." for n in sorted(names) if n.upper() == n))
            if prefixes:
                print(f"  Consider: {prefixes}", file=sys.stderr)

        print(f"\n===ruff check output, first 100l===\n{'\n'.join(result.stdout.splitlines()[:100])}")


if __name__ == "__main__":
    main()
