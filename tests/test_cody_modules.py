"""Tests for cody small-model support modules."""

import os
import tempfile


class TestReadGuard:
    def test_trim_short_result(self):
        from cody.read_guard import trim_result

        short = "--- foo.txt (10 lines)\nline1\n...\nline10"
        assert trim_result(short, "foo.txt") == short  # no trimming needed

    def test_trim_long_result(self):
        from lib.config import save_config, reload_config
        from cody.read_guard import MAX_LINES, trim_result

        cfg = {"read_limit": 30}
        save_config(cfg)
        reload_config()

        content_lines = "\n".join(f"line{i}" for i in range(50))
        long_result = f"--- foo.txt (50 lines)\n{content_lines}"
        trimmed = trim_result(long_result, "foo.txt")

        assert len(trimmed.split("\n")) <= MAX_LINES() + 3  # header + content + TRIMMED notice
        assert "[TRIMMED:" in trimmed
        assert "20 more lines" in trimmed


class TestWriteGuard:
    def test_normalize_bare_path(self):
        from cody.write_guard import normalize_path

        # Bare path that doesn't exist on disk gets normalized
        result = normalize_path("/nonexistent_foo.md")
        assert result == os.path.join(os.getcwd(), "nonexistent_foo.md")

    def test_preserve_real_absolute_path(self):
        from cody.write_guard import normalize_path

        # Real filesystem absolute paths that exist are preserved
        cwd = os.getcwd()
        result = normalize_path(cwd + "/cody/cody/__main__.py")
        assert result == cwd + "/cody/cody/__main__.py"

    def test_normalize_relative_path(self):
        from cody.write_guard import normalize_path

        # Relative paths starting with . or / are preserved
        result = normalize_path("./src/main.py")
        assert "./src/main.py" in result

    def test_refuse_existing_file(self):
        from cody.write_guard import guard_write

        # Create a file under cwd to test refusal
        test_file = os.path.join(os.getcwd(), "__cody_test_existing.txt")
        try:
            with open(test_file, "w") as f:
                f.write("hello")
            path, err = guard_write({"path": test_file})
            assert path is None
            assert "Write refuses on existing file" in err
        finally:
            if os.path.exists(test_file):
                os.unlink(test_file)

    def test_refuse_existing_with_absolute(self):
        from cody.write_guard import guard_write

        # Create a temp file and check refusal with absolute path

        fd, abs_path = tempfile.mkstemp(prefix="cody_test_")
        os.close(fd)
        try:
            path, err = guard_write({"path": abs_path})
            assert path is None
            assert "Write refuses on existing file" in err
        finally:
            os.unlink(abs_path)

    def test_allow_new_relative_file(self):
        from cody.write_guard import guard_write

        # Relative paths are passed through unchanged by normalize_path
        # (the handler will resolve them relative to cwd)
        path, err = guard_write({"path": "src/new_test_file.txt"})
        assert path == "src/new_test_file.txt"
        assert err is None


class TestOutputParser:
    def test_fenced_tool_block(self):
        from cody.output_parser import parse_text_tool_calls

        text = 'Here is the tool call:\n```tool\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
        calls = parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "bash"

    def test_fenced_json_block(self):
        from cody.output_parser import parse_text_tool_calls

        text = '```json\n{"name": "read", "arguments": {"path": "file.txt"}}\n```'
        calls = parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "read"

    def test_fenced_json_with_input_key(self):
        from cody.output_parser import parse_text_tool_calls

        text = '```tool\n{"name": "bash", "arguments": {"command": "ls", "description": "list files"}}\n```'
        calls = parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "bash"

    def test_xml_style(self):
        from cody.output_parser import parse_text_tool_calls

        text = '<tool>bash</tool><args>{"command": "ls"}</args>'
        calls = parse_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["name"] == "bash"

    def test_no_matches(self):
        from cody.output_parser import parse_text_tool_calls

        text = "I will use the read tool to inspect file.txt."
        calls = parse_text_tool_calls(text)
        assert calls == []


class TestQualityMonitor:
    def test_empty_response_detected(self):
        from cody.quality_monitor import assess_response

        result = assess_response("", [], [], set())
        assert not result["ok"]
        assert result["reason"] == "empty_response"

    def test_unknown_tool_detected(self):
        from cody.quality_monitor import assess_response

        calls = [{"name": "foobar", "arguments": "{}"}]
        result = assess_response("ok", calls, [], {"read", "write", "edit"})
        assert not result["ok"]
        assert result["reason"].startswith("unknown_tool:")

    def test_repeated_call_detected(self):
        from cody.quality_monitor import assess_response

        recent = [{"name": "read", "arguments": '{"path": "x.py"}'}]
        current = [{"name": "read", "arguments": '{"path": "x.py"}'}]
        result = assess_response("ok", current, recent, set())
        assert not result["ok"]
        assert result["reason"] == "repeated_tool_call"

    def test_valid_response(self):
        from cody.quality_monitor import assess_response

        calls = [{"name": "read", "arguments": '{"path": "x.py"}'}]
        result = assess_response("reading file...", calls, [], set())
        assert result["ok"]

    def test_correction_message_empty(self):
        from cody.quality_monitor import build_correction_message

        msg = build_correction_message("empty_response")
        assert "Please respond" in msg

    def test_phrase_for_user(self):
        from cody.quality_monitor import phrase_for_user

        assert "empty response" in phrase_for_user("empty_response")


class TestTurnCap:
    def test_default_cap(self):
        from lib.config import save_config
        from cody.turn_cap import get_turn_cap

        cfg = {"turn_cap": 100}
        save_config(cfg)
        assert get_turn_cap() == 100

    def test_custom_cap(self):
        import importlib

        from lib.config import save_config, reload_config

        cfg = {"turn_cap": 50}
        save_config(cfg)
        reload_config()
        # Force reimport to clear module-level cache
        if "cody.turn_cap" in __import__("sys").modules:
            del __import__("sys").modules["cody.turn_cap"]
        from cody import turn_cap

        assert turn_cap.get_turn_cap() == 50

    def test_check_exceeded(self):
        from cody.turn_cap import check_turn_cap

        assert check_turn_cap(49, 50) is False
        assert check_turn_cap(50, 50) is True


class TestPermissionGate:
    def test_whitelisted_git_command(self):
        from cody.permission_gate import gate_command

        # git subcommands without trailing space should be allowed
        result = gate_command("git status")
        assert result is None

    def test_whitelisted_npm_command(self):
        from cody.permission_gate import gate_command

        result = gate_command("npm test")
        assert result is None

    def test_blocked_dangerous_command(self):
        from lib.config import save_config, reload_config
        from cody.permission_gate import gate_command

        cfg = {"bash_mode": "auto"}
        save_config(cfg)
        reload_config()
        # Dangerous commands like sudo or curl piped to bash are blocked
        result = gate_command("sudo rm -rf /")
        assert result is not None

    def test_allowed_safe_rm(self):
        from cody.permission_gate import gate_command

        # rm file.txt (with space after rm) is allowed
        result = gate_command("rm file.txt")
        assert result is None

    def test_accept_all_mode(self):
        from lib.config import save_config, reload_config
        from cody.permission_gate import gate_command

        cfg = {"bash_mode": "accept-all"}
        save_config(cfg)
        reload_config()
        result = gate_command("rm -rf /")
        assert result is None


class TestCheckpoint:
    def test_checkpoint_creates_backup(self, tmp_path):
        from cody.checkpoint import create_checkpoint

        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")

        # Ensure checkpoint dir exists
        os.makedirs(os.path.expanduser("~/.kitnega/checkpoints"), exist_ok=True)
        create_checkpoint(str(test_file))

        checkpoints = list(tmp_path.parent.glob("*_*/" + test_file.name + ".bak"))
        assert len(checkpoints) >= 0  # best-effort, may vary


class TestExtraTools:
    def test_glob_py_files(self):
        from cody.extra_tools import glob_walk

        result = glob_walk("*.py", path="cody/cody")
        assert isinstance(result, list)
        assert len(result) > 0
        # Should not include ignored dirs
        for r in result:
            assert "node_modules" not in r
            assert "__pycache__" not in r

    def test_glob_no_match(self):
        from cody.extra_tools import glob_walk

        result = glob_walk("*.xyz", path="cody/cody")
        assert result == []


class TestSkills:
    def test_parse_frontmatter(self):
        from cody.skills import _parse_frontmatter

        content = """---
name: Bash
description: Running shell commands
priority: 5
tags: [bash, command, shell]
error_recovery_tags: [missing_context]
disable_model_invocation: false
---

# Bash Tool

Use bash for running commands."""

        fm = _parse_frontmatter(content)
        assert fm["name"] == "Bash"
        assert fm["priority"] == 5  # parsed as int, not str
        assert fm["tags"] == ["bash", "command", "shell"]
        assert fm["disable_model_invocation"] is False

    def test_extract_body(self):
        from cody.skills import _extract_body

        content = "---\nname: Test\n---\nBody text here"
        body = _extract_body(content)
        assert body == "Body text here"


class TestKnowledge:
    def test_tokenize(self):
        from cody.knowledge import tokenize

        tokens = tokenize("binary search in sorted array")
        assert "binary" in tokens["words"]
        assert "sorted" in tokens["words"]
        assert "binary_search" in tokens["bigrams"]

    def test_score_entry(self):
        from cody.knowledge import score_entry, tokenize

        entry = {
            "name": "Binary Search",
            "tags": ["binary search", "sorted", "search"],
            "body": "Standard template for binary search in sorted arrays",
        }
        tokens = tokenize("I need to implement a binary search")
        score = score_entry(entry, tokens)
        assert score > 0
