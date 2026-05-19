"""Unit tests for handler functions in cody.handlers."""


class TestHandleRead:
    def test_read_existing_file(self, tmp_path):
        from packages.cody.handlers import _handle_read as h

        target = tmp_path / "test.txt"
        target.write_text("hello\nworld")
        
        result = h({"path": str(target)})
        assert "(1-2/2 lines)" in result


class TestHandleWrite:
    def test_write_new_file(self, tmp_path):
        from packages.cody.handlers import _handle_write as h

        target = tmp_path / "new.txt"
        
