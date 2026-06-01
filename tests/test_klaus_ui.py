"""Playwright-based browser tests for the klaus chat server."""

import socket
import threading
import time
from uuid import uuid4

import pytest
from playwright.sync_api import expect


def _unique(prefix="u"):
    return f"{prefix}{uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def db_path(tmp_path_factory):
    return str(tmp_path_factory.mktemp("klaus_data") / "klaus.db")


@pytest.fixture(scope="session")
def server_url(db_path):
    from lib.bottle import run

    from klaus import app as chat_app

    chat_app.configure("test-secret-key", db_path)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    def serve():
        run(chat_app.app, host="127.0.0.1", port=port, quiet=True)

    t = threading.Thread(target=serve, daemon=True)
    t.start()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=1)
            s.close()
            break
        except OSError:
            time.sleep(0.1)
    else:
        raise RuntimeError("Server did not start in time")

    return f"http://127.0.0.1:{port}"


def _register(page, server_url, username=None, display_name=None, password="pass"):
    username = username or _unique()
    display_name = display_name or username
    page.goto(f"{server_url}/register")
    page.fill("#username", username)
    page.fill("#display_name", display_name)
    page.fill("#password", password)
    page.locator("button[type='submit']").click()
    page.wait_for_url(f"{server_url}/")
    return username


class TestAuth:
    def test_redirects_unauthenticated_to_login(self, page, server_url):
        page.goto(server_url)
        assert page.url.rstrip("/") == f"{server_url}/login"
        expect(page.locator("h1")).to_have_text("Sign in to klaus")

    def test_register_and_see_sidebar(self, page, server_url):
        uname = _register(page, server_url)
        expect(page.locator(".sidebar-footer")).to_contain_text(uname)

    def test_bad_login_shows_error(self, page, server_url):
        page.goto(f"{server_url}/login")
        page.fill("#username", "nonexistent")
        page.fill("#password", "wrong")
        page.locator("button[type='submit']").click()
        expect(page.locator(".error")).to_have_text("Invalid username or password")

    def test_logout_redirects_to_login(self, page, server_url):
        _register(page, server_url, _unique("lo"))
        page.locator("button:has-text('Sign out')").click()
        page.wait_for_url(f"{server_url}/login")
        expect(page.locator("h1")).to_have_text("Sign in to klaus")

    def test_login_after_logout(self, page, server_url):
        uname = _unique("ll")
        pwd = "secret"
        _register(page, server_url, uname, password=pwd)
        page.locator("button:has-text('Sign out')").click()
        page.wait_for_url(f"{server_url}/login")
        page.fill("#username", uname)
        page.fill("#password", pwd)
        page.locator("button[type='submit']").click()
        page.wait_for_url(f"{server_url}/")
        expect(page.locator(".sidebar-footer")).to_contain_text(uname)


class TestMessages:
    def test_send_message_in_general(self, page, server_url):
        _register(page, server_url, _unique("msgg"))
        page.locator("#msg-input").fill("hello from playwright")
        page.locator("#msg-form button[type='submit']").click()
        expect(page.locator(".message-content").last).to_have_text("hello from playwright")

    def test_multiple_messages_appear(self, page, server_url):
        _register(page, server_url, _unique("msgm"))
        page.locator("#msg-input").fill("first")
        page.locator("#msg-form button[type='submit']").click()
        expect(page.locator(".message-content").last).to_have_text("first")
        page.locator("#msg-input").fill("second")
        page.locator("#msg-form button[type='submit']").click()
        expect(page.locator(".message-content").last).to_have_text("second")


class TestRooms:
    def test_create_room_and_shows_in_header(self, page, server_url):
        _register(page, server_url, _unique("roomc"))
        page.locator("a:has-text('+ New')").click()
        page.wait_for_selector("#room-modal")
        expect(page.locator("#room-modal")).to_be_visible()
        page.fill("input[name='name']", "my-room")
        page.fill("input[name='topic']", "testing rooms")
        page.locator("button:has-text('Create')").click()
        page.wait_for_url(f"{server_url}/?room=*")
        expect(page.locator("h2")).to_contain_text("my-room")

    def test_created_room_appears_in_sidebar(self, page, server_url):
        uname = _unique("rooms")
        _register(page, server_url, uname)
        page.locator("a:has-text('+ New')").click()
        page.wait_for_selector("#room-modal")
        page.fill("input[name='name']", "side-room")
        page.locator("button:has-text('Create')").click()
        page.wait_for_url(f"{server_url}/?room=*")
        expect(page.get_by_role("link", name="# side-room")).to_be_visible()
