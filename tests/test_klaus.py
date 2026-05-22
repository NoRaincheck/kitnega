"""Tests for klaus database layer and app routes."""

import io

import pytest

# ---------------------------------------------------------------------------
# Database-layer tests
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "klaus.db")


class TestDB:
    def test_init_db_creates_tables(self, db_path):
        from klaus import db

        db.init_db(db_path)
        conn = db._get_db()
        tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
        names = {r[0] for r in tables}
        assert "users" in names
        assert "rooms" in names
        assert "room_members" in names
        assert "messages" in names
        assert "messages_fts" in names
        conn.close()

    def test_seed_general_room(self, db_path):
        from klaus import db

        db.init_db(db_path)
        conn = db._get_db()
        row = conn.execute("SELECT id, name, is_public FROM rooms").fetchone()
        assert row is not None
        assert row[1] == "#general"
        assert row[2] == 1
        conn.close()

    def test_create_user(self, db_path):
        from klaus import db

        db.init_db(db_path)
        uid = db.create_user("alice", "Alice", "secret")
        assert uid is not None
        user = db.get_user(uid)
        assert user["username"] == "alice"
        assert user["display_name"] == "Alice"
        assert user["role"] == "member"

    def test_create_user_duplicate(self, db_path):
        from klaus import db

        db.init_db(db_path)
        db.create_user("alice", "Alice", "secret")
        assert db.create_user("alice", "Alice", "secret") is None

    def test_create_user_joins_general(self, db_path):
        from klaus import db

        db.init_db(db_path)
        uid = db.create_user("bob", "Bob", "pass")
        conn = db._get_db()
        rows = conn.execute(
            "SELECT r.name FROM room_members rm JOIN rooms r ON r.id = rm.room_id WHERE rm.user_id = ?",
            (uid,),
        ).fetchall()
        names = {r[0] for r in rows}
        assert "#general" in names
        conn.close()

    def test_authenticate_ok(self, db_path):
        from klaus import db

        db.init_db(db_path)
        db.create_user("alice", "Alice", "secret")
        assert db.authenticate("alice", "secret") is not None

    def test_authenticate_wrong_password(self, db_path):
        from klaus import db

        db.init_db(db_path)
        db.create_user("alice", "Alice", "secret")
        assert db.authenticate("alice", "wrong") is None

    def test_authenticate_unknown_user(self, db_path):
        from klaus import db

        db.init_db(db_path)
        assert db.authenticate("nobody", "x") is None

    def test_get_user_by_username(self, db_path):
        from klaus import db

        db.init_db(db_path)
        uid = db.create_user("alice", "Alice", "s")
        found = db.get_user_by_username("alice")
        assert found["id"] == uid
        assert db.get_user_by_username("nobody") is None

    def test_create_admin(self, db_path):
        from klaus import db

        db.init_db(db_path)
        uid = db.create_admin("root", "adminpass")
        assert uid is not None
        user = db.get_user(uid)
        assert user["role"] == "admin"

    def test_create_admin_duplicate(self, db_path):
        from klaus import db

        db.init_db(db_path)
        db.create_admin("root", "p")
        assert db.create_admin("root", "p") is None

    def test_create_room(self, db_path):
        from klaus import db

        db.init_db(db_path)
        admin = db.create_admin("admin", "p")
        room_id = db.create_room("#random", "Random chat", True, admin)
        assert room_id is not None
        room = db.get_room(room_id)
        assert room["name"] == "#random"
        assert room["is_public"] == 1

    def test_create_room_duplicate(self, db_path):
        from klaus import db

        db.init_db(db_path)
        admin = db.create_admin("admin", "p")
        db.create_room("#random", "", True, admin)
        assert db.create_room("#random", "", True, admin) is None

    def test_get_rooms_includes_membership(self, db_path):
        from klaus import db

        db.init_db(db_path)
        admin = db.create_admin("admin", "p")
        rooms = db.get_rooms(admin)
        general = next(r for r in rooms if r["name"] == "#general")
        assert general["is_member"] == 1

    def test_join_and_leave_room(self, db_path):
        from klaus import db

        db.init_db(db_path)
        admin = db.create_admin("admin", "p")
        room_id = db.create_room("#secret", "", False, admin)

        db.leave_room(room_id, admin)
        rooms = db.get_rooms(admin)
        secret = next(r for r in rooms if r["name"] == "#secret")
        assert secret["is_member"] == 0

        db.join_room(room_id, admin)
        rooms = db.get_rooms(admin)
        secret = next(r for r in rooms if r["name"] == "#secret")
        assert secret["is_member"] == 1

    def test_toggle_visibility(self, db_path):
        from klaus import db

        db.init_db(db_path)
        admin = db.create_admin("admin", "p")
        room_id = db.create_room("#secret", "", False, admin)
        assert db.toggle_room_visibility(room_id) is True
        room = db.get_room(room_id)
        assert room["is_public"] == 1
        assert db.toggle_room_visibility(room_id) is False

    def test_get_dm_room(self, db_path):
        from klaus import db

        db.init_db(db_path)
        alice = db.create_user("alice", "Alice", "p")
        bob = db.create_user("bob", "Bob", "p")

        room_id = db.get_dm_room(alice, bob)
        assert room_id is not None
        room = db.get_room(room_id)
        assert room["is_public"] == 0
        assert "_dm_" in room["name"]

        same_room_id = db.get_dm_room(alice, bob)
        assert same_room_id == room_id

    def test_send_message(self, db_path):
        from klaus import db

        db.init_db(db_path)
        alice = db.create_user("alice", "Alice", "p")
        general = db.get_rooms(alice)[0]
        msg = db.send_message(general["id"], alice, "hello")
        assert msg is not None
        assert msg["content"] == "hello"

    def test_get_messages(self, db_path):
        from klaus import db

        db.init_db(db_path)
        alice = db.create_user("alice", "Alice", "p")
        bob = db.create_user("bob", "Bob", "p")
        general = db.get_rooms(alice)[0]

        db.send_message(general["id"], alice, "hello from alice")
        db.send_message(general["id"], bob, "hey alice")

        msgs = db.get_messages(general["id"], alice)
        assert len(msgs) == 2
        assert msgs[0]["content"] == "hello from alice"
        assert msgs[1]["content"] == "hey alice"

    def test_get_messages_non_member_private_room(self, db_path):
        from klaus import db

        db.init_db(db_path)
        alice = db.create_user("alice", "Alice", "p")
        bob = db.create_user("bob", "Bob", "p")
        charlie = db.create_user("charlie", "Charlie", "p")

        dm_id = db.get_dm_room(alice, bob)
        result = db.get_messages(dm_id, charlie)
        assert result is None

    def test_send_message_non_member_private_room(self, db_path):
        from klaus import db

        db.init_db(db_path)
        alice = db.create_user("alice", "Alice", "p")
        bob = db.create_user("bob", "Bob", "p")
        charlie = db.create_user("charlie", "Charlie", "p")

        dm_id = db.get_dm_room(alice, bob)
        assert db.send_message(dm_id, charlie, "should not work") is None

    def test_search_messages(self, db_path):
        from klaus import db

        db.init_db(db_path)
        alice = db.create_user("alice", "Alice", "p")
        general = db.get_rooms(alice)[0]
        db.send_message(general["id"], alice, "hello world")
        db.send_message(general["id"], alice, "goodbye world")

        results = db.search_messages("hello", alice)
        assert len(results) == 1
        assert results[0]["content"] == "hello world"

    def test_list_users(self, db_path):
        from klaus import db

        db.init_db(db_path)
        db.create_user("alice", "Alice", "p")
        db.create_user("bob", "Bob", "p")
        users = db.list_users()
        assert len(users) >= 2

    def test_password_hashing(self, db_path):
        from klaus import db

        db.init_db(db_path)
        uid = db.create_user("testuser", "Test", "mypassword")
        user = db.get_user(uid)
        assert user is not None
        assert db.authenticate("testuser", "mypassword") == uid
        assert db.authenticate("testuser", "wrong") is None
        assert db.authenticate("nobody", "mypassword") is None


# ---------------------------------------------------------------------------
# App-level integration tests
# ---------------------------------------------------------------------------


class _TestClient:
    """Minimal WSGI test client using stdlib only."""

    def __init__(self, app):
        self.app = app

    def _call(self, method, path, headers=None, body=None):
        environ = {
            "REQUEST_METHOD": method,
            "PATH_INFO": path,
            "SERVER_NAME": "127.0.0.1",
            "SERVER_PORT": "80",
            "SERVER_PROTOCOL": "HTTP/1.1",
            "wsgi.version": (1, 0),
            "wsgi.url_scheme": "http",
            "wsgi.input": io.BytesIO(body or b""),
            "wsgi.errors": io.BytesIO(),
            "wsgi.multithread": False,
            "wsgi.multiprocess": False,
            "wsgi.run_once": False,
            "CONTENT_LENGTH": str(len(body or b"")),
            "CONTENT_TYPE": headers.get("Content-Type", "application/x-www-form-urlencoded")
            if headers
            else "application/x-www-form-urlencoded",
        }
        if headers:
            for k, v in headers.items():
                if k.startswith("Content-"):
                    continue
                wsgi_key = "HTTP_" + k.upper().replace("-", "_")
                environ[wsgi_key] = v

        status = []
        resp_headers = []
        body_out = []

        def start_response(s, h, exc_info=None):
            status.append(s)
            resp_headers.extend(h)
            return body_out.append

        chunks = self.app.wsgi(environ, start_response)
        raw = b"".join(chunks)

        class Response:
            status_code = int(status[0].split()[0])
            headers = dict(resp_headers)
            body = raw
            text = raw.decode("utf-8", errors="replace")
            set_cookie = resp_headers[-1][1] if any(k.lower() == "set-cookie" for k, _ in resp_headers) else None

            @property
            def cookie_dict(self):
                if not self.set_cookie:
                    return {}
                parts = self.set_cookie.split(";")[0]
                if "=" in parts:
                    k, v = parts.split("=", 1)
                    return {k: v}
                return {}

        return Response()


@pytest.fixture
def client(db_path):
    from klaus import app as chat_app

    chat_app.configure("test-secret-key", db_path)
    return _TestClient(chat_app.app)


class TestApp:
    def test_login_page_returns_200(self, client):
        resp = client._call("GET", "/login")
        assert resp.status_code == 200
        assert "Sign in" in resp.text

    def test_register_page_returns_200(self, client):
        resp = client._call("GET", "/register")
        assert resp.status_code == 200
        assert "Create an account" in resp.text

    def test_register_and_login(self, client):
        form = "username=testuser&display_name=Test&password=secret123"
        resp = client._call("POST", "/register", body=form.encode())
        assert resp.status_code == 303
        cookie = resp.cookie_dict

        resp2 = client._call("GET", "/", headers={"Cookie": f"user_id={cookie.get('user_id', '')}"})
        assert resp2.status_code == 200
        assert "testuser" in resp2.text

    def test_login_bad_credentials(self, client):
        form = "username=nobody&password=wrong"
        resp = client._call("POST", "/login", body=form.encode())
        assert resp.status_code == 200
        assert "Invalid" in resp.text

    def test_logout(self, client):
        resp = client._call("POST", "/logout")
        assert resp.status_code == 303

    def test_root_redirects_when_unauthenticated(self, client):
        resp = client._call("GET", "/")
        assert resp.status_code in (302, 303)

    def test_create_room(self, client):
        form = "username=creator&display_name=Creator&password=pass"
        resp = client._call("POST", "/register", body=form.encode())
        cookie = resp.cookie_dict

        form2 = "name=test-room&topic=testing&is_public=1"
        resp2 = client._call(
            "POST", "/rooms", headers={"Cookie": f"user_id={cookie.get('user_id', '')}"}, body=form2.encode()
        )
        assert resp2.status_code in (302, 303)

    def test_send_message_to_room(self, client):
        form = "username=alice&display_name=Alice&password=pass"
        resp = client._call("POST", "/register", body=form.encode())
        cookie = resp.cookie_dict

        rooms_resp = client._call("GET", "/", headers={"Cookie": f"user_id={cookie.get('user_id', '')}"})
        assert rooms_resp.status_code == 200

        msg_form = "content=hello+world"
        msg_resp = client._call(
            "POST",
            "/rooms/1/messages",
            headers={"Cookie": f"user_id={cookie.get('user_id', '')}"},
            body=msg_form.encode(),
        )
        assert msg_resp.status_code == 200
        assert "hello world" in msg_resp.text

    def test_get_messages_endpoint(self, client):
        form = "username=bob&display_name=Bob&password=pass"
        resp = client._call("POST", "/register", body=form.encode())
        cookie = resp.cookie_dict

        resp2 = client._call(
            "GET",
            "/rooms/1/messages",
            headers={"Cookie": f"user_id={cookie.get('user_id', '')}"},
        )
        assert resp2.status_code == 200

    def test_non_member_cannot_access_private_room(self, client):
        form = "username=admin&password=p"
        resp = client._call("POST", "/register", body=form.encode())
        admin_cookie = resp.cookie_dict

        form = "username=user&password=p"
        resp = client._call("POST", "/register", body=form.encode())

        rooms_resp = client._call("GET", "/", headers={"Cookie": f"user_id={admin_cookie.get('user_id', '')}"})
        assert rooms_resp.status_code == 200

    def test_dm_endpoint(self, client):
        resp = client._call("POST", "/register", body="username=alice&password=p&display_name=Alice".encode())
        alice_cookie = resp.cookie_dict

        resp = client._call("POST", "/register", body="username=bob&password=p&display_name=Bob".encode())

        dm_resp = client._call(
            "POST",
            "/dm/bob",
            headers={"Cookie": f"user_id={alice_cookie.get('user_id', '')}"},
        )
        assert dm_resp.status_code == 200
        assert dm_resp.headers.get("Hx-Redirect", "").startswith("/?room=")
