import hashlib
import os
import sqlite3

DB_PATH = None


def _get_db():
    return sqlite3.connect(DB_PATH)


def init_db(path, force=False):
    global DB_PATH
    if os.path.exists(path):
        if not force:
            raise FileExistsError(f"Database already exists at {path}. Use --force to overwrite.")
        os.remove(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    DB_PATH = path
    conn = _get_db()
    conn.executescript(_SCHEMA)
    try:
        conn.execute("ALTER TABLE room_members ADD COLUMN last_read_at TEXT NOT NULL DEFAULT (datetime('now'))")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    _seed_general(conn)
    conn.commit()
    conn.close()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'member',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rooms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    topic TEXT NOT NULL DEFAULT '',
    is_public INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS room_members (
    room_id INTEGER NOT NULL REFERENCES rooms(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    last_read_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (room_id, user_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    room_id INTEGER NOT NULL REFERENCES rooms(id),
    user_id INTEGER NOT NULL REFERENCES users(id),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content, content=messages, content_rowid=id);
"""


def _seed_general(conn):
    existing = conn.execute("SELECT id FROM rooms WHERE name = ?", ("#general",)).fetchone()
    if existing:
        return
    conn.execute("INSERT INTO rooms (name, topic, is_public) VALUES (?, ?, 1)", ("#general", ""))


def hash_password(password):
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
    return salt.hex() + ":" + dk.hex()


def check_password(password, stored):
    salt_hex, dk_hex = stored.split(":", 1)
    salt = bytes.fromhex(salt_hex)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 100000)
    return dk.hex() == dk_hex


def create_admin(username, password):
    conn = _get_db()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        return None
    cur = conn.execute(
        "INSERT INTO users (username, display_name, password_hash, role) VALUES (?, ?, ?, 'admin')",
        (username, username, hash_password(password)),
    )
    conn.commit()
    uid = cur.lastrowid
    general = conn.execute("SELECT id FROM rooms WHERE name = '#general'").fetchone()
    conn.execute("INSERT OR IGNORE INTO room_members (room_id, user_id) VALUES (?, ?)", (general[0], uid))
    conn.commit()
    conn.close()
    return uid


def create_user(username, display_name, password):
    conn = _get_db()
    existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        conn.close()
        return None
    cur = conn.execute(
        "INSERT INTO users (username, display_name, password_hash) VALUES (?, ?, ?)",
        (username, display_name, hash_password(password)),
    )
    uid = cur.lastrowid
    general = conn.execute("SELECT id FROM rooms WHERE name = '#general'").fetchone()
    conn.execute("INSERT OR IGNORE INTO room_members (room_id, user_id) VALUES (?, ?)", (general[0], uid))
    conn.commit()
    conn.close()
    return uid


def authenticate(username, password):
    conn = _get_db()
    row = conn.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if row and check_password(password, row[1]):
        return row[0]
    return None


def get_user(user_id):
    conn = _get_db()
    row = conn.execute("SELECT id, username, display_name, role FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    if row:
        return {"id": row[0], "username": row[1], "display_name": row[2], "role": row[3]}
    return None


def get_user_by_username(username):
    conn = _get_db()
    row = conn.execute("SELECT id, username, display_name, role FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    if row:
        return {"id": row[0], "username": row[1], "display_name": row[2], "role": row[3]}
    return None


def list_users():
    conn = _get_db()
    rows = conn.execute("SELECT id, username, display_name FROM users ORDER BY username").fetchall()
    conn.close()
    return [{"id": r[0], "username": r[1], "display_name": r[2]} for r in rows]


def get_rooms(user_id):
    conn = _get_db()
    rows = conn.execute("""
        SELECT r.id, r.name, r.topic, r.is_public,
               (SELECT COUNT(*) FROM messages m WHERE m.room_id = r.id AND (rm.last_read_at IS NULL OR m.created_at > rm.last_read_at)) AS unread_count,
               (SELECT MAX(m.created_at) FROM messages m WHERE m.room_id = r.id) AS last_message_at,
               CASE WHEN rm.user_id IS NOT NULL THEN 1 ELSE 0 END AS is_member
        FROM rooms r
        LEFT JOIN room_members rm ON rm.room_id = r.id AND rm.user_id = ?
        ORDER BY r.is_public DESC, last_message_at DESC
    """, (user_id,)).fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "topic": r[2], "is_public": r[3],
             "unread_count": r[4], "last_message_at": r[5], "is_member": r[6]} for r in rows]


def get_room(room_id):
    conn = _get_db()
    row = conn.execute("SELECT id, name, topic, is_public FROM rooms WHERE id = ?", (room_id,)).fetchone()
    conn.close()
    if row:
        return {"id": row[0], "name": row[1], "topic": row[2], "is_public": row[3]}
    return None


def create_room(name, topic, is_public, creator_id):
    conn = _get_db()
    existing = conn.execute("SELECT id FROM rooms WHERE name = ?", (name,)).fetchone()
    if existing:
        conn.close()
        return None
    cur = conn.execute(
        "INSERT INTO rooms (name, topic, is_public) VALUES (?, ?, ?)",
        (name, topic, is_public),
    )
    room_id = cur.lastrowid
    conn.execute("INSERT OR IGNORE INTO room_members (room_id, user_id) VALUES (?, ?)", (room_id, creator_id))
    conn.commit()
    conn.close()
    return room_id


def get_dm_room(user_id, other_id):
    name = "_dm_{}_{}".format(*sorted([user_id, other_id]))
    conn = _get_db()
    row = conn.execute("SELECT id FROM rooms WHERE name = ?", (name,)).fetchone()
    if row:
        conn.close()
        return row[0]
    cur = conn.execute("INSERT INTO rooms (name, topic, is_public) VALUES (?, '', 0)", (name,))
    room_id = cur.lastrowid
    conn.execute("INSERT OR IGNORE INTO room_members (room_id, user_id) VALUES (?, ?), (?, ?)",
                 (room_id, user_id, room_id, other_id))
    conn.commit()
    conn.close()
    return room_id


def join_room(room_id, user_id):
    conn = _get_db()
    conn.execute("INSERT OR IGNORE INTO room_members (room_id, user_id) VALUES (?, ?)", (room_id, user_id))
    conn.commit()
    conn.close()


def leave_room(room_id, user_id):
    conn = _get_db()
    conn.execute("DELETE FROM room_members WHERE room_id = ? AND user_id = ?", (room_id, user_id))
    conn.commit()
    conn.close()


def mark_read(room_id, user_id):
    conn = _get_db()
    conn.execute(
        "UPDATE room_members SET last_read_at = datetime('now') WHERE room_id = ? AND user_id = ?",
        (room_id, user_id),
    )
    conn.commit()
    conn.close()


def toggle_room_visibility(room_id):
    conn = _get_db()
    row = conn.execute("SELECT is_public FROM rooms WHERE id = ?", (room_id,)).fetchone()
    if row and row[0] == 0:
        conn.execute("UPDATE rooms SET is_public = 1 WHERE id = ?", (room_id,))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False


def get_messages(room_id, user_id, limit=50):
    conn = _get_db()
    is_member = conn.execute(
        "SELECT 1 FROM room_members WHERE room_id = ? AND user_id = ?",
        (room_id, user_id),
    ).fetchone()
    if not is_member:
        is_public = conn.execute("SELECT is_public FROM rooms WHERE id = ?", (room_id,)).fetchone()
        if not is_public or not is_public[0]:
            conn.close()
            return None
    rows = conn.execute("""
        SELECT m.id, m.content, m.created_at, u.id, u.username, u.display_name
        FROM messages m
        JOIN users u ON u.id = m.user_id
        WHERE m.room_id = ?
        ORDER BY m.created_at ASC
        LIMIT ?
    """, (room_id, limit)).fetchall()
    conn.close()
    return [{"id": r[0], "content": r[1], "created_at": r[2],
             "user_id": r[3], "username": r[4], "display_name": r[5]} for r in rows]


def send_message(room_id, user_id, content):
    conn = _get_db()
    is_member = conn.execute(
        "SELECT 1 FROM room_members WHERE room_id = ? AND user_id = ?",
        (room_id, user_id),
    ).fetchone()
    if not is_member:
        is_public = conn.execute("SELECT is_public FROM rooms WHERE id = ?", (room_id,)).fetchone()
        if not is_public or not is_public[0]:
            conn.close()
            return None
    cur = conn.execute(
        "INSERT INTO messages (room_id, user_id, content) VALUES (?, ?, ?)",
        (room_id, user_id, content),
    )
    msg_id = cur.lastrowid
    conn.execute(
        "INSERT INTO messages_fts (rowid, content) VALUES (?, ?)",
        (msg_id, content),
    )
    row = conn.execute("""
        SELECT m.id, m.content, m.created_at, u.id, u.username, u.display_name
        FROM messages m
        JOIN users u ON u.id = m.user_id
        WHERE m.id = ?
    """, (msg_id,)).fetchone()
    conn.commit()
    conn.close()
    return {"id": row[0], "content": row[1], "created_at": row[2],
            "user_id": row[3], "username": row[4], "display_name": row[5]}


def search_messages(query, user_id, limit=20):
    conn = _get_db()
    rows = conn.execute("""
        SELECT m.id, m.content, m.created_at, m.room_id, r.name,
               u.id, u.username, u.display_name
        FROM messages_fts f
        JOIN messages m ON m.id = f.rowid
        JOIN rooms r ON r.id = m.room_id
        JOIN users u ON u.id = m.user_id
        JOIN room_members rm ON rm.room_id = r.id AND rm.user_id = ?
        WHERE messages_fts MATCH ?
        ORDER BY m.created_at DESC
        LIMIT ?
    """, (user_id, query, limit)).fetchall()
    conn.close()
    return [{"id": r[0], "content": r[1], "created_at": r[2], "room_id": r[3],
             "room_name": r[4], "user_id": r[5], "username": r[6], "display_name": r[7]}
            for r in rows]


def get_recent_rooms(user_id):
    conn = _get_db()
    rows = conn.execute("""
        SELECT DISTINCT r.id, r.name, r.is_public
        FROM messages m
        JOIN rooms r ON r.id = m.room_id
        WHERE m.user_id = ?
        ORDER BY m.created_at DESC
        LIMIT 5
    """, (user_id,)).fetchall()
    conn.close()
    return [{"id": r[0], "name": r[1], "is_public": r[2]} for r in rows]
