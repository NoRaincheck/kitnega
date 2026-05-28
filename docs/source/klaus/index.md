---
title: Klaus — Minimal Slack Clone
---

# Klaus

Minimal Slack clone — a chat server with rooms, DMs, and htmx-driven UI. Built
with `lib.bottle` (Bottle web framework from `lib/`), sqlite3, and htmx. Zero JS
framework dependencies — just htmx for partial page swaps.

## Quick Start

```bash
# Initialize the database (creates ~/.kitnega/klaus.db by default)
uv run klaus init-db

# Create an admin user (auto-generates a password if --password omitted)
uv run klaus create-admin admin -p test123

# Start the server on http://127.0.0.1:8080
uv run klaus serve
```

Open `http://127.0.0.1:8080` in your browser. Register a new user from the login
page, or sign in with the admin account.

## CLI

```
klaus [--db PATH] [--secret-file PATH] <command>

Commands:
  serve                      Start the web server
    --host HOST              Bind address (default: 127.0.0.1)
    --port PORT              Port (default: 8080)
    --debug                  Enable debug mode
    --reload                 Auto-reload on file changes

  create-admin <username>    Create an admin user
    --password, -p PASSWORD  Password (auto-generated if omitted)

  init-db                    Initialize the database
    --force                  Overwrite existing database
```

The database and secret file default to `~/.kitnega/klaus.db` and
`~/.kitnega/klaus_secret`. Override with `--db` and `--secret-file` on any
command.

## Creating an Admin

```bash
# Auto-generated password (printed to stdout)
uv run klaus create-admin root

# Or set your own
uv run klaus create-admin root --password "hunter2"
```

Admins can toggle private rooms to public (one-way). Regular members cannot.

## Concepts

### Rooms

| Type    | Visibility                 | Joining                                |
| ------- | -------------------------- | -------------------------------------- |
| Public  | Anyone can see and join    | Click room or use `/rooms/<id>/join`   |
| Private | Only members               | Invite via DM or group DM              |
| DM      | Automatic 1:1 private room | Send `@<user> <message>` from any room |

- Public rooms have names starting with `#`
- Private rooms (DMs, group DMs) have auto-generated `_dm_<id1>_<id2>` names
- A private room **can** be toggled to public (by admin)
- A public room **cannot** be made private

### Direct Messages

Type `@<username> <message>` in any room's message input to send a DM. A private
room is created automatically between you and the target user. Future messages
to the same person reuse the same room.

### Search

Messages are indexed with FTS5. Search via `search_messages()` in the DB layer
(not yet exposed via UI — extend as needed).

## Database Schema

```
users        — id, username, display_name, password_hash, role, created_at
rooms        — id, name, topic, is_public, created_at
room_members — room_id, user_id  (PK on both)
messages     — id, room_id, user_id, content, created_at
messages_fts — FTS5 index on messages.content
```

Passwords are hashed with PBKDF2-HMAC-SHA256 (100k iterations, random 16-byte
salt per user).

## Architecture

```
                   ┌──────────────────┐
Browser ──htmx──►  │  Bottle (WSGI)   │
(no JS framework)  │  lib.bottle      │
                   └────────┬─────────┘
                            │
                   ┌────────▼─────────┐
                   │  sqlite3         │
                    │  ~/.kitnega/klaus.db │
                   └──────────────────┘
```

Auth is handled via signed cookies (`user_id` stored in cookie, verified with
HMAC-SHA256 using the secret from `~/.kitnega/klaus_secret`).

## Project Layout

`klaus/` is a `uv` workspace member. The server code lives in `klaus/klaus/`:
`__main__.py` (CLI), `app.py` (Bottle routes), `db.py` (SQLite layer), and a
`views/` directory of htmx templates.

## Known Issues

- **Leave room is non-functional** — the `/rooms/<id>/leave` endpoint exists but
  does not work correctly. Requires investigation.
- **Room ordering** — there is no way to reorder or hide rooms and DMs in the
  sidebar. Rooms are sorted by most recent message, which can be noisy.

## Tests

```bash
uv run pytest tests/test_klaus.py -v
```

Tests cover the database layer (user auth, room CRUD, messages, access control)
and app-layer integration (routes, auth flow, cookie handling).
