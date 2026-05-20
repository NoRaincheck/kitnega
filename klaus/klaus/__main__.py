import argparse
import os
import sys

from . import app as chat_app
from . import db

DEFAULT_PORT = 8080
DEFAULT_DB = os.path.expanduser("~/.kitnega/klaus.db")
DEFAULT_SECRET = os.path.expanduser("~/.kitnega/klaus_secret")


def _get_or_create_secret(path):
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    import secrets
    secret = secrets.token_hex(32)
    with open(path, "w") as f:
        f.write(secret)
    os.chmod(path, 0o600)
    return secret


def cmd_serve(args):
    secret = _get_or_create_secret(args.secret_file)
    chat_app.configure(secret, args.db)
    chat_app.app.run(host=args.host, port=args.port, debug=args.debug, reloader=args.reload)


def cmd_create_admin(args):
    password = args.password
    if not password:
        password = os.urandom(12).hex()
    secret = _get_or_create_secret(args.secret_file)
    chat_app.configure(secret, args.db)
    uid = db.create_admin(args.username, password)
    if uid:
        print(f"Admin user '{args.username}' created (id={uid}).")
        if not args.password:
            print(f"Password: {password}")
    else:
        print(f"User '{args.username}' already exists.", file=sys.stderr)
        sys.exit(1)


def cmd_init_db(args):
    try:
        db.init_db(args.db, force=args.force)
    except FileExistsError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    label = "reinitialized" if args.force else "initialized"
    print(f"Database {label} at {args.db}")


def main():
    parser = argparse.ArgumentParser(
        description="klaus — minimal Slack clone",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db", default=DEFAULT_DB, help=f"Database path (default: {DEFAULT_DB})")
    parser.add_argument("--secret-file", default=DEFAULT_SECRET,
                        help=f"Secret file path (default: {DEFAULT_SECRET})")

    sub = parser.add_subparsers(dest="command")
    sub.required = True

    serve_p = sub.add_parser("serve", help="Start the web server")
    serve_p.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    serve_p.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default: {DEFAULT_PORT})")
    serve_p.add_argument("--debug", action="store_true", help="Enable debug mode")
    serve_p.add_argument("--reload", action="store_true", help="Auto-reload on file changes")
    serve_p.set_defaults(func=cmd_serve)

    admin_p = sub.add_parser("create-admin", help="Create an admin user")
    admin_p.add_argument("username", help="Admin username")
    admin_p.add_argument("--password", "-p", default="", help="Password (auto-generated if omitted)")
    admin_p.set_defaults(func=cmd_create_admin)

    init_p = sub.add_parser("init-db", help="Initialize the database")
    init_p.add_argument("--force", action="store_true", help="Overwrite existing database")
    init_p.set_defaults(func=cmd_init_db)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
