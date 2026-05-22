import os
import re

from lib.bottle import TEMPLATE_PATH, Bottle, redirect, request, response, static_file, template

from . import db

app = Bottle()

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH.insert(0, os.path.join(HERE, "views"))

SECRET = None


def configure(secret, db_path):
    global SECRET
    SECRET = secret
    db.init_db(db_path)


def _enrich_room(room, user_id):
    room = dict(room)
    name = room["name"]
    if name.startswith("_dm_"):
        parts = name.split("_")
        other_id = int(parts[2]) if int(parts[2]) != user_id else int(parts[3])
        other = db.get_user(other_id)
        room["title"] = other["display_name"] or other["username"] if other else name
    else:
        room["title"] = name
    return room


def _render_content(content):
    def _mention_link(m):
        username = m.group(1)
        return (
            f'<a href="#" hx-post="/dm/{username}" hx-target="body" hx-swap="innerHTML" class="mention">@{username}</a>'
        )

    return re.sub(r"@(\w+)", _mention_link, content)


def _current_user():
    uid = request.get_cookie("user_id", secret=SECRET)
    if uid:
        return db.get_user(int(uid))
    return None


def _require_user():
    user = _current_user()
    if not user:
        redirect("/login")
    return user


@app.hook("before_request")
def _setup_context():
    pass


@app.get("/static/<filepath:path>")
def serve_static(filepath):
    static_dir = os.path.join(HERE, "static")
    return static_file(filepath, root=static_dir)


@app.get("/")
def index():
    user = _current_user()
    if not user:
        redirect("/login")
    rooms = [_enrich_room(r, user["id"]) for r in db.get_rooms(user["id"])]
    room_id = request.query.get("room")
    active_room = None
    messages = []
    if room_id:
        active_room = db.get_room(int(room_id))
        if active_room:
            active_room = _enrich_room(active_room, user["id"])
            msgs = db.get_messages(int(room_id), user["id"])
            if msgs is not None:
                messages = msgs
            db.mark_read(int(room_id), user["id"])
    if not active_room and rooms:
        active_room = rooms[0]
        msgs = db.get_messages(active_room["id"], user["id"])
        if msgs is not None:
            messages = msgs
        db.mark_read(active_room["id"], user["id"])
    users = db.list_users()
    return template(
        "chat",
        user=user,
        rooms=rooms,
        active_room=active_room,
        messages=messages,
        users=users,
        render_content=_render_content,
    )


@app.get("/login")
def login_form():
    if _current_user():
        redirect("/")
    return template("login")


@app.post("/login")
def login_submit():
    username = request.forms.get("username", "").strip()
    password = request.forms.get("password", "")
    uid = db.authenticate(username, password)
    if uid:
        response.set_cookie("user_id", str(uid), secret=SECRET, path="/", max_age=86400 * 30)
        redirect("/")
    return template("login", error="Invalid username or password")


@app.get("/register")
def register_form():
    if _current_user():
        redirect("/")
    return template("register")


@app.post("/register")
def register_submit():
    username = request.forms.get("username", "").strip()
    display_name = request.forms.get("display_name", "").strip() or username
    password = request.forms.get("password", "")
    if not username or not password:
        return template("register", error="Username and password are required")
    if not re.match(r"^[a-zA-Z0-9_]+$", username):
        return template("register", error="Username can only contain letters, numbers, and underscores")
    uid = db.create_user(username, display_name, password)
    if not uid:
        return template("register", error="Username already taken")
    response.set_cookie("user_id", str(uid), secret=SECRET, path="/", max_age=86400 * 30)
    redirect("/")


@app.post("/logout")
def logout():
    response.delete_cookie("user_id", path="/")
    redirect("/login")


@app.get("/rooms")
def room_list():
    user = _require_user()
    rooms = [_enrich_room(r, user["id"]) for r in db.get_rooms(user["id"])]
    return template("_room_list", rooms=rooms, user=user)


@app.post("/rooms")
def room_create():
    user = _require_user()
    name = request.forms.get("name", "").strip()
    topic = request.forms.get("topic", "").strip()
    if not name:
        return "Room name is required"
    if not re.match(r"^[a-zA-Z0-9 -]+$", name):
        return "Room name can only contain letters, numbers, spaces, and dashes"
    is_public = request.forms.get("is_public", "1") == "1"
    room_id = db.create_room(name, topic, is_public, user["id"])
    if not room_id:
        return "Room name already exists"
    redirect(f"/?room={room_id}")


@app.get("/rooms/<room_id>/messages")
def room_messages(room_id):
    user = _require_user()
    msgs = db.get_messages(int(room_id), user["id"])
    if msgs is None:
        redirect("/")
    db.mark_read(int(room_id), user["id"])
    active_room = _enrich_room(db.get_room(int(room_id)), user["id"])
    return template("_messages", messages=msgs, active_room=active_room, user=user, render_content=_render_content)


@app.post("/rooms/<room_id>/messages")
def room_send_message(room_id):
    user = _require_user()
    content = request.forms.get("content", "").strip()
    if not content:
        return ""
    msg = db.send_message(int(room_id), user["id"], content)
    if not msg:
        return ""
    active_room = _enrich_room(db.get_room(int(room_id)), user["id"])
    return template("_messages", messages=[msg], active_room=active_room, user=user, render_content=_render_content)


@app.post("/rooms/<room_id>/join")
def room_join(room_id):
    user = _require_user()
    db.join_room(int(room_id), user["id"])
    response.set_header("HX-Refresh", "true")
    return ""


@app.post("/rooms/<room_id>/leave")
def room_leave(room_id):
    user = _require_user()
    db.leave_room(int(room_id), user["id"])
    response.set_header("HX-Refresh", "true")
    return ""


@app.post("/rooms/<room_id>/toggle-visibility")
def room_toggle_visibility(room_id):
    user = _require_user()
    room = db.get_room(int(room_id))
    if not room or room["is_public"]:
        return "Room is already public or not found"
    if user["role"] != "admin":
        return "Only admins can toggle room visibility"
    db.toggle_room_visibility(int(room_id))
    response.set_header("HX-Refresh", "true")
    return ""


@app.get("/users")
def user_list():
    user = _require_user()
    users = db.list_users()
    return template("_user_list", users=users, current_user=user)


@app.post("/dm/<username>")
def dm_start(username):
    user = _require_user()
    target = db.get_user_by_username(username)
    if not target or target["id"] == user["id"]:
        response.set_header("HX-Redirect", "/")
        return
    room_id = db.get_dm_room(user["id"], target["id"])
    response.set_header("HX-Redirect", f"/?room={room_id}")


@app.post("/dm")
def dm_send():
    user = _require_user()
    username = request.forms.get("username", "").strip()
    if not username:
        return "Enter a username"
    target = db.get_user_by_username(username)
    if not target or target["id"] == user["id"]:
        return "User not found"
    room_id = db.get_dm_room(user["id"], target["id"])
    response.set_header("HX-Redirect", f"/?room={room_id}")
