"""What the toy remembers across restarts: the faces it was asked to learn, and where it lives.

In the Postgres database at DATABASE_URL (Neon's free plan), so a restart or a redeploy on Render
forgets nothing. Without DATABASE_URL -- the tests, or a database that cannot be reached -- it is kept
in memory instead, and said so once in the log: the toy still works, it only forgets on restart.

Kept per toy (its X-Jarvis-Camera name):
  faces:  a name and one face's fingerprint (SFace's 128 numbers) per picture learned from. Only
          faces someone asked to be learned ("Jarvis, learn my face, I'm Sarah"); a stranger's face
          is compared and let go, never kept. "Jarvis, forget my face" deletes them.
  places: where it lives ("Jarvis, I live in Leeds"): the town, its latitude, longitude, time zone.
"""

import os
import threading

import numpy as np

# A person is known from at most this many pictures; the oldest go first.
KEEP_PER_PERSON = 30

_lock = threading.Lock()
_memory = {"faces": [], "places": {}}
_said_memory = False
_ready = False


def _url():
    return (os.getenv("DATABASE_URL") or "").strip()


def _connect():
    import psycopg

    return psycopg.connect(_url(), connect_timeout=8, autocommit=True)


def _setup(connection):
    global _ready

    if _ready:
        return

    connection.execute(
        "CREATE TABLE IF NOT EXISTS toy_faces (id BIGSERIAL PRIMARY KEY, toy TEXT NOT NULL, name TEXT NOT NULL,"
        " feature BYTEA NOT NULL, learned TIMESTAMPTZ NOT NULL DEFAULT now())")
    connection.execute("CREATE INDEX IF NOT EXISTS toy_faces_toy ON toy_faces (toy)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS toy_places (toy TEXT PRIMARY KEY, city TEXT NOT NULL, lat DOUBLE PRECISION,"
        " lon DOUBLE PRECISION, timezone TEXT)")
    _ready = True


def _in_memory(why=""):
    global _said_memory

    if not _said_memory:
        _said_memory = True
        print(f"[toy-cloud] memory: kept in memory only, forgotten on restart{(' (' + why + ')') if why else ''}",
              flush=True)

    return True


def _use_memory():
    return not _url() and _in_memory("no DATABASE_URL")


def _run(work):
    """[work](connection) on the database; None when it cannot be reached (said in the log)."""
    try:
        with _connect() as connection:
            _setup(connection)
            return work(connection)
    except Exception as error:   # noqa: BLE001 - a database that is down must not stop the toy
        print(f"[toy-cloud] memory: the database failed ({type(error).__name__})", flush=True)
        return None


def _pack(feature):
    return np.asarray(feature, dtype=np.float32).reshape(-1).tobytes()


def _unpack(data):
    return np.frombuffer(bytes(data), dtype=np.float32)


# ---- faces --------------------------------------------------------------------------------------

def add_faces(toy, name, features):
    """Keeps [features] as [name]'s, on [toy]; the oldest past KEEP_PER_PERSON go. True when kept."""
    if _use_memory():
        with _lock:
            _memory["faces"].extend((toy, name, _pack(feature)) for feature in features)
            mine = [row for row in _memory["faces"] if row[0] == toy and row[1] == name]

            for old in mine[:max(0, len(mine) - KEEP_PER_PERSON)]:
                _memory["faces"].remove(old)

        return True

    def work(connection):
        for feature in features:
            connection.execute("INSERT INTO toy_faces (toy, name, feature) VALUES (%s, %s, %s)",
                               (toy, name, _pack(feature)))

        connection.execute(
            "DELETE FROM toy_faces WHERE toy = %s AND name = %s AND id NOT IN (SELECT id FROM toy_faces"
            " WHERE toy = %s AND name = %s ORDER BY id DESC LIMIT %s)", (toy, name, toy, name, KEEP_PER_PERSON))
        return True

    return bool(_run(work))


def faces(toy):
    """[(name, feature), ...] the toy knows."""
    if _use_memory():
        with _lock:
            return [(name, _unpack(data)) for owner, name, data in _memory["faces"] if owner == toy]

    rows = _run(lambda connection: connection.execute(
        "SELECT name, feature FROM toy_faces WHERE toy = %s ORDER BY id", (toy,)).fetchall())
    return [(name, _unpack(data)) for name, data in rows or []]


def names(toy):
    """The names the toy knows, in the order first learned."""
    seen = []

    for name, _feature in faces(toy):
        if name not in seen:
            seen.append(name)

    return seen


def forget_faces(toy, name=None):
    """Deletes [name]'s faces on [toy], or everyone's when [name] is None. How many went, or None when
    the database could not be reached."""
    if _use_memory():
        with _lock:
            before = len(_memory["faces"])
            _memory["faces"] = [row for row in _memory["faces"]
                                if not (row[0] == toy and (name is None or row[1] == name))]
            return before - len(_memory["faces"])

    def work(connection):
        if name is None:
            return connection.execute("DELETE FROM toy_faces WHERE toy = %s", (toy,)).rowcount

        return connection.execute("DELETE FROM toy_faces WHERE toy = %s AND name = %s", (toy, name)).rowcount

    return _run(work)


# ---- where it lives -----------------------------------------------------------------------------

def set_place(toy, place):
    if _use_memory():
        with _lock:
            _memory["places"][toy] = dict(place)

        return True

    return bool(_run(lambda connection: connection.execute(
        "INSERT INTO toy_places (toy, city, lat, lon, timezone) VALUES (%s, %s, %s, %s, %s)"
        " ON CONFLICT (toy) DO UPDATE SET city = EXCLUDED.city, lat = EXCLUDED.lat, lon = EXCLUDED.lon,"
        " timezone = EXCLUDED.timezone",
        (toy, place["city"], place.get("lat"), place.get("lon"), place.get("timezone"))) or True))


def get_place(toy):
    if _use_memory():
        with _lock:
            found = _memory["places"].get(toy)

        return dict(found) if found else None

    row = _run(lambda connection: connection.execute(
        "SELECT city, lat, lon, timezone FROM toy_places WHERE toy = %s", (toy,)).fetchone())

    if not row:
        return None

    return {"city": row[0], "lat": row[1], "lon": row[2], "timezone": row[3]}


def forget_all():
    """Everything in memory (for tests)."""
    with _lock:
        _memory["faces"].clear()
        _memory["places"].clear()
