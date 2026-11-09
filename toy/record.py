"""One record per thing the toy asked of the cloud, kept in the database, read back as a report.

Render keeps its logs for a short while only; these stay (in Neon, beside the faces). Like PC JARVIS's
request_log, a record holds how a request was handled, never what was said or seen: so it can be read
and shared to make the toy faster and more offline without giving anything away.

A talk record (kind "talk"): which toy, how it was heard (button, sound, or "wake" on the chip), what
it was (intent: "time", "affection", "radio", "question" ... -- a label, not the words), how it was
answered ("free": the cloud's own phrases, which a phrase list on the chip could do; "model": the AI;
"search": the AI after a web search), each AI call (provider, ms, ok), what failed over, how long
hearing, answering and the voice took, and the whole of it.
A look record (kind "look", "Hey Jarvis, what can you see?") and a check record (kind "check", the
second look at something new): how long the big brain took and whether anything was said.
Faces and the toy's own sightings come often, so they are counted per day instead (toy_counts),
written with the next record, or every six hours.

Written in the background: the toy's answer never waits for the database. Kept 90 days. Never raises.

    "Hey Jarvis, request report"    spoken by the toy: the last 7 days
    GET /report (app.py)            the same as a page, behind TOY_REPORT_PASSWORD
"""

import json
import statistics
import threading
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

from toy import store

KEEP_DAYS = 90
# Counts ride along with the next record written, and are written on their own only this often: Neon's
# free plan sleeps its database after five quiet minutes, and a write every few minutes would keep it
# awake all day, using up the month's hours.
COUNTS_EVERY_SECONDS = 6 * 3600.0
MEMORY_KEEP = 5000

# What a chip could answer by itself, with no internet once it can hear these few commands: fixed replies
# that need nothing from outside. Not the weather (a forecast service), the report (kept in the database),
# what it can see (the big brain), faces (the database), music, or any question. The time needs the
# chip's clock set from the internet once after switching on, then holds.
CHIP_INTENTS = frozenset({"time", "affection", "social", "lights", "mute", "unmute", "name alone", "switch mode"})

_local = threading.local()
_lock = threading.Lock()
_memory = {"records": [], "counts": Counter()}
_pending = {"counts": Counter(), "flushed": time.monotonic(), "pruned": 0.0}


def later(work, *args):
    """[work] in the background, so the toy never waits on the database (the tests run it at once)."""
    threading.Thread(target=work, args=args, name="toy-record", daemon=True).start()


# ---- the record being made ----------------------------------------------------------------------

def begin(kind, toy, **fields):
    """A record starts, for this request's thread; returned, for end()."""
    record = {"kind": kind, "toy": toy or "toy", "started": time.monotonic(), "ai": [], "fallbacks": [],
              "searches": 0, "ms": {}, **fields}
    _local.record = record
    return record


def _current():
    return getattr(_local, "record", None)


def note(**fields):
    """Adds [fields] to the record being made (intent, outcome ...); the first intent wins."""
    record = _current()

    if record is None:
        return

    if "intent" in fields and record.get("intent"):
        fields.pop("intent")

    record.update(fields)


def timed(part, seconds):
    """How long [part] ("hearing", "answer", "voice", "brain") took."""
    record = _current()

    if record is not None and seconds is not None:
        record["ms"][part] = int(seconds * 1000)


def ai_call(provider, seconds, ok):
    record = _current()

    if record is not None:
        record["ai"].append({"provider": provider, "ms": int(seconds * 1000), "ok": bool(ok)})


def fallback(what):
    """A part that failed over: "groq>gemini", "gemini>none"."""
    record = _current()

    if record is not None:
        record["fallbacks"].append(what)


def searched():
    record = _current()

    if record is not None:
        record["searches"] += 1


def end(record, outcome=None):
    """[record] (from begin) is over: written in the background. Twice is harmless."""
    if record is None or "started" not in record:
        return None

    if _current() is record:
        _local.record = None

    record["ms"]["total"] = int((time.monotonic() - record.pop("started")) * 1000)

    if outcome:
        record["outcome"] = outcome

    if record["kind"] == "talk":
        record.setdefault("outcome", "answered")   # every silent or refused way notes its own
        record["route"] = "search" if record["searches"] else "model" if record["ai"] else "free"

    later(_write, record)
    later(flush_counts)
    return record


def count(toy, what, times=1):
    """Something frequent ("face", "seen"), counted for today and written every few minutes."""
    day = datetime.now(timezone.utc).date().isoformat()

    with _lock:
        _pending["counts"][(toy or "toy", day, what)] += times
        due = time.monotonic() - _pending["flushed"] >= COUNTS_EVERY_SECONDS

    if due:
        later(flush_counts)


# ---- the database -------------------------------------------------------------------------------

def _setup(connection):
    connection.execute(
        "CREATE TABLE IF NOT EXISTS toy_requests (id BIGSERIAL PRIMARY KEY, toy TEXT NOT NULL,"
        " at TIMESTAMPTZ NOT NULL DEFAULT now(), kind TEXT NOT NULL, record JSONB NOT NULL)")
    connection.execute("CREATE INDEX IF NOT EXISTS toy_requests_at ON toy_requests (at)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS toy_counts (toy TEXT NOT NULL, day DATE NOT NULL, what TEXT NOT NULL,"
        " n BIGINT NOT NULL, PRIMARY KEY (toy, day, what))")


def _write(record):
    try:
        record = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **record}

        if store._use_memory():
            with _lock:
                _memory["records"].append(record)
                del _memory["records"][:-MEMORY_KEEP]
            return

        def work(connection):
            _setup(connection)
            connection.execute("INSERT INTO toy_requests (toy, kind, record) VALUES (%s, %s, %s::jsonb)",
                               (record["toy"], record["kind"], json.dumps(record)))

            # Old records go, once a day per process.
            if time.monotonic() - _pending["pruned"] > 86400 or not _pending["pruned"]:
                _pending["pruned"] = time.monotonic()
                connection.execute("DELETE FROM toy_requests WHERE at < now() - make_interval(days => %s::int)",
                                   (KEEP_DAYS,))
                connection.execute("DELETE FROM toy_counts WHERE day < current_date - %s::int", (KEEP_DAYS,))

            return True

        store._run(work)
    except Exception as error:   # noqa: BLE001 - a record must never cost the toy its answer
        print(f"[toy-cloud] record: not kept ({type(error).__name__})", flush=True)


def flush_counts():
    """Writes the counts made since the last time."""
    with _lock:
        counts, _pending["counts"] = _pending["counts"], Counter()
        _pending["flushed"] = time.monotonic()

    if not counts:
        return

    try:
        if store._use_memory():
            with _lock:
                _memory["counts"].update(counts)
            return

        def work(connection):
            _setup(connection)

            for (toy, day, what), n in counts.items():
                connection.execute(
                    "INSERT INTO toy_counts (toy, day, what, n) VALUES (%s, %s, %s, %s) ON CONFLICT (toy, day, what)"
                    " DO UPDATE SET n = toy_counts.n + EXCLUDED.n", (toy, day, what, n))

            return True

        if not store._run(work):
            with _lock:
                _pending["counts"].update(counts)   # tried again next time
    except Exception as error:   # noqa: BLE001
        print(f"[toy-cloud] record: counts not kept ({type(error).__name__})", flush=True)


def records(days=7):
    """The records of the last [days] days, oldest first."""
    since = datetime.now(timezone.utc) - timedelta(days=days)

    if store._use_memory():
        with _lock:
            return [dict(record) for record in _memory["records"] if datetime.fromisoformat(record["at"]) >= since]

    rows = store._run(lambda connection: (_setup(connection), connection.execute(
        "SELECT record FROM toy_requests WHERE at >= %s ORDER BY id", (since,)).fetchall())[1])
    return [row[0] if isinstance(row[0], dict) else json.loads(row[0]) for row in rows or []]


def counts(days=7):
    """{what: how many} over the last [days] days, the counts not yet written included."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
    found = Counter()

    with _lock:
        waiting = Counter({key: n for key, n in _pending["counts"].items() if key[1] >= since})
        kept = Counter({key: n for key, n in _memory["counts"].items() if key[1] >= since}) \
            if store._use_memory() else Counter()

    for (_toy, _day, what), n in (waiting + kept).items():
        found[what] += n

    if not store._use_memory():
        rows = store._run(lambda connection: (_setup(connection), connection.execute(
            "SELECT what, sum(n) FROM toy_counts WHERE day >= %s::date GROUP BY what", (since,)).fetchall())[1])

        for what, n in rows or []:
            found[what] += int(n)

    return dict(found)


def forget_all():
    """Everything in memory (for tests)."""
    with _lock:
        _memory["records"].clear()
        _memory["counts"].clear()
        _pending["counts"].clear()


# ---- the report ---------------------------------------------------------------------------------

def _median(values):
    values = [value for value in values if isinstance(value, (int, float))]
    return statistics.median(values) if values else None


def summary(days=7, found=None, counted=None):
    """The numbers, as a dict."""
    found = records(days) if found is None else found
    counted = counts(days) if counted is None else counted
    talks = [record for record in found if record.get("kind") == "talk"]
    answered = [record for record in talks if record.get("outcome") == "answered"]
    looks = [record for record in found if record.get("kind") == "look"]
    checks = [record for record in found if record.get("kind") == "check"]
    calls = [call for record in found for call in record.get("ai") or []]

    return {
        "days": days,
        "talks": len(talks),
        "answered": len(answered),
        "outcomes": dict(Counter(record.get("outcome") for record in talks)),
        "heard_by": dict(Counter(record.get("heard_by") for record in talks)),
        "routes": dict(Counter(record.get("route") for record in answered)),
        "intents": dict(Counter(record.get("intent") for record in answered).most_common()),
        "free": sum(1 for record in answered if record.get("route") == "free"),
        "chip": sum(1 for record in answered if record.get("intent") in CHIP_INTENTS),
        "model": sum(1 for record in answered if record.get("route") == "model"),
        "search": sum(1 for record in answered if record.get("route") == "search"),
        "hearing_ms": _median((record.get("ms") or {}).get("hearing") for record in talks),
        "answer_ms": _median((record.get("ms") or {}).get("answer") for record in answered),
        "voice_ms": _median((record.get("ms") or {}).get("voice") for record in answered),
        "total_ms": _median((record.get("ms") or {}).get("total") for record in answered),
        "slowest_ms": max(((record.get("ms") or {}).get("total") or 0 for record in answered), default=None),
        "voice_chars": sum(record.get("chars") or 0 for record in answered),
        "ai_calls": len(calls),
        "ai_by": dict(Counter(call.get("provider") for call in calls)),
        "ai_ms": _median(call.get("ms") for call in calls if call.get("ok")),
        "fallbacks": dict(Counter(what for record in found for what in record.get("fallbacks") or [])),
        "looks": len(looks),
        "look_ms": _median((record.get("ms") or {}).get("brain") for record in looks),
        "checks": len(checks),
        "checks_said": sum(1 for record in checks if record.get("said")),
        "check_ms": _median((record.get("ms") or {}).get("brain") for record in checks),
        "faces": counted.get("face", 0),
        "seen": counted.get("seen", 0),
    }


def _seconds(ms):
    return f"{ms / 1000:.1f} seconds"


def _plural(count, word):
    return f"{count} {word}{'' if count == 1 else 's'}"


def report_words(days=7):
    """The report, as the toy says it."""
    numbers = summary(days)
    span = "today" if days == 1 else f"in the last {days} days"

    if not numbers["talks"] and not numbers["looks"] and not numbers["checks"]:
        return f"I have nothing on record {span}, sir. I keep a record from now on."

    parts = [f"Request report, sir. You asked me {_plural(numbers['answered'], 'thing')} {span}."]

    if numbers["answered"]:
        free = numbers["free"]
        parts.append(f"Of those, {free} {'was' if free == 1 else 'were'} answered with no AI, "
                     f"{numbers['model']} needed the AI, and {numbers['search']} a web search as well. "
                     f"{numbers['chip']} could work on my chip with no internet.")

    if numbers["total_ms"] is not None:
        parts.append(f"An answer took {_seconds(numbers['total_ms'])} typically, "
                     f"{_seconds(numbers['slowest_ms'])} at the slowest.")

    if numbers["checks"]:
        turned = numbers["checks"] - numbers["checks_said"]
        parts.append(f"My second look checked {_plural(numbers['checks'], 'sighting')} "
                     f"and turned {turned} away.")

    failed = sum(numbers["fallbacks"].values())

    if failed:
        parts.append(f"Parts failed over {_plural(failed, 'time')}.")

    return " ".join(parts)


REPORT_WORDS = frozenset({"request report", "weekly report", "usage report", "give me a request report",
                          "give me a weekly report", "give me a usage report", "how much did you use the internet"})


def report_asked(text):
    words = " ".join(word.strip(".,!?'\"") for word in str(text or "").casefold().split())
    return words in REPORT_WORDS
