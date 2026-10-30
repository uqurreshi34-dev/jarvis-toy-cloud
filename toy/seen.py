"""What the toy's camera sees, and what the toy says about it, in cloud mode.

The toy runs YOLO26n on its own chip and posts what it is sure of to POST /camera/seen (names,
scores, where; never the picture), as it does to PC JARVIS. Ported from JARVIS's
actions/detections.py: a thing is news when it comes into view, and again only once it has been
out of view for AWAY_SECONDS, or when more of it come into view than were said. The words are
the toy's own, about itself: "I can see someone, sir." The toy fetches them from /toy/announce
and says them on its speaker.

TOY_DETECT_IGNORE names things never said ("dining table,sink": what the model takes furniture
for). Nothing here is kept beyond memory.
"""

import os
import re
import threading
import time

AWAY_SECONDS = 30.0
MAX_OBJECTS = 50
MAX_LABEL = 40

# Said sentences wait this long for the toy to fetch them, then are stale.
PENDING_SECONDS = 60.0

# Someone coming into view is held this long for their face, which the toy sends a moment after its
# report: "I can see you, Umer." rather than "I can see someone, sir." and then nothing. A stranger's
# face lets it go at once; with no face at all it goes with the toy's next report (15 s at most).
FACE_WAIT = 5.0

_NUMBERS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

_IRREGULAR = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep", "skis": "skis",
              "scissors": "scissors", "bus": "buses", "wine glass": "wine glasses", "couch": "couches",
              "sandwich": "sandwiches", "toothbrush": "toothbrushes", "hair drier": "hair driers"}

_lock = threading.Lock()
_present = {}    # (toy, label) -> {"last": monotonic time last in view, "said": how many said}
_pending = {}    # toy -> [{"said", "at", "until", "counts"}, ...], said in order; held until "until"
_latest = {}     # toy -> ({label: count}, monotonic time): its last report

# The toy reports every change, and every 15 seconds while anything is in view: a report older
# than this, with things in it, is not what is in view now.
FRESH_SECONDS = 40.0


def _ignored():
    return {name.strip().casefold() for name in (os.getenv("TOY_DETECT_IGNORE") or "").split(",") if name.strip()}


def _clean(payload):
    """The objects in a report, or None when it is not one."""
    if not isinstance(payload, dict) or not isinstance(payload.get("objects"), list):
        return None

    objects = payload["objects"]

    if len(objects) > MAX_OBJECTS:
        return None

    kept = []

    for item in objects:
        if not isinstance(item, dict):
            return None

        label = item.get("label")

        if not isinstance(label, str) or not label.strip() or len(label) > MAX_LABEL:
            return None

        kept.append(label.strip().casefold())

    return kept


def _counted(label, count):
    if count == 1:
        return f"{'an' if label[:1] in 'aeiou' else 'a'} {label}"

    plural = _IRREGULAR.get(label) or (label + ("es" if label.endswith(("s", "x", "ch", "sh")) else "s"))
    return f"{_NUMBERS.get(count, count)} {plural}"


def words(arrived):
    """What the toy says when [arrived] ({label: how many}) comes into view, someone first:
    "I can see someone, sir.", "I can see someone and two cups, sir."."""
    order = sorted(arrived, key=lambda label: (label != "person", label))
    parts = []

    for label in order:
        count = arrived[label]

        if label == "person":
            parts.append("someone" if count == 1 else f"{_NUMBERS.get(count, count)} people")
        else:
            parts.append(_counted(label, count))

    said = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    return f"I can see {said}, sir."


def record(toy, payload, now=None):
    """Takes a report; True when something is now worth saying (fetch it with take()), False when
    not, None when the report is not one."""
    labels = _clean(payload)

    if labels is None:
        return None

    now = time.monotonic() if now is None else now
    counts = {}

    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    ignored = _ignored()
    due = {}

    with _lock:
        _latest[toy] = (dict(counts), now)

        for label, count in counts.items():
            seen = _present.get((toy, label))

            if seen is None or now - seen["last"] > AWAY_SECONDS:
                seen = {"last": now, "said": 0}

            seen["last"] = now
            _present[(toy, label)] = seen

            if count > seen["said"] and label not in ignored:
                seen["said"] = count
                due[label] = count

        if due:
            until = now + FACE_WAIT if "person" in due else now
            _pending.setdefault(toy, []).append({"said": words(due), "at": now, "until": until, "counts": due})

    return ready(toy, now)


def queue(toy, sentence, now=None):
    """Something else for [toy] to say unasked (a greeting, how learning a face went)."""
    now = time.monotonic() if now is None else now

    with _lock:
        _pending.setdefault(toy, []).append({"said": sentence, "at": now, "until": now, "counts": None})


def ready(toy, now=None):
    """True when something for [toy] can be said now (not held for a face, not stale)."""
    now = time.monotonic() if now is None else now

    with _lock:
        return any(item["until"] <= now and now - item["at"] <= PENDING_SECONDS for item in _pending.get(toy, []))


def take(toy, now=None, everything=False):
    """What is ready for [toy], once, in order; None when nothing is. Held sentences stay, unless
    [everything] (a muted toy drops them all)."""
    now = time.monotonic() if now is None else now

    with _lock:
        waiting = _pending.pop(toy, [])
        held = [] if everything else [item for item in waiting if item["until"] > now]

        if held:
            _pending[toy] = held

    fresh = [item["said"] for item in waiting
             if item["until"] <= now and now - item["at"] <= PENDING_SECONDS]
    return " ".join(fresh) if fresh else None


def drop_arrival(toy, now=None):
    """Lets go of a held "I can see someone, sir." about one person and nothing else: the cloud saw a
    face it knows a moment ago, so it is them moving about (turning to leave, back to the camera)."""
    now = time.monotonic() if now is None else now

    with _lock:
        waiting = _pending.get(toy, [])
        kept = [item for item in waiting if not (item["until"] > now and item["counts"] == {"person": 1})]
        dropped = len(kept) != len(waiting)

        if waiting:
            _pending[toy] = kept

    return dropped


def face_arrived(toy, name, greeting=None, now=None):
    """A face came after someone came into view: a held "I can see someone, sir." becomes "I can see
    you, Umer." ([greeting], when it is time for one, "Good evening, Umer.", in its place), or, for a
    stranger ([name] None), is let go as it was. True when there was one held to change."""
    now = time.monotonic() if now is None else now

    with _lock:
        item = next((item for item in _pending.get(toy, [])
                     if item["until"] > now and item["counts"] and "person" in item["counts"]), None)

        if item is None:
            return False

        item["until"] = now

        if not name:
            return True

        counts = item["counts"]
        people = counts["person"]
        others = {label: count for label, count in counts.items() if label != "person"}
        extras = []

        if people > 1:
            extras.append(f"{_NUMBERS.get(people - 1, people - 1)} other {'person' if people == 2 else 'people'}")

        if others:
            extras.append(words(others)[len("I can see "):-len(", sir.")])

        extra = " and ".join(extras)

        if greeting:
            item["said"] = f"{greeting} I can see {extra} too." if extra else greeting
        else:
            item["said"] = f"I can see you, {name}, and {extra}." if extra else f"I can see you, {name}."

    return True


_SIGHT = re.compile(
    r"^(?:what|who)\s+(?:can|do)\s+you\s+see(?:\s+(?:now|right\s+now|there))?"
    r"|^(?:whats|whos|(?:what|who)\s+is)\s+in\s+(?:front\s+of\s+you|view|the\s+room)"
    r"|^(?:is\s+)?(?:any\s*one|any\s*body|someone|somebody)\s+(?:there|here|in\s+view)"
    r"|^can\s+you\s+see\s+(?:me|any\s*one|any\s*body|anything)"
    r"|^what\s+are\s+you\s+looking\s+at"
)


def sight_asked(text):
    """True when [text] asks what the toy can see ("what can you see", "is anyone there")."""
    words = " ".join(re.sub(r"[^\w\s]", " ", str(text or "").casefold().replace("'", "")).split())
    words = " ".join(word for word in words.split() if word not in ("jarvis", "sir"))
    return bool(_SIGHT.match(words))


def describe(toy, name=None, now=None):
    """What the toy can see now, said, from its last report; [name] is the person it last knew."""
    now = time.monotonic() if now is None else now

    with _lock:
        latest = _latest.get(toy)

    if not latest:
        return "My eyes haven't told me anything yet, sir. Give me a moment."

    counts, at = latest

    if not counts or now - at > FRESH_SECONDS:
        return "Nothing I recognise right now, sir."

    if name and "person" in counts:
        others = {label: count for label, count in counts.items() if label != "person"}
        people = counts["person"]
        said = name if people == 1 else f"{name} and {_NUMBERS.get(people - 1, people - 1)} " \
            f"{'other person' if people == 2 else 'other people'}"

        if others:
            rest = words(others)[len("I can see "):-len(", sir.")]
            return f"I can see you, {name}, and {rest}." if people == 1 else f"I can see {said}, and {rest}."

        return f"I can see you, {name}." if people == 1 else f"I can see {said}."

    return words(counts)


def forget():
    """Everything, as after a restart (for tests)."""
    with _lock:
        _present.clear()
        _pending.clear()
        _latest.clear()
