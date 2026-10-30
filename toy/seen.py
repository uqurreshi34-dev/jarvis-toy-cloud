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
import threading
import time

AWAY_SECONDS = 30.0
MAX_OBJECTS = 50
MAX_LABEL = 40

# Said sentences wait this long for the toy to fetch them, then are stale.
PENDING_SECONDS = 60.0

_NUMBERS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}

_IRREGULAR = {"person": "people", "mouse": "mice", "knife": "knives", "sheep": "sheep", "skis": "skis",
              "scissors": "scissors", "bus": "buses", "wine glass": "wine glasses", "couch": "couches",
              "sandwich": "sandwiches", "toothbrush": "toothbrushes", "hair drier": "hair driers"}

_lock = threading.Lock()
_present = {}    # (toy, label) -> {"last": monotonic time last in view, "said": how many said}
_pending = {}    # toy -> (sentence, monotonic time)


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
            _pending[toy] = (words(due), now)

    return bool(due)


def take(toy, now=None):
    """The sentence waiting for [toy], once; None when there is none (or it went stale)."""
    now = time.monotonic() if now is None else now

    with _lock:
        waiting = _pending.pop(toy, None)

    if not waiting or now - waiting[1] > PENDING_SECONDS:
        return None

    return waiting[0]


def forget():
    """Everything, as after a restart (for tests)."""
    with _lock:
        _present.clear()
        _pending.clear()
