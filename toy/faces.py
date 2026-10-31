"""Whose face the toy sees: anyone who asked it to learn theirs, by name; nobody else.

The toy's own face finder (YuNet, on its chip) sends the picture of a look with a face in it to
POST /camera/face. Here OpenCV finds the face again with its five points (YuNet), cuts it to 112 x 112
the way SFace was trained (FaceRecognizerSF.alignCrop), and takes its fingerprint: SFace's 128
numbers, as PC JARVIS's actions/known_faces.py. Alike is their cosine.

    "Jarvis, learn my face, I'm Sarah"   the next LEARN_FROM faces the toy sends are Sarah's
                                         (it is asked for them: X-Jarvis-Faces on the answer)
    "Jarvis, who am I?"                  the face it saw last, by name, if it knows it
    "Jarvis, who do you know?"           the names
    "Jarvis, forget my face"             the only face it knows, or ask whose
    "Jarvis, forget Sarah"               Sarah's
    "Jarvis, forget everyone"            all of them

A known face coming into view is greeted by name ("Good evening, Sarah."), once in GREET_AGAIN.
A stranger's face is compared and let go: nothing about it is kept.

Models: YuNet (MIT) and SFace (Apache 2.0) from OpenCV's model zoo, checked against their SHA-256
as PC JARVIS's tools/face_model.py checks them. Fetched while Render builds (python -m toy.faces, in
the build command) into models/ beside the code, so a restart has them at once; fetched on first
use when they are not there. Loaded in the background as the service starts: /health?detail=1
says "ready", "loading", or "failed: why".
"""

import hashlib
import os
import re
import threading
import time
from pathlib import Path

# Imported here, once, as the app starts, on its main thread. Imported inside the functions, the first
# face's request thread and the start-up thread that loads the models imported it at the same time,
# and every face failed: "partially initialized module 'cv2' has no attribute 'imdecode'" (30 October 2026).
import cv2
import numpy as np
import requests

from toy import store
from toy.wake import WAKE_PHRASE

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODELS = {
    "finder": ("face_detection_yunet_2023mar.onnx", ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
               "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "fingerprint": ("face_recognition_sface_2021dec.onnx",
                    ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
                    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}
MODEL_FOLDER = Path(os.getenv("TOY_MODEL_FOLDER") or Path(__file__).resolve().parents[1] / "models")

# As alike as this, from one picture: the same person. OpenCV's own for SFace is 0.363, over many
# pictures; one picture from a small camera needs more. PC JARVIS measured you 0.38 to 0.50 by day
# and night against a fingerprint learned in the other light, and strangers 0.25 at most.
SAME_AT_LEAST = 0.40

# Learning: this many faces, each at least this like the others, within LEARN_SECONDS.
LEARN_FROM = 4
LEARN_LIKE_THE_REST = 0.30
LEARN_SECONDS = 60.0

FINDER_SURE = 0.6
GREET_AGAIN = 600.0
# Back after this long out of sight (the toy sends a face in view each minute): "I can see you, Umer."
BACK_AFTER = 90.0
NAME_MOST = 30

_lock = threading.Lock()          # who is learning, seen, greeted: held only for a moment
_model_lock = threading.Lock()    # the models: fetched, loaded and run one at a time, never under _lock
_models = {}
_learning = {}    # toy -> {"name", "until", "features": []}
_last_seen = {}   # toy -> (name or None, monotonic time)
_greeted = {}     # (toy, name) -> monotonic time
_seen_at = {}     # (toy, name) -> monotonic time their face last came


# ---- the models ---------------------------------------------------------------------------------

def _model_file(which):
    name, url, digest = MODELS[which]
    path = MODEL_FOLDER / name

    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest:
        return path

    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=(10, 120))
    response.raise_for_status()

    if hashlib.sha256(response.content).hexdigest() != digest:
        raise RuntimeError(f"{name} is not the file expected")

    partial = path.with_suffix(".partial")
    partial.write_bytes(response.content)
    partial.replace(path)
    print(f"[toy-cloud] faces: fetched {name}", flush=True)
    return path


_status = {"now": "loading", "since": time.monotonic()}
MODEL_WAIT = 20.0   # seconds a face waits for the models before it is let go, never hanging the toy


def _say(text):
    print(f"[toy-cloud] faces ({os.getpid()}): {text}", flush=True)


def status():
    """"ready", or what it is doing and for how long ("loading the finder for 12 s"), or "failed:
    why" (for /health)."""
    if _models.get("fingerprint") is not None:
        return "ready"

    return f"{_status['now']} for {time.monotonic() - _status['since']:.0f} s"


def _step(text):
    _status["now"] = text
    _status["since"] = time.monotonic()


def ready():
    """True once the models are loaded."""
    return bool(_models)


_start_lock = threading.Lock()
_warmed_in = {"pid": None}


def warm_up_once():
    """Starts warm_up() in the background, once in each process. Called on a request, never at
    import: gunicorn with --preload imports the app in its parent process and then forks the worker,
    and a worker forked while the parent's warm-up held the models' lock inherits that lock held by
    a thread it doesn't have -- every face then waits forever ("fetching the fingerprinter for 519 s")."""
    with _start_lock:
        if _warmed_in["pid"] == os.getpid():
            return False

        _warmed_in["pid"] = os.getpid()

    threading.Thread(target=warm_up, daemon=True).start()
    return True


def _after_fork():
    """In a new child process: fresh locks and no models, whatever the parent was doing."""
    global _lock, _model_lock, _start_lock
    _lock = threading.Lock()
    _model_lock = threading.Lock()
    _start_lock = threading.Lock()
    _models.clear()
    _warmed_in["pid"] = None
    _step("loading")


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def warm_up():
    """Loads the models; says how it went in the log and in status()."""
    started = time.monotonic()

    try:
        _ready()
    except Exception as error:   # noqa: BLE001 - tried again on the first face
        _step(f"failed: {type(error).__name__}: {str(error)[:120]}")
        _say(_status["now"])
        return False

    _say(f"ready in {time.monotonic() - started:.1f} s")
    return True


def _ready():
    if not _model_lock.acquire(timeout=MODEL_WAIT):
        raise TimeoutError(f"the models are still busy after {MODEL_WAIT:.0f} s ({_status['now']})")

    try:
        if _models.get("fingerprint") is None:
            _step("fetching the finder")
            finder_file = _model_file("finder")
            _step("fetching the fingerprinter")
            fingerprint_file = _model_file("fingerprint")
            _step("loading the finder")
            finder = cv2.FaceDetectorYN.create(str(finder_file), "", (320, 240), FINDER_SURE, 0.3, 5000)
            _step("loading the fingerprinter")
            _models["finder"] = finder
            _models["fingerprint"] = cv2.FaceRecognizerSF.create(str(fingerprint_file), "")

        return _models["finder"], _models["fingerprint"]
    finally:
        _model_lock.release()


def fingerprint(jpeg):
    """The fingerprint of the one clear face in [jpeg], or None (no face, or more than one: whose
    would it be?)."""

    picture = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    if picture is None:
        return None

    finder, recogniser = _ready()

    if not _model_lock.acquire(timeout=MODEL_WAIT):
        raise TimeoutError(f"the models are still busy after {MODEL_WAIT:.0f} s")

    try:
        finder.setInputSize((picture.shape[1], picture.shape[0]))
        _count, found = finder.detect(picture)

        if found is None or len(found) != 1:
            return None

        cut = recogniser.alignCrop(picture, found[0])
        feature = recogniser.feature(cut)
    finally:
        _model_lock.release()

    return _unit(feature)


def _unit(feature):
    feature = np.asarray(feature, dtype=np.float32).reshape(-1)
    size = float(np.linalg.norm(feature))
    return feature / size if size else feature


def alike(a, b):
    return float(np.dot(_unit(a), _unit(b)))


# ---- who it is ----------------------------------------------------------------------------------

def who(toy, feature):
    """(name, how alike) of the most alike known person, or (None, best) below SAME_AT_LEAST."""
    best_name, best = None, 0.0

    for name, known in store.faces(toy):
        score = alike(feature, known)

        if score > best:
            best_name, best = name, score

    return (best_name, best) if best >= SAME_AT_LEAST else (None, best)


def greeting(name, hour):
    part = "morning" if 5 <= hour < 12 else "afternoon" if 12 <= hour < 18 else "evening"
    return f"Good {part}, {name}."


def take(toy, jpeg, hour, now=None):
    """A face picture from [toy]: learned from while learning, else recognised. Returns what to say
    about it (a greeting, or how learning went), or None."""
    now = time.monotonic() if now is None else now
    feature = fingerprint(jpeg)

    with _lock:
        lesson = _learning.get(toy)

    if lesson and now > lesson["until"]:
        with _lock:
            _learning.pop(toy, None)

        lesson = None

    if lesson:
        if feature is None:
            _say(f"learning {lesson['name']}: no single clear face in that picture")
            return None

        lesson["features"].append(feature)
        _say(f"learning {lesson['name']}: {len(lesson['features'])} of {LEARN_FROM} faces")

        if len(lesson["features"]) < LEARN_FROM:
            return None

        with _lock:
            _learning.pop(toy, None)

        return _finish(toy, lesson, now)

    if feature is None:
        _say("no single clear face in that picture")
        return None

    name, score = who(toy, feature)
    _say(f"{name} ({score:.2f})" if name else f"a face I don't know (closest {score:.2f})")

    with _lock:
        _last_seen[toy] = (name, now)

        if not name:
            return None

        last = _seen_at.get((toy, name))
        _seen_at[(toy, name)] = now

        # Only as they come into view (first seen, or back after BACK_AFTER out of sight): never
        # because ten minutes have passed while they sat there (the toy sends a face each minute).
        if last is not None and now - last < BACK_AFTER:
            return None

        if now - _greeted.get((toy, name), -GREET_AGAIN - 1) < GREET_AGAIN:
            return back_words(name)

        _greeted[(toy, name)] = now

    return greeting(name, hour)


def back_words(name):
    """Said when someone known comes back before it is time to greet them again."""
    return f"I can see you, {name}."


def _finish(toy, lesson, now):
    features = lesson["features"]
    name = lesson["name"]
    kept = []

    # Each must be like the rest: a different face, or a bad find, is left out.
    for index, feature in enumerate(features):
        others = [alike(feature, other) for other_index, other in enumerate(features) if other_index != index]

        if others and sum(others) / len(others) >= LEARN_LIKE_THE_REST:
            kept.append(feature)

    _say(f"learning {name}: {len(kept)} of {len(features)} faces alike")

    if len(kept) < LEARN_FROM - 1:
        return f"I couldn't get a clear look at you, {name}. Let's try again in better light."

    if not store.add_faces(toy, name, kept):
        return f"I saw you, {name}, but couldn't save it just now. Try me again in a moment."

    # Just seen, by name: "who am I" knows them at once, and they are not greeted straight after.
    with _lock:
        _greeted[(toy, name)] = now
        _seen_at[(toy, name)] = now
        _last_seen[toy] = (name, now)

    return f"Got it. I'll know you from now on, {name}."


def learning_timed_out(toy, now=None):
    """What to say when learning ran out of time without enough faces, once; else None."""
    now = time.monotonic() if now is None else now

    with _lock:
        lesson = _learning.get(toy)

        if not lesson or now <= lesson["until"]:
            return None

        _learning.pop(toy, None)

    return f"I couldn't see your face clearly, {lesson['name']}. Face me in good light and ask again."


# ---- what is said to it -------------------------------------------------------------------------

_NAME = r"([a-z][a-z' -]{0,%d}?)" % (NAME_MOST - 1)
# Said casually, and as Whisper writes it: "learned my face", "can you learn my face please",
# "remember me, I'm Sarah", "learn my face now, my name's Sarah".
_ASKING = r"(?:(?:please|can|could|will|would|you|i|want|wanna|need|id|like|to|go|on|ok|okay|hey|right|now|so|and)\s+)*"
_LEARN_VERB = r"(?:learn|learns|learned|learnt|learning|remember|remembers|save|memori[sz]e|scan|study)"
_LEARN_WHAT = r"(?:my\s+face|me|my\s+face\s+and\s+name|who\s+i\s+am|what\s+i\s+look\s+like)"
_POLITE = r"(?:\s+(?:please|now|again|for\s+me))*"
_CALLED = r"(?:my\s+name\s+is|my\s+names|i\s+am|im|its|its\s+me|call\s+me|this\s+is|i\s+go\s+by)"
_LEARN = re.compile("^" + _ASKING + _LEARN_VERB + r"\s+" + _LEARN_WHAT + _POLITE +
                    r"(?:\s+(?:and\s+|so\s+)?(?:" + _CALLED + r"\s+)?" + _NAME + ")?" + _POLITE + "$")
_LEARN_NAMED = re.compile(r"^(?:hi\s+|hello\s+)?" + _CALLED + r"\s+" + _NAME + r"\s+(?:and\s+|so\s+)?" +
                          _ASKING + _LEARN_VERB + r"\s+" + _LEARN_WHAT + _POLITE + "$")
_FORGET_MINE = re.compile(r"^(?:please\s+)?forget\s+my\s+face(?:\s+please)?$")
_FORGET_ALL = re.compile(r"^(?:please\s+)?forget\s+(?:everyone|everybody|all\s+(?:the\s+)?faces"
                         r"|everyones\s+faces?|every\s+face)(?:\s+please)?$")
_FORGET_NAMED = re.compile(r"^(?:please\s+)?forget\s+([a-z][a-z' -]{0,%d}?)(?:s\s+face)?(?:\s+please)?$"
                           % (NAME_MOST - 1))
_WHO_AM_I = re.compile(r"^(?:who\s+am\s+i|do\s+you\s+know\s+(?:me|who\s+i\s+am)|whats\s+my\s+name"
                       r"|do\s+you\s+recogni[sz]e\s+me)$")
_WHO_KNOWN = re.compile(r"^(?:who\s+do\s+you\s+know|whose\s+faces\s+do\s+you\s+know|who\s+can\s+you\s+recogni[sz]e)$")


def _words(text):
    text = str(text or "").casefold().replace("'", "").replace("\u2019", "")
    words = " ".join(re.sub(r"[^\w\s]", " ", text).split())
    return " ".join(word for word in words.split() if word not in ("jarvis", "sir"))


def _proper(name):
    return " ".join(part.capitalize() for part in name.split())


def asked(text):
    """(what, name) when [text] is about faces: ("learn", name or None), ("forget", name or None),
    ("forget_all", None), ("who_am_i", None), ("who_known", None); else None."""
    words = _words(text)

    for shape in (_LEARN, _LEARN_NAMED):
        found = shape.match(words)

        if found:
            return "learn", _proper(found.group(1).strip()) if found.group(1) else None

    if _FORGET_ALL.match(words):
        return "forget_all", None

    if _FORGET_MINE.match(words):
        return "forget", None

    found = _FORGET_NAMED.match(words)

    if found and found.group(1).strip() not in ("it", "that", "this", "everything", "about it"):
        return "forget", _proper(found.group(1).strip())

    if _WHO_AM_I.match(words):
        return "who_am_i", None

    if _WHO_KNOWN.match(words):
        return "who_known", None

    return None


def handle(toy, what, name, now=None):
    """(what to say, how many faces to ask the toy for) for a face request."""
    now = time.monotonic() if now is None else now

    if what == "learn":
        if not name:
            return f"Tell me your name too: say, {WAKE_PHRASE}, learn my face, I'm, and your name.", 0

        with _lock:
            _learning[toy] = {"name": name, "until": now + LEARN_SECONDS, "features": []}

        _say(f"learning {name}: asking the toy for faces")

        return f"Look at me for a few seconds, {name}, in good light.", LEARN_FROM + 2

    if what == "forget_all":
        gone = store.forget_faces(toy)
        return ("I've forgotten every face I knew." if gone else "I didn't know anyone's face.") if gone is not None \
            else "I couldn't reach my memory just now. Try me again in a moment.", 0

    if what == "forget":
        known = store.names(toy)

        if not name:
            if len(known) == 1:
                name = known[0]
            elif not known:
                return "I don't know anyone's face yet.", 0
            else:
                return f"Whose face? Say, {WAKE_PHRASE}, forget, and the name.", 0

        match = next((person for person in known if person.casefold() == name.casefold()), None)

        if not match:
            return f"I don't know {name}'s face.", 0

        gone = store.forget_faces(toy, match)
        return (f"I've forgotten {match}'s face." if gone is not None
                else "I couldn't reach my memory just now. Try me again in a moment."), 0

    if what == "who_am_i":
        with _lock:
            last = _last_seen.get(toy)

        if not last or now - last[1] > 120:
            return "I haven't had a clear look at you yet. Stand in front of me for a moment and ask again.", 0

        if last[0]:
            return f"You're {last[0]}.", 0

        return f"I don't know you yet. Say, {WAKE_PHRASE}, learn my face, I'm, and your name.", 0

    if what == "who_known":
        known = store.names(toy)

        if not known:
            return "I don't know anyone's face yet.", 0

        listed = known[0] if len(known) == 1 else ", ".join(known[:-1]) + " and " + known[-1]
        return f"I know {listed}.", 0

    return None, 0


def last_seen(toy):
    """(name or None, when) of the last face recognised or not, as kept: a new one each face."""
    with _lock:
        return _last_seen.get(toy)


def last_known(toy, now=None, within=120.0):
    """The name of the person the toy last recognised, if that was within [within] seconds."""
    now = time.monotonic() if now is None else now

    with _lock:
        last = _last_seen.get(toy)

    return last[0] if last and last[0] and now - last[1] <= within else None


def forget_state():
    """For tests."""
    with _lock:
        _learning.clear()
        _last_seen.clear()
        _greeted.clear()
        _seen_at.clear()


if __name__ == "__main__":
    # The build step: python -m toy.faces fetches both models into MODEL_FOLDER.
    for which in MODELS:
        print(f"{MODELS[which][0]}: {_model_file(which)}")
