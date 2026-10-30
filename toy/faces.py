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

Models: YuNet (MIT) and SFace (Apache 2.0) from OpenCV's model zoo, fetched on first use into
MODEL_FOLDER and checked against their SHA-256, as PC JARVIS's tools/face_model.py fetches them.
"""

import hashlib
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
import requests

from toy import store

ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
MODELS = {
    "finder": ("face_detection_yunet_2023mar.onnx", ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
               "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "fingerprint": ("face_recognition_sface_2021dec.onnx",
                    ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
                    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}
MODEL_FOLDER = Path(os.getenv("TOY_MODEL_FOLDER") or "/tmp/toy-models")

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
NAME_MOST = 30

_lock = threading.Lock()          # who is learning, seen, greeted: held only for a moment
_model_lock = threading.Lock()    # the models: fetched, loaded and run one at a time, never under _lock
_models = {}
_learning = {}    # toy -> {"name", "until", "features": []}
_last_seen = {}   # toy -> (name or None, monotonic time)
_greeted = {}     # (toy, name) -> monotonic time


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


def ready():
    """True once the models are loaded (for /health)."""
    return bool(_models)


def _ready():
    with _model_lock:
        if not _models:
            import cv2

            _models["finder"] = cv2.FaceDetectorYN.create(str(_model_file("finder")), "", (320, 240),
                                                          FINDER_SURE, 0.3, 5000)
            _models["fingerprint"] = cv2.FaceRecognizerSF.create(str(_model_file("fingerprint")), "")

        return _models["finder"], _models["fingerprint"]


def fingerprint(jpeg):
    """The fingerprint of the one clear face in [jpeg], or None (no face, or more than one: whose
    would it be?)."""
    import cv2

    picture = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)

    if picture is None:
        return None

    finder, recogniser = _ready()

    with _model_lock:
        finder.setInputSize((picture.shape[1], picture.shape[0]))
        _count, found = finder.detect(picture)

        if found is None or len(found) != 1:
            return None

        cut = recogniser.alignCrop(picture, found[0])
        feature = recogniser.feature(cut)

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
            return None

        lesson["features"].append(feature)

        if len(lesson["features"]) < LEARN_FROM:
            return None

        with _lock:
            _learning.pop(toy, None)

        return _finish(toy, lesson)

    if feature is None:
        return None

    name, _score = who(toy, feature)

    with _lock:
        _last_seen[toy] = (name, now)

        if not name or now - _greeted.get((toy, name), -GREET_AGAIN - 1) < GREET_AGAIN:
            return None

        _greeted[(toy, name)] = now

    return greeting(name, hour)


def _finish(toy, lesson):
    features = lesson["features"]
    name = lesson["name"]
    kept = []

    # Each must be like the rest: a different face, or a bad find, is left out.
    for index, feature in enumerate(features):
        others = [alike(feature, other) for other_index, other in enumerate(features) if other_index != index]

        if others and sum(others) / len(others) >= LEARN_LIKE_THE_REST:
            kept.append(feature)

    if len(kept) < LEARN_FROM - 1:
        return f"I couldn't get a clear look at you, {name}. Let's try again in better light."

    if not store.add_faces(toy, name, kept):
        return f"I saw you, {name}, but couldn't save it just now. Try me again in a moment."

    with _lock:
        _greeted[(toy, name)] = time.monotonic()

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

_NAME = r"([a-z][a-z' -]{0,%d})" % (NAME_MOST - 1)
_LEARN = re.compile(r"^(?:please\s+)?(?:learn|remember|save)\s+my\s+face(?:\s+please)?"
                    r"(?:\s+(?:my\s+name\s+is|my\s+names|i\s+am|im|its|call\s+me|this\s+is)\s+" + _NAME + r")?$")
_LEARN_NAMED = re.compile(r"^(?:my\s+name\s+is|my\s+names|i\s+am|im|this\s+is)\s+" + _NAME +
                          r"\s+(?:please\s+)?(?:learn|remember|save)\s+my\s+face(?:\s+please)?$")
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
            return "Tell me your name too: say, Jarvis, learn my face, I'm, and your name.", 0

        with _lock:
            _learning[toy] = {"name": name, "until": now + LEARN_SECONDS, "features": []}

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
                return "Whose face? Say, Jarvis, forget, and the name.", 0

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

        return "I don't know you yet. Say, Jarvis, learn my face, I'm, and your name.", 0

    if what == "who_known":
        known = store.names(toy)

        if not known:
            return "I don't know anyone's face yet.", 0

        listed = known[0] if len(known) == 1 else ", ".join(known[:-1]) + " and " + known[-1]
        return f"I know {listed}.", 0

    return None, 0


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
