"""JARVIS toy cloud: the toy's brain on the internet, so it needs no PC switched on.

POST /toy/talk is the same contract as PC JARVIS's phone.py, so the toy firmware only changes
where it sends and which token it uses:

  in:  a WAV (16 kHz mono 16-bit) the toy recorded; headers X-Jarvis-Token, X-Jarvis-Camera
       (which toy), X-Jarvis-Toy ("button" when BOOT was held, "sound" when heard hands-free)
  out: 200 with the reply as raw PCM (audio/L16, 16 kHz) and X-Jarvis-Mood "affection"/"normal",
       or 204 when the toy should say nothing and keep listening.

  The answer may carry X-Jarvis-Switch "pc" or "cloud" when it was asked to change mode ("Jarvis,
  switch to PC mode"): the toy keeps the choice. X-Jarvis-Fallback "pc" on a request means the toy
  is in PC mode but the PC is not answering; the first answer in a while says so.

POST /camera/seen takes what the toy's own detection is sure of, as PC JARVIS does, and answers
{"ok": true, "say": true} when something has come into view; the toy then fetches the words from
POST /toy/announce (raw PCM, or 204) and says them. POST /camera/face takes the picture of a face
the toy found; recognising it is the next chunk, and nothing is kept.

GET /health answers {"ok": true} with nothing secret in it, for Render and UptimeRobot.
"""

import os
import secrets
import threading
import time

from flask import Flask, jsonify, request

from toy import brain, faces, hearing, modes, seen, social, store, voice, wake, world

# What the toy sends at most: it stops recording at ten seconds, 32000 bytes a second, plus room.
MAX_TALK_BYTES = 44 + 32000 * 12

# Shorter than a third of a second is a click, not speech.
MIN_VOICE_BYTES = 44 + 32000 // 3

# After an answer heard hands-free, a follow-up needs no name for this long.
FOLLOW_UP_SECONDS = 12.0

# A report of what the toy sees is small (names and boxes); a face picture is a VGA JPEG.
MAX_SEEN_BYTES = 32 * 1024
MAX_FACE_BYTES = 200 * 1024

# In PC mode with the PC off, the toy falls back to the cloud: the first answer says so, and again
# only after this long, so every answer is not prefixed with it.
FALLBACK_REMIND_SECONDS = 600.0
FALLBACK_WORDS = "Defaulting to the cloud, sir: your PC isn't answering. Turn it on and try again for PC mode."

SWITCH_WORDS = {
    "pc": "Switching to PC mode, sir. If your PC is off, I'll fall back to the cloud.",
    "cloud": "Switching to cloud mode, sir. Your PC can rest.",
}

MUTE_WORDS = frozenset({"mute", "mute yourself", "be quiet", "go quiet", "stop listening"})
UNMUTE_WORDS = frozenset({"unmute", "unmute yourself", "start listening", "listen again",
                          "you can listen again"})

app = Flask(__name__)

# Per toy (X-Jarvis-Camera): muted, and until when a follow-up needs no name. Kept in memory: a
# restart unmutes the toy and ends a follow-up window, which is the safe way round.
_toys = {}
_toys_lock = threading.Lock()


def _toy(name):
    with _toys_lock:
        name = name or "toy"
        return _toys.setdefault(name, {"name": name, "muted": False, "follow_until": 0.0, "fallback_said": None,
                                       "faces_wanted": 0})


def _authorised():
    expected = (os.getenv("TOY_TOKEN") or "").strip()
    supplied = (request.headers.get("X-Jarvis-Token") or "").strip()

    # No token configured means no toy may talk, never that every caller may.
    return bool(expected) and secrets.compare_digest(supplied, expected)


def mute_asked(text):
    """"mute" or "unmute" when [text] asks for it ("mute", "un-mute.", "unmute please"), else None."""
    words = str(text or "").casefold().replace("-", "")
    words = " ".join(word.strip(".,!?'\"") for word in words.split())
    words = " ".join(word for word in words.split() if word not in ("please", "sir", "now", "jarvis"))

    if words in UNMUTE_WORDS or words.replace(" ", "") == "unmute":
        return "unmute"

    if words in MUTE_WORDS:
        return "mute"

    return None


def _toy_address():
    """The toy's public address as Render saw it (its first X-Forwarded-For), for where it is."""
    forwarded = (request.headers.get("X-Forwarded-For") or "").split(",")[0].strip()
    return forwarded or request.remote_addr


def answer_words(text, ip=None, toy=None):
    """(what to say, mood) for a command: thanks and affection, faces, where it lives, the time, the
    date and the weather free; everything else the model's, searched first when it needs live
    information. A face lesson asks the toy for faces (toy["faces_wanted"], sent as X-Jarvis-Faces)."""
    toy = toy if toy is not None else _toy("toy")
    name = toy["name"]
    which = social.kind(text)

    if which:
        return social.reply(which), "affection" if which in social.FOND_KINDS else ""

    face = faces.asked(text)

    if face:
        said, wanted = faces.handle(name, *face)
        toy["faces_wanted"] = wanted
        return said, ""

    if seen.sight_asked(text):
        return seen.describe(name, faces.last_known(name)), ""

    town = world.lives_in_asked(text)

    if town:
        return world.learn_place(name, town), ""

    asked = world.time_asked(text)
    where = None

    if asked:
        where = world.place(ip, name)
        return world.time_words(asked, world.now(where)), ""

    when = world.weather_asked(text)

    if when:
        where = world.place(ip, name)

        if where.get("lat") is None:
            return world.UNKNOWN_PLACE, ""

        said = world.weather_words(when, where)

        if said:
            return said, ""

        # The forecast service failed: the weather is still a live question, so it is looked up.
        moment = world.now(where)
        today = f"{moment.strftime('%A')} {moment.day} {moment.strftime('%B %Y')}, {moment.strftime('%H:%M')}"
        city = where.get("city") or ""
        return brain.answer(f"What is the weather in {city} {when}? ({text})", today=today, city=city), ""

    where = where or world.place(ip, name)
    moment = world.now(where)
    today = f"{moment.strftime('%A')} {moment.day} {moment.strftime('%B %Y')}, {moment.strftime('%H:%M')}"
    return brain.answer(text, today=today, city=where.get("city") or ""), ""


def respond(wav, heard_by, toy, ip=None):
    """(what to say, mood, switch) for a recording, or (None, "", "") to stay silent. Mirrors phone.py's
    _toy_answer: hands-free speech is his only when it starts with his name or comes within a
    follow-up window; anything else is not his, and gets silence rather than a complaint."""
    hands_free = heard_by == "sound"
    following = hands_free and time.monotonic() < toy["follow_until"]

    if len(wav) < MIN_VOICE_BYTES:
        return (None, "", "") if hands_free else ("That was too short, sir. Hold the button while you speak.", "", "")

    text = hearing.transcribe(wav)

    if not text:
        return (None, "", "") if hands_free else ("I couldn't make out what you said, sir.", "", "")

    if wake.only_filler(text):
        print(f'[filtered] "{text}" (single filler word)', flush=True)
        return (None, "", "") if hands_free else ("I only caught a stray word, sir.", "", "")

    if hands_free and not following:
        command = wake.addressed_command(text)

        if command is None:
            print(f'[ignored] "{text}" (not said to JARVIS)', flush=True)
            return None, "", ""

        print(f'[toy] woken by "{text}"', flush=True)

        if not command:
            if toy["muted"]:
                return None, "", ""

            toy["follow_until"] = time.monotonic() + FOLLOW_UP_SECONDS
            return "Yes, sir?", "", ""

        text = command
    elif following:
        # A follow-up may still start with his name: it is not part of the question.
        command = wake.addressed_command(text)
        text = command if command else text

    asked = mute_asked(text)

    if asked == "unmute":
        toy["muted"] = False
        toy["follow_until"] = 0.0
        print("[toy] unmuted", flush=True)
        return "I'm listening again, sir.", "", ""

    if toy["muted"]:
        print(f'[ignored] "{text}" (muted)', flush=True)
        return None, "", ""

    if asked == "mute":
        toy["muted"] = True
        toy["follow_until"] = 0.0
        print("[toy] muted", flush=True)
        return "Muted, sir. Say Jarvis, unmute, when you want me.", "", ""

    switch = modes.switch_asked(text)

    if switch:
        print(f"[toy] switching to {switch} mode", flush=True)
        toy["follow_until"] = 0.0
        return SWITCH_WORDS[switch], "", switch

    print(f'[toy] heard "{text}"', flush=True)

    said, mood = answer_words(text, ip, toy)

    toy["follow_until"] = time.monotonic() + FOLLOW_UP_SECONDS if hands_free else 0.0
    return said, mood, ""


def _fallback_note(toy, now=None):
    """FALLBACK_WORDS when the toy fell back from an unanswering PC and has not been told lately."""
    now = time.monotonic() if now is None else now
    said = toy["fallback_said"]

    if said is not None and now - said < FALLBACK_REMIND_SECONDS:
        return ""

    toy["fallback_said"] = now
    return FALLBACK_WORDS


def _pcm_response(said, mood="", switch="", faces_wanted=0):
    data = voice.pcm16(said)

    if not data:
        return jsonify({"error": "No audio."}), 503

    headers = {
        "Content-Type": f"audio/L16; rate={voice.RATE}; channels=1",
        "Cache-Control": "no-store",
        "X-Jarvis-Mood": mood or "normal",
    }

    if switch:
        headers["X-Jarvis-Switch"] = switch

    if faces_wanted:
        headers["X-Jarvis-Faces"] = str(faces_wanted)

    return data, 200, headers


@app.get("/health")
def health():
    """Open, and nothing secret in it. With ?detail=1, where memory is kept and whether the face
    models are loaded: open it in a browser to check the toy's cloud without a laptop."""
    if request.args.get("detail"):
        return jsonify({"ok": True, "memory": store.kind(), "faces": "ready" if faces.ready() else "loading"})

    return jsonify({"ok": True})


@app.post("/toy/talk")
def toy_talk():
    if not _authorised():
        return jsonify({"error": "unauthorised"}), 403

    if request.content_length is None or request.content_length > MAX_TALK_BYTES:
        return jsonify({"error": "too large"}), 413

    content_type = request.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold()

    if content_type != "audio/wav":
        return jsonify({"error": "toy talk must be WAV audio"}), 400

    heard_by = (request.headers.get("X-Jarvis-Toy") or "button").strip().casefold()
    toy = _toy((request.headers.get("X-Jarvis-Camera") or "").strip())

    fell_back = (request.headers.get("X-Jarvis-Fallback") or "").strip().casefold() == "pc"

    try:
        said, mood, switch = respond(request.get_data(cache=False), heard_by, toy, _toy_address())
    except Exception as error:
        print(f"[toy-cloud] talk failed: {error}", flush=True)
        said, mood, switch = "Something went wrong there, sir.", "", ""

    if said is None:
        return "", 204

    # Asked for PC mode while already falling back from it: the switch itself says what happens.
    if fell_back and not switch:
        note = _fallback_note(toy)
        said = f"{note} {said}" if note else said

    wanted, toy["faces_wanted"] = toy["faces_wanted"], 0
    return _pcm_response(said, mood, switch, wanted)


@app.post("/camera/seen")
def camera_seen():
    if not _authorised():
        return jsonify({"error": "unauthorised"}), 403

    if request.content_length is None or request.content_length > MAX_SEEN_BYTES:
        return jsonify({"error": "too large"}), 413

    name = (request.headers.get("X-Jarvis-Camera") or "").strip() or "toy"
    due = seen.record(name, request.get_json(silent=True))

    if due is None:
        return jsonify({"error": "not a report"}), 400

    # A face lesson that ran out of time says so, once.
    late = faces.learning_timed_out(name)

    if late:
        seen.queue(name, late)
        due = True

    # A muted toy says nothing unasked either.
    if _toy(name)["muted"]:
        seen.take(name)
        due = False

    return jsonify({"ok": True, "say": bool(due)}), 200


@app.post("/toy/announce")
def toy_announce():
    if not _authorised():
        return jsonify({"error": "unauthorised"}), 403

    name = (request.headers.get("X-Jarvis-Camera") or "").strip() or "toy"
    said = seen.take(name)

    if not said or _toy(name)["muted"]:
        return "", 204

    return _pcm_response(said)


@app.post("/camera/face")
def camera_face():
    """A face the toy's own face finder saw: learned from while a lesson is on ("Jarvis, learn my face,
    I'm Sarah"), else recognised, and a known face greeted by name. The picture itself is never kept,
    and nothing about a stranger's face. Answers "say": true when there are words for the toy."""
    if not _authorised():
        return jsonify({"error": "unauthorised"}), 403

    if request.content_length is None or request.content_length > MAX_FACE_BYTES:
        return jsonify({"error": "too large"}), 413

    if request.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold() != "image/jpeg":
        return jsonify({"error": "a face must be a JPEG"}), 400

    name = (request.headers.get("X-Jarvis-Camera") or "").strip() or "toy"
    picture = request.get_data(cache=False)
    hour = world.now(world.place(_toy_address(), name)).hour

    try:
        said = faces.take(name, picture, hour)
    except Exception as error:   # noqa: BLE001 - a face that cannot be read must not stop the toy
        print(f"[toy-cloud] faces: {type(error).__name__}: {error}", flush=True)
        said = None

    if said and not _toy(name)["muted"]:
        seen.queue(name, said)
        return jsonify({"ok": True, "say": True}), 200

    return jsonify({"ok": True, "say": False}), 200


def _warm_up():
    """The face models fetched and loaded before the first face arrives, not while it waits."""
    try:
        faces._ready()
    except Exception as error:   # noqa: BLE001 - tried again on the first face
        print(f"[toy-cloud] faces: not ready yet ({type(error).__name__})", flush=True)


if os.getenv("TOY_WARM_UP", "1") != "0":
    threading.Thread(target=_warm_up, daemon=True).start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 8000))
