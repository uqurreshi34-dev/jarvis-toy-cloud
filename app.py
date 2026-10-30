"""JARVIS toy cloud: the toy's brain on the internet, so it needs no PC switched on.

POST /toy/talk is the same contract as PC JARVIS's phone.py, so the toy firmware only changes
where it sends and which token it uses:

  in:  a WAV (16 kHz mono 16-bit) the toy recorded; headers X-Jarvis-Token, X-Jarvis-Camera
       (which toy), X-Jarvis-Toy ("button" when BOOT was held, "sound" when heard hands-free)
  out: 200 with the reply as raw PCM (audio/L16, 16 kHz) and X-Jarvis-Mood "affection"/"normal",
       or 204 when the toy should say nothing and keep listening.

GET /health answers {"ok": true} with nothing secret in it, for Render and UptimeRobot.
"""

import os
import secrets
import threading
import time

from flask import Flask, jsonify, request

from toy import brain, hearing, social, voice, wake

# What the toy sends at most: it stops recording at ten seconds, 32000 bytes a second, plus room.
MAX_TALK_BYTES = 44 + 32000 * 12

# Shorter than a third of a second is a click, not speech.
MIN_VOICE_BYTES = 44 + 32000 // 3

# After an answer heard hands-free, a follow-up needs no name for this long.
FOLLOW_UP_SECONDS = 12.0

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
        return _toys.setdefault(name or "toy", {"muted": False, "follow_until": 0.0})


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


def respond(wav, heard_by, toy):
    """(what to say, mood) for a recording, or (None, "") to stay silent. Mirrors phone.py's
    _toy_answer: hands-free speech is his only when it starts with his name or comes within a
    follow-up window; anything else is not his, and gets silence rather than a complaint."""
    hands_free = heard_by == "sound"
    following = hands_free and time.monotonic() < toy["follow_until"]

    if len(wav) < MIN_VOICE_BYTES:
        return (None, "") if hands_free else ("That was too short, sir. Hold the button while you speak.", "")

    text = hearing.transcribe(wav)

    if not text:
        return (None, "") if hands_free else ("I couldn't make out what you said, sir.", "")

    if wake.only_filler(text):
        print(f'[filtered] "{text}" (single filler word)', flush=True)
        return (None, "") if hands_free else ("I only caught a stray word, sir.", "")

    if hands_free and not following:
        command = wake.addressed_command(text)

        if command is None:
            print(f'[ignored] "{text}" (not said to JARVIS)', flush=True)
            return None, ""

        print(f'[toy] woken by "{text}"', flush=True)

        if not command:
            if toy["muted"]:
                return None, ""

            toy["follow_until"] = time.monotonic() + FOLLOW_UP_SECONDS
            return "Yes, sir?", ""

        text = command

    asked = mute_asked(text)

    if asked == "unmute":
        toy["muted"] = False
        toy["follow_until"] = 0.0
        print("[toy] unmuted", flush=True)
        return "I'm listening again, sir.", ""

    if toy["muted"]:
        print(f'[ignored] "{text}" (muted)', flush=True)
        return None, ""

    if asked == "mute":
        toy["muted"] = True
        toy["follow_until"] = 0.0
        print("[toy] muted", flush=True)
        return "Muted, sir. Say Jarvis, unmute, when you want me.", ""

    print(f'[toy] heard "{text}"', flush=True)

    which = social.kind(text)
    mood = "affection" if which in social.FOND_KINDS else ""
    said = social.reply(which) if which else brain.answer(text)

    toy["follow_until"] = time.monotonic() + FOLLOW_UP_SECONDS if hands_free else 0.0
    return said, mood


@app.get("/health")
def health():
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

    try:
        said, mood = respond(request.get_data(cache=False), heard_by, toy)
    except Exception as error:
        print(f"[toy-cloud] talk failed: {error}", flush=True)
        said, mood = "Something went wrong there, sir.", ""

    if said is None:
        return "", 204

    data = voice.pcm16(said)

    if not data:
        return jsonify({"error": "No audio."}), 503

    return data, 200, {
        "Content-Type": f"audio/L16; rate={voice.RATE}; channels=1",
        "Cache-Control": "no-store",
        "X-Jarvis-Mood": mood or "normal",
    }


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT") or 8000))
