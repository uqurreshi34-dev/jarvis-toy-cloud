"""The toy's voice: Google's Chirp 3 HD, as PC JARVIS, sent as raw 16 kHz 16-bit mono PCM.

Google is asked for LINEAR16 at the toy's own rate, so nothing is decoded or resampled here; its
WAV header is cut off and the loudest moment raised to TOY_TALK_PEAK of full scale, because the
toy's speaker is small and quiet (as on the PC).
"""

import array
import base64
import os
import re
import sys
import threading
import time

import requests

GOOGLE_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"
DEFAULT_VOICE = "en-GB-Chirp3-HD-Achird"
RATE = 16000
PEAK = 0.95
TIMEOUT = (5, 20)

# Chirp 3 HD is free for 1,000,000 characters a month. This stops short of it so nothing is charged.
# The count lives in memory only until chunk 2 gives the toy a database: a restart starts it again,
# so it is a guard against a runaway loop, not an exact meter.
DEFAULT_MONTHLY_LIMIT = 950_000

_VOICE_SHAPE = re.compile(r"^[a-z]{2,3}-[A-Z]{2}(-[A-Za-z0-9]+){2,}$")
_used = {"month": "", "characters": 0}
_lock = threading.Lock()


def voice_name():
    chosen = (os.getenv("GOOGLE_TTS_VOICE") or "").strip()
    return chosen if _VOICE_SHAPE.match(chosen) else DEFAULT_VOICE


def speaking_rate():
    try:
        value = float(os.getenv("GOOGLE_TTS_RATE") or 0)
    except ValueError:
        return None

    return value if 0.25 <= value <= 2.0 else None


def _monthly_limit():
    try:
        return int(os.getenv("GOOGLE_TTS_MONTHLY_CHARACTERS") or DEFAULT_MONTHLY_LIMIT)
    except ValueError:
        return DEFAULT_MONTHLY_LIMIT


def _spend(characters):
    month = time.strftime("%Y-%m")

    with _lock:
        if _used["month"] != month:
            _used.update(month=month, characters=0)

        if _used["characters"] + characters > _monthly_limit():
            raise RuntimeError("this month's Google characters are used up")

        _used["characters"] += characters


def loudest_to(pcm, peak=PEAK):
    """16-bit little-endian [pcm] scaled so its loudest sample is [peak] of full scale."""
    samples = array.array("h")
    samples.frombytes(pcm[: len(pcm) - len(pcm) % 2])

    if sys.byteorder == "big":
        samples.byteswap()

    loudest = max((abs(sample) for sample in samples), default=0)

    if not loudest:
        return samples.tobytes()

    gain = peak * 32767 / loudest
    scaled = array.array("h", (max(-32768, min(32767, int(sample * gain))) for sample in samples))

    if sys.byteorder == "big":
        scaled.byteswap()

    return scaled.tobytes()


def _without_header(wav):
    """The samples of a RIFF/WAV, found by its "data" chunk rather than assuming 44 bytes."""
    if wav[:4] != b"RIFF":
        return wav

    found = wav.find(b"data", 12)
    return wav[found + 8:] if found != -1 else wav[44:]


def pcm16(text):
    """[text] spoken as raw 16-bit mono PCM at RATE, or None when it could not be made."""
    text = " ".join(str(text or "").split())

    if not text:
        return None

    key = (os.getenv("GOOGLE_TTS_API_KEY") or "").strip()

    if not key:
        print("[toy-cloud] GOOGLE_TTS_API_KEY is not set", flush=True)
        return None

    try:
        _spend(len(text))
    except RuntimeError as error:
        print(f"[toy-cloud] {error}", flush=True)
        return None

    voice = voice_name()
    config = {"audioEncoding": "LINEAR16", "sampleRateHertz": RATE}
    rate = speaking_rate()

    if rate:
        config["speakingRate"] = rate

    try:
        response = requests.post(
            GOOGLE_URL,
            json={"input": {"text": text},
                  "voice": {"languageCode": "-".join(voice.split("-")[:2]), "name": voice},
                  "audioConfig": config},
            headers={"X-Goog-Api-Key": key, "User-Agent": "JARVIS-toy-cloud"},
            timeout=TIMEOUT,
        )
    except requests.RequestException as error:
        print(f"[toy-cloud] Google could not be reached: {error}", flush=True)
        return None

    if response.status_code != 200:
        print(f"[toy-cloud] Google voice answered {response.status_code}", flush=True)
        return None

    audio = base64.b64decode(response.json().get("audioContent") or "")
    pcm = _without_header(audio)

    return loudest_to(pcm) if pcm else None
