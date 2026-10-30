"""Talk to the toy cloud from the PC, as the toy would, without reflashing anything.

Records a few seconds from the PC microphone, sends it to /toy/talk as if BOOT were held, and
plays the answer. Run it with JARVIS's .venv Python (it has sounddevice and numpy):

    $env:TOY_TOKEN = 'the token you put in Render'
    python tools/try_talk.py
    python tools/try_talk.py --seconds 6 --hands-free

The token is read from TOY_TOKEN and never printed.
"""

import argparse
import io
import os
import sys
import time
import wave

import requests

RATE = 16000
DEFAULT_URL = "https://jarvis-toy-cloud.onrender.com"


def record(seconds):
    import sounddevice as sd

    print(f"Speak now ({seconds} s)...", flush=True)
    audio = sd.rec(int(seconds * RATE), samplerate=RATE, channels=1, dtype="int16")
    sd.wait()
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(RATE)
        out.writeframes(audio.tobytes())

    return buffer.getvalue()


def play(pcm):
    import numpy as np
    import sounddevice as sd

    sd.play(np.frombuffer(pcm, dtype=np.int16), samplerate=RATE)
    sd.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=os.getenv("TOY_CLOUD_URL") or DEFAULT_URL)
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--hands-free", action="store_true",
                        help="send as heard hands-free, so it must start with Jarvis")
    args = parser.parse_args()

    token = (os.getenv("TOY_TOKEN") or "").strip()

    if not token:
        sys.exit("Set TOY_TOKEN first:  $env:TOY_TOKEN = 'your token'")

    health = requests.get(f"{args.url}/health", timeout=90)
    print(f"health: {health.status_code} {health.text.strip()}")

    wav = record(args.seconds)
    started = time.monotonic()
    response = requests.post(f"{args.url}/toy/talk", data=wav, timeout=60, headers={
        "X-Jarvis-Token": token,
        "Content-Type": "audio/wav",
        "X-Jarvis-Camera": "pc test",
        "X-Jarvis-Toy": "sound" if args.hands_free else "button",
    })
    took = time.monotonic() - started

    print(f"answer: {response.status_code}, mood {response.headers.get('X-Jarvis-Mood', '-')}, "
          f"{len(response.content)} bytes, {took:.1f} s")

    if response.status_code == 200 and response.headers.get("Content-Type", "").startswith("audio/L16"):
        play(response.content)
    elif response.status_code == 204:
        print("(silent: not said to Jarvis)")
    else:
        print(response.text[:300])


if __name__ == "__main__":
    main()
