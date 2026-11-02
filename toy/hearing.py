"""What was said in a WAV the toy recorded: Whisper on Groq, as PC JARVIS does with STT_ENGINE=groq."""

import os

import requests

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
DEFAULT_MODEL = "whisper-large-v3-turbo"

# Naming the expected words biases Whisper towards them, so "Jarvis" is not heard as "java's", nor
# "Jamendo" (said "JAM-endo") as "jam endo" (2 November 2026).
PROMPT = ("Jarvis. Things said to Jarvis, a toy: mute, unmute, what time is it, how are you, I love you, "
          "play some hip hop from Jamendo, what's playing, pause, resume.")

TIMEOUT = (5, 20)


def transcribe(wav):
    """The words in [wav], or "" when nothing could be made out. Raises when Groq cannot be reached."""
    key = (os.getenv("GROQ_API_KEY") or "").strip()

    if not key:
        raise RuntimeError("GROQ_API_KEY is not set")

    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {key}"},
        files={"file": ("speech.wav", wav, "audio/wav")},
        data={
            "model": (os.getenv("GROQ_WHISPER_MODEL") or DEFAULT_MODEL).strip(),
            "language": "en",
            "temperature": "0",
            "prompt": PROMPT,
            "response_format": "json",
        },
        timeout=TIMEOUT,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Groq transcription answered {response.status_code}")

    return str(response.json().get("text") or "").strip()
