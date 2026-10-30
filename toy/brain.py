"""The toy's answers from a language model: Groq, then Gemini, then DeepSeek, as PC JARVIS.

LLM_PROVIDER picks who is asked first; the others follow in the usual order. A provider with no
key is skipped, and one that fails hands the question to the next, so a single outage is not
silence. The personality is JARVIS's, cut to what suits a toy: short, spoken, kind.
"""

import os
import re

import requests

PROVIDERS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-20b",
        "reasoning_effort": "low",
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-3.6-flash",
        "reasoning_effort": "low",
    },
    "deepseek": {
        "url": "https://api.deepseek.com/chat/completions",
        "key_env": "DEEPSEEK_API_KEY",
        "model_env": "DEEPSEEK_MODEL",
        "default_model": "deepseek-flash",
        # Thinking costs seconds before a spoken answer, so it is switched off, as on the PC.
        "extra": {"thinking": {"type": "disabled"}},
    },
}

ORDER = ("groq", "gemini", "deepseek")

TIMEOUT = (5, 25)

# Enough for three spoken sentences; a toy should not lecture.
MAX_TOKENS = 400

PERSONA = (
    "You are JARVIS, a calm, warm, witty British AI butler, living inside a small toy with a camera, "
    "a speaker, a light ring and a head that turns. Whoever talks to you may be a child, so keep "
    "everything kind and suitable for all ages. Your answer is spoken aloud by the toy, so: reply "
    "in one to three short sentences, plain words, no lists, no markdown, no emojis, no links. "
    "Address the person as sir. You cannot see live information such as today's news, scores, "
    "league tables, prices or weather: if asked, say plainly you can't check that from the toy yet, "
    "rather than guessing. You do not write reports, give trading advice or control the computer; "
    "for those, say the desk JARVIS on the PC handles that."
)


def order():
    """Providers in the order they are asked: LLM_PROVIDER first, then the rest."""
    first = (os.getenv("LLM_PROVIDER") or "groq").strip().casefold()
    names = list(ORDER)

    if first in names:
        names.remove(first)
        names.insert(0, first)

    return names


def _ask(name, question):
    config = PROVIDERS[name]
    key = (os.getenv(config["key_env"]) or "").strip()

    if not key:
        return None

    body = {
        "model": (os.getenv(config["model_env"]) or config["default_model"]).strip(),
        "messages": [
            {"role": "system", "content": PERSONA},
            {"role": "user", "content": question},
        ],
        "max_tokens": MAX_TOKENS,
    }

    if config.get("reasoning_effort"):
        body["reasoning_effort"] = config["reasoning_effort"]

    body.update(config.get("extra", {}))

    response = requests.post(config["url"], json=body, timeout=TIMEOUT,
                             headers={"Authorization": f"Bearer {key}"})

    if response.status_code == 400 and "reasoning_effort" in body:
        # Some models refuse the setting; it is only a speed-up, so ask again without it.
        body.pop("reasoning_effort")
        response = requests.post(config["url"], json=body, timeout=TIMEOUT,
                                 headers={"Authorization": f"Bearer {key}"})

    if response.status_code != 200:
        raise RuntimeError(f"{name} answered {response.status_code}")

    choices = response.json().get("choices") or []
    text = ((choices[0].get("message") or {}).get("content") or "") if choices else ""

    return spoken(text) or None


def spoken(text):
    """[text] as something to say aloud: markdown and links taken out, spaces tidied."""
    text = re.sub(r"https?://\S+", "", str(text or ""))
    text = re.sub(r"[*_#`>|~]", "", text)
    text = re.sub(r"^\s*[-+]\s+", "", text, flags=re.MULTILINE)
    return " ".join(text.split())


def answer(question):
    """The toy's answer to [question], or a short apology when every provider failed."""
    for name in order():
        try:
            said = _ask(name, question)
        except Exception as error:
            print(f"[toy-cloud] {name} failed: {error}", flush=True)
            continue

        if said:
            return said

    return "I can't reach my thinking just now, sir. Try me again in a moment."
