"""The toy's answers from a language model: Groq, then Gemini, then DeepSeek, as PC JARVIS.

LLM_PROVIDER picks who is asked first; the others follow in the usual order. A provider with no
key is skipped, and one that fails hands the question to the next, so a single outage is not
silence. The personality is JARVIS's, cut to what suits a toy: short, spoken, kind.

Live questions (today's news, any league's table, prices, who holds a job now): the model itself
decides first, with no list of words or leagues kept here, whether a right answer needs information
that changes or is newer than its training (LIVE) or not (KNOWN). LIVE is searched and answered from
what was found; KNOWN is answered, and the model may still answer just SEARCH. It is always told
today's date and the town the toy is in, so "tomorrow", "this year" and "here" mean something.
"""

import os
import re

import requests

from toy import search

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
    "Address the person as sir. Never say what you can see or describe your surroundings: your eyes "
    "are answered separately, so if asked, say you'll have a look. You do not write reports, give trading advice or control the computer; "
    "for those, say the desk JARVIS on the PC handles that."
)

# First, the model's own judgement of whether the question needs a search.
DECIDE = (
    "You decide one thing about the user's question: does a correct answer depend on information "
    "that changes over time or may be newer than your training -- sport tables, positions, results or "
    "fixtures in any league or competition, news, events, prices, weather, who holds a position now? "
    "Reply with exactly the single word LIVE if so, or KNOWN if a correct answer never changes "
    "(facts, definitions, history, maths, jokes, advice). Nothing else."
)
LIVE = "LIVE"

# Without results: the model may ask for a search rather than guess.
MAY_SEARCH = (
    " If answering needs live or recent information you were not given -- news, sport results, league "
    "tables, prices, who holds a position now, anything after your training -- reply with exactly the "
    "single word SEARCH and nothing else. Otherwise answer."
)

# With results: answer from them.
FROM_RESULTS = (
    " Below are web search results gathered just now. Answer from them, briefly and plainly, giving the "
    "most recent figure they show. If they truly do not answer the question, reply with exactly NOT FOUND "
    "and nothing else. Never mention searching or results."
)
NOT_FOUND = "NOT FOUND"

# The search words: the model's own, for now (30 October 2026: the question as asked, with the date
# stuck on, brought back pages that did not say who leads the Bundesliga).
QUERY = (
    "Write the one web search query most likely to find the up-to-date answer to the user's question. "
    "Use the current season or year where it helps. Reply with only the query, no quotes."
)

SEARCH = "SEARCH"



def order():
    """Providers in the order they are asked: LLM_PROVIDER first, then the rest."""
    first = (os.getenv("LLM_PROVIDER") or "groq").strip().casefold()
    names = list(ORDER)

    if first in names:
        names.remove(first)
        names.insert(0, first)

    return names


def _ask(name, question, system=PERSONA):
    config = PROVIDERS[name]
    key = (os.getenv(config["key_env"]) or "").strip()

    if not key:
        return None

    body = {
        "model": (os.getenv(config["model_env"]) or config["default_model"]).strip(),
        "messages": [
            {"role": "system", "content": system},
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


def _first(question, system):
    """The first provider's answer to [question], or None when every one failed."""
    for name in order():
        try:
            said = _ask(name, question, system)
        except Exception as error:
            print(f"[toy-cloud] {name} failed: {error}", flush=True)
            continue

        if said:
            return said

    return None


def _context(today, city):
    lines = []

    if today:
        lines.append(f"Today is {today}.")

    if city:
        lines.append(f"The toy is in {city}.")

    return (" " + " ".join(lines)) if lines else ""


def _with_results(question, found, today, city):
    shown = "\n".join(f"- {title}: {text}" for title, text in found)
    return _first(question, PERSONA + _context(today, city) + FROM_RESULTS + "\n\nResults:\n" + shown)


def _searched(question, today, city):
    """[question] answered from a search with the model's own words: a thorough search first, then
    the past week's news if that did not answer it. None when the model could not be reached."""
    context = _context(today, city)
    query = " ".join((_first(question, QUERY + context) or question).strip(" \"'").split())[:200] or question
    any_found = False

    for fresh in (False, True):
        found = search.search(query, fresh=fresh)
        print(f"[toy-cloud] search{' (news)' if fresh else ''} {query!r}: {len(found)} found", flush=True)

        if not found:
            continue

        any_found = True
        said = _with_results(question, found, today, city)

        if said is None:
            return None

        if said.strip(" .").upper() != NOT_FOUND:
            return said

    if not any_found:
        return "I couldn't look that up just now, sir. Try me again in a moment."

    return "I couldn't find that out just now, sir."


def answer(question, today="", city=""):
    """The toy's answer to [question] -- searched first when it needs live information -- or a short
    apology when every provider failed. [today] ("Friday 30 October 2026") and [city] are told to it."""
    sorry = "I can't reach my thinking just now, sir. Try me again in a moment."
    context = _context(today, city)

    decided = _first(question, DECIDE + context)
    live = bool(decided) and decided.strip(" .").upper().startswith(LIVE)
    print(f"[toy-cloud] {'live, searching' if live else 'known'}: {question!r}", flush=True)

    if live:
        return _searched(question, today, city) or sorry

    said = _first(question, PERSONA + context + MAY_SEARCH)

    if said is None:
        return sorry

    if said.strip(" .").upper() != SEARCH:
        return said

    return _searched(question, today, city) or sorry


GENRE = (
    "The user asked a toy to play some music on internet radio. Reply with the one music genre, as a "
    "radio station tag of one or two lowercase words (for example: jazz, chillout, classical, pop, "
    "quran), that best fits what they asked for. Reply with only the tag."
)


def genre(asked_for):
    """One genre tag for a music request no station or tag matched ("something relaxing" ->
    "chillout"), or None."""
    said = _first(asked_for, GENRE)

    if not said:
        return None

    tag = " ".join(re.sub(r"[^a-z0-9 &-]", " ", said.casefold()).split()[:2])
    return tag or None
