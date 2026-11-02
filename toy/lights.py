"""The toy's light show: "Hey Jarvis, do a light show", "show me your lights", "rainbow".

The cloud says so and tells the toy (X-Jarvis-Show: lights); the toy's rings do the show itself, after
the words: every colour of the wheel, three times round (TOY_SHOW_* in the toy's firmware).
"""

import re

SAID = "Here's my light show, sir: every colour, three times round."

_SHOW = re.compile(r"^(?:(?:can|could|will|would)\s+you\s+)?(?:please\s+)?(?:(?:do|put\s+on|give\s+(?:me|us)|show\s+"
                   r"(?:me|us)|start)\s+)?(?:a\s+|the\s+|your\s+)?(?:light\s*show|lights\s+show|rainbow|"
                   r"(?:your\s+)?(?:lights|colou?rs))(?:\s+please)?$")


def _words(text):
    text = str(text or "").casefold().replace("'", "").replace("\u2019", "")
    words = " ".join(re.sub(r"[^\w\s]", " ", text).split())
    return " ".join(word for word in words.split() if word not in ("jarvis", "sir", "hey"))


def asked(text):
    """True for a light show asked for."""
    return bool(_SHOW.match(_words(text)))
