"""PC mode and cloud mode: "Jarvis, switch to PC mode" / "Jarvis, switch to cloud mode".

The toy keeps the choice itself (in its own flash), and is told it by the X-Jarvis-Switch header on
the answer: "pc" or "cloud". PC JARVIS's phone.py understands the same words, so either side can
send the toy to the other. The same matcher lives there (voice_modes in phone.py): keep them alike.
"""

import re

_SWITCH = re.compile(
    r"^(?:please\s+)?"
    r"(?:(?:switch|change|go|move|swap|turn|put\s+yourself)\s+(?:yourself\s+)?(?:over\s+)?(?:back\s+)?(?:to|into|in)\s+)?"
    r"(?:the\s+)?(pc|computer|cloud)\s+mode(?:\s+please)?$"
)


def switch_asked(text):
    """"pc" or "cloud" when [text] asks the toy to change mode ("switch to PC mode", "cloud mode",
    "go back to P.C. mode please"), otherwise None."""
    words = str(text or "").casefold().replace(".", "").replace("'s", "")
    words = " ".join(re.sub(r"[^\w\s]", " ", words).split())
    words = re.sub(r"\bp c\b", "pc", words)
    words = " ".join(word for word in words.split() if word not in ("jarvis", "sir", "now"))
    found = _SWITCH.match(words)

    if not found:
        return None

    return "cloud" if found.group(1) == "cloud" else "pc"
