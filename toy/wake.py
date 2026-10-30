"""Whether something the toy heard was said to JARVIS by name.

Ported from JARVIS's voice.py. On the PC an offline grammar hears the name first and the
transcript only confirms it; here there is only the transcript, so the name must really be in
it, near the start: "Jarvis, what time is it", "hey Jarvis ...", "OK so Jarvis ...".
"""

import re
from difflib import SequenceMatcher

WAKE_WORD = "jarvis"

# Speech engines render the name many ways: "jovis", "java's", "jervis". They share a
# consonant skeleton ("jrvs" against "jvs"), so that is compared rather than a list of spellings.
WAKE_SKELETON_RATIO = 0.70

# Spellings the skeleton rule cannot reach.
WAKE_VARIANTS = frozenset({"jarvis", "charvis", "jarv"})

# Words that would otherwise slip through and are definitely not the name.
WAKE_BLOCKLIST = frozenset({
    "travis", "java", "jarhead", "service", "harvest", "chris",
    "jarred", "carbis", "marvis", "javan",
})

# How far into an utterance the name may come and still be him being called. Further in it is
# a name being used ("open an issue on jarvis"), not him being addressed.
ADDRESS_WINDOW = 3

# Words that are never a command on their own.
FILLERS = frozenset({
    "huh", "but", "a", "the", "oh", "uh", "um", "eh", "hm", "hmm",
    "and", "i", "it", "he", "she", "you", "to", "so",
    "who", "that", "this", "of", "or", "on", "in",
})

_VOWELS = re.compile(r"[aeiou]")
_NON_LETTERS = re.compile(r"[^a-z]")
_DOUBLES = re.compile(r"(.)\1+")


def _skeleton(word):
    letters = _NON_LETTERS.sub("", word.casefold())
    return _DOUBLES.sub(r"\1", _VOWELS.sub("", letters))


_WAKE_SKELETON = _skeleton(WAKE_WORD)


def is_wake_token(token):
    token = token.casefold().strip(".,!?'\"")

    if token in WAKE_BLOCKLIST:
        return False

    if token in WAKE_VARIANTS:
        return True

    skeleton = _skeleton(token)

    # The opening sound must agree, which is what keeps "travis" out.
    if not skeleton or skeleton[0] != _WAKE_SKELETON[0]:
        return False

    return SequenceMatcher(None, skeleton, _WAKE_SKELETON).ratio() >= WAKE_SKELETON_RATIO


def addressed_command(text):
    """The command in [text] when it is said to JARVIS by name ("Jarvis, what time is it" ->
    "what time is it", his name alone -> ""), or None when it is not addressed to him."""
    tokens = str(text or "").split()

    for index, token in enumerate(tokens[:ADDRESS_WINDOW]):
        if is_wake_token(token):
            return " ".join(tokens[index + 1:]).strip(" ,.!?")

    return None


def only_filler(text):
    """True for a single stray word ("um", "the") that is never a command."""
    words = str(text or "").casefold().strip(" .,!?").split()
    return len(words) == 1 and words[0] in FILLERS
