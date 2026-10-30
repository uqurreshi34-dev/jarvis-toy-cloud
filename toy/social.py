"""Thanks, praise, "you good?" and affection: things said to JARVIS rather than asked of him.

Ported from JARVIS's actions/social.py and the matching pools in phrases.py, so the toy answers
these the same way the PC does, for free, without a model call. Every shape must be the whole
utterance: "I love you, now play some music" is a command, not affection.
"""

import random
import re
import threading

_ADDRESS = r"(?:\s+(?:jarvis|sir|mate|buddy|pal|man|bro|my\s+friend|my\s+man|old\s+friend))?"

_LEAD = (
    r"(?:(?:great|excellent|brilliant|perfect|nice|lovely|awesome|cool|good|fantastic|"
    r"amazing|superb|wonderful|right|alright|ok|okay|oh|ah|aw|cheers|yeah|yes|so|well|um|uh|and|that\s+s\s+great|"
    r"thats\s+great|that\s+was\s+(?:great|quick|fast|helpful|perfect))\s+)*"
)

_THANKS = (
    r"(?:thank\s+(?:you|u|ya|yous)|thankyou|thanks|thanx|thx|ty|cheers|ta|"
    r"much\s+appreciated|(?:i\s+)?(?:really\s+)?appreciate\s+(?:it|that|this|you))"
)

_AMOUNT = r"(?:\s+(?:so|very)\s+much|\s+a\s+(?:lot|bunch|million|ton)|\s+loads|\s+again)?"

_REASON = (
    r"(?:\s+for\s+(?:that|this|it|everything|all\s+that|all\s+of\s+that|helping|"
    r"your\s+help|the\s+help|the\s+\w+|your\s+\w+))?"
)

_PRAISE = (
    r"(?:(?:good|great|nice|excellent|brilliant|awesome|top|amazing)\s+(?:job|work|one|stuff)|"
    r"well\s+done|nicely\s+done|nice\s+going|"
    r"(?:you\s+are|youre|you\s+re)\s+(?:the\s+best|brilliant|amazing|a\s+star|a\s+legend|a\s+genius|awesome)|"
    r"(?:what\s+a\s+)?legend|(?:what\s+a\s+)?star)"
)

_GRATITUDE = re.compile(rf"^{_LEAD}(?:{_PRAISE}{_ADDRESS}\s+)?{_THANKS}{_AMOUNT}{_REASON}{_ADDRESS}$")

_PRAISED = re.compile(rf"^{_LEAD}{_PRAISE}{_ADDRESS}$")

_WELLBEING = re.compile(
    r"^(?:"
    r"(?:are\s+)?(?:you|u)\s+(?:good|ok|okay|alright|all\s+right|well|doing\s+(?:ok|okay|alright|well|good|fine))"
    r"|how\s+(?:are|r)\s+(?:you|u)(?:\s+(?:doing|getting\s+on|holding\s+up|keeping|feeling))?"
    r"(?:\s+(?:today|tonight|this\s+(?:morning|afternoon|evening)))?"
    r"|how\s+(?:you|u)\s+doing"
    r"|how\s+(?:is|s)\s+it\s+going|hows\s+it\s+going"
    r"|how\s+are\s+things|hows\s+things|how\s+s\s+things"
    r"|whats\s+up|what\s+s\s+up|sup|wassup"
    r"|you\s+all\s+good|all\s+good"
    r")"
    rf"{_ADDRESS}$"
)

_FEELING = r"(?:\s+(?:so\s+much|very\s+much|loads|lots|a\s+lot|too|as\s+well|3000|forever|always))*"

_LOVE = re.compile(
    rf"^{_LEAD}(?:i\s+)?(?:really\s+|just\s+|do\s+)?(?:love|luv|adore)\s+(?:you|u|ya){_FEELING}{_ADDRESS}$"
    rf"|^{_LEAD}love\s+(?:you|u|ya){_FEELING}{_ADDRESS}$"
    rf"|^{_LEAD}(?:we|i)\s+love\s+(?:you|u){_FEELING}{_ADDRESS}$"
)

_FOND = re.compile(
    rf"^{_LEAD}(?:i\s+)?(?:really\s+|just\s+|do\s+)?(?:like|fancy|dig)\s+(?:you|u|ya){_FEELING}{_ADDRESS}$"
    rf"|^{_LEAD}(?:you\s+are|youre|you\s+re|ur)\s+(?:so\s+|really\s+|very\s+)?"
    rf"(?:cool|nice|sweet|kind|lovely|cute|funny|great|fun|clever|smart){_ADDRESS}$"
)

_FRIEND = re.compile(
    rf"^{_LEAD}(?:you\s+are|youre|you\s+re|ur)\s+(?:my\s+)?(?:best\s+|bestest\s+|good\s+|true\s+)?"
    rf"(?:friend|buddy|pal|mate|bestie|bff){_ADDRESS}$"
    rf"|^{_LEAD}(?:we\s+are|were|we\s+re)\s+(?:best\s+|good\s+)?friends{_ADDRESS}$"
    rf"|^{_LEAD}(?:will\s+you\s+be|can\s+you\s+be|be)\s+my\s+(?:best\s+)?friend{_ADDRESS}$"
    rf"|^{_LEAD}(?:are\s+you|r\s+u)\s+my\s+(?:best\s+)?friend{_ADDRESS}$"
)

_MISSED = re.compile(rf"^{_LEAD}(?:i\s+)?(?:really\s+)?(?:missed|miss)\s+(?:you|u|ya){_FEELING}{_ADDRESS}$")

_AFFECTION = (("love", _LOVE), ("fond", _FOND), ("friend", _FRIEND), ("missed", _MISSED))

# The warm kinds: answered with the toy's ring pulsing red.
FOND_KINDS = frozenset({"love", "fond", "friend", "missed"})

_CALLED = re.compile(r"^(?:jarvis|hey|hi|oi|please)\s+")

POOLS = {
    "welcome": (
        "You're welcome, sir.",
        "My pleasure, sir.",
        "Any time, sir.",
        "Always a pleasure, sir.",
        "Happy to help, sir.",
        "Glad to be of use, sir.",
    ),
    "praise": (
        "Thank you, sir.",
        "Kind of you to say, sir.",
        "I do my best, sir.",
        "Much appreciated, sir.",
        "High praise, sir. Thank you.",
    ),
    "wellbeing": (
        "All systems running smoothly, sir. Thank you for asking.",
        "Never better, sir.",
        "In fine form, sir.",
        "Very well, sir. Thank you for asking.",
        "Running like clockwork, sir.",
    ),
    "love": (
        "And I'm very fond of you, sir. It's an honour to look after you.",
        "That's kind of you to say, sir. You've made my circuits glow.",
        "Thank you, sir. I'll always be here when you need me.",
        "The feeling is entirely mutual, sir, in every way a machine can manage.",
        "That means a great deal, sir. You're my favourite person to help.",
    ),
    "fond": (
        "And I like you very much, sir.",
        "The feeling's mutual, sir. You're rather good company.",
        "Thank you, sir. You're not so bad yourself.",
        "You've made my day, sir.",
        "Most kind, sir. I enjoy our chats too.",
    ),
    "friend": (
        "Always, sir. You can count on me.",
        "I'd be honoured to be your friend, sir.",
        "Friends it is, sir. I'll be right here.",
        "The very best of friends, sir.",
        "Of course, sir. That's what I'm here for.",
    ),
    "missed": (
        "And I missed you, sir. Welcome back.",
        "It's good to hear your voice again, sir.",
        "I kept everything running while you were away, sir. Good to have you back.",
        "Welcome back, sir. It was rather quiet without you.",
    ),
}

_last = {}
_lock = threading.Lock()


def _normalise(said):
    text = str(said or "").casefold().replace("'", "").replace("\u2019", "")
    text = " ".join(re.sub(r"[^\w\s]", " ", text).split())
    previous = None

    while previous != text:
        previous = text
        text = _CALLED.sub("", text)

    return text


def kind(said):
    """'gratitude', 'praise', 'wellbeing', 'love', 'fond', 'friend', 'missed', or None."""
    text = _normalise(said)

    if not text:
        return None

    if _GRATITUDE.match(text):
        return "gratitude"

    if _PRAISED.match(text):
        return "praise"

    if _WELLBEING.match(text):
        return "wellbeing"

    for which, shape in _AFFECTION:
        if shape.match(text):
            return which

    return None


def reply(which):
    """What JARVIS says back: varied, never the same line twice running."""
    pool = POOLS.get("welcome" if which == "gratitude" else which)

    if not pool:
        return None

    with _lock:
        previous = _last.get(which)
        choices = [line for line in pool if line != previous] or list(pool)
        chosen = random.choice(choices)
        _last[which] = chosen

    return chosen
