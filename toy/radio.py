"""Music on the toy: internet radio, by station or by genre ("Hey Jarvis, play some jazz",
"play Capital FM", "play a Quran station"), and pause, resume and stop.

Stations come from Radio Browser (radio-browser.info), a free public directory of about 50,000
stations, each broadcasting under its own licence -- how internet radios work. The toy plays the
stream itself (its chip decodes MP3), so only the station's address goes to it, in X-Jarvis-Play;
"pause", "resume" and "stop" go the same way. Nothing here is kept.

A request is looked up as a genre first (a station tagged exactly so: "jazz"), then as a station's
name ("Capital FM"), and only if both find nothing does the model turn it into a genre ("something
relaxing" -> "chillout"): no list of genres or stations is kept here. Names first had "play some
jazz" pick "Lounge Ibiza Chillout Lounge ... Jazz ...", a name stuffed with genres to be found, and
read it all out for nine seconds (1 November 2026); a name is said only up to its first dash, bar,
comma or bracket, at most SPOKEN_WORDS words, and a station is found by name only when what was asked
for is in that part: "play kendrick lamar" found "70s & 80s - ... Kendrick Lamar, Drake ...", a name
listing artists to be found, and played it (2 November 2026). An artist the model has to place gets
their genre, and the toy says it cannot choose their songs: radio plays what the station plays. Only MP3 streams over HTTPS that Radio Browser last found working,
not HLS, at most MAX_BITRATE kbps, most listened to first.

TOY_RADIO_SERVERS: Radio Browser's servers to ask, in turn (comma-separated), when the default
ones change.
"""

import os
import re
from difflib import SequenceMatcher

import requests

DEFAULT_SERVERS = ("https://de1.api.radio-browser.info", "https://de2.api.radio-browser.info",
                   "https://fi1.api.radio-browser.info")
USER_AGENT = "jarvis-toy-cloud/1.0 (+https://github.com/uqurreshi34-dev/jarvis-toy-cloud)"
TIMEOUT = (5, 10)
MAX_BITRATE = 192
LIMIT = 20
SPOKEN_WORDS = 5

_PLAY = re.compile(
    r"^(?:(?:can|could|will|would)\s+you\s+)?(?:please\s+)?(?:play|put\s+on|stream|start\s+playing)\s+"
    r"(?:me\s+)?(?:some\s+|a\s+|an\s+|the\s+)?(.+?)(?:\s+please)?$")
_PAUSE = re.compile(r"^(?:please\s+)?pause(?:\s+(?:the\s+)?(?:music|radio|it))?(?:\s+please)?$")
_RESUME = re.compile(r"^(?:please\s+)?(?:resume|unpause|carry\s+on|continue|play|play\s+again|keep\s+playing)"
                     r"(?:\s+(?:the\s+)?(?:music|radio|it))?(?:\s+please)?$")
_STOP = re.compile(r"^(?:please\s+)?(?:stop|stop\s+playing|stop\s+the\s+(?:music|radio)|turn\s+(?:off\s+the\s+(?:music|radio)"
                   r"|the\s+(?:music|radio)\s+off))(?:\s+please)?$")

_NOW = re.compile(r"^(?:what(?:s| is)\s+(?:playing|this(?:\s+song)?|(?:the\s+)?song)|what\s+song\s+is\s+(?:this|it|playing)"
                  r"|who(?:s| is)\s+(?:this|singing|playing)|who\s+sings\s+this|what\s+(?:am\s+i|are\s+we)\s+"
                  r"listening\s+to)(?:\s+(?:now|please))?$")
# "Next song", "skip", "play the next one", "another song": the next song of what is playing.
_NEXT = re.compile(r"^(?:please\s+)?(?:(?:play\s+)?(?:the\s+)?next(?:\s+(?:song|one|track))?|skip(?:\s+(?:it|this|"
                   r"(?:this|the)\s+(?:song|one|track)))?|(?:play\s+)?another\s+(?:song|one|track)|change\s+"
                   r"(?:the\s+)?(?:song|track))(?:\s+please)?$")
_JAMENDO_WORDS = frozenset({"jamendo", "from", "on", "off", "some", "music", "songs", "a", "an", "the", "please"})

# Words that say "music" without being a station or a genre.
_FILLER = re.compile(r"\b(?:some|music|songs?|station|stations|radio|channel|please|a|an|the)\b")


def _words(text):
    text = str(text or "").casefold().replace("'", "").replace("\u2019", "")
    words = " ".join(re.sub(r"[^\w\s]", " ", text).split())
    return " ".join(word for word in words.split() if word not in ("jarvis", "sir", "hey"))


def asked(text):
    """("play", what), ("next", None), ("pause", None), ("resume", None) or ("stop", None); else None."""
    words = _words(text)

    # Before "play ...": "play the next song" is not a station called "the next song" (2 November 2026).
    if _NEXT.match(words):
        return "next", None

    if _PAUSE.match(words):
        return "pause", None

    if _RESUME.match(words):
        return "resume", None

    if _STOP.match(words):
        return "stop", None

    found = _PLAY.match(words)

    if found and found.group(1).strip():
        return "play", found.group(1).strip()

    return None


def now_asked(text):
    """True for "what's playing", "what song is this", "who is this", "who's singing"."""
    return bool(_NOW.match(_words(text)))


# How alike a word (or two or three run together) must be to "jamendo" to be it: Whisper, not knowing
# the name, wrote "JAM-endo" as "jam endo" and its like, and the radio was searched instead (2 November
# 2026). It starts as "JAM" does ("j", or "g" as in "gemendo"): "jam", "commando" or "amend" are not it.
JAMENDO_LIKE = 0.8


def _jamendo_at(words):
    """(where, how many words) of "jamendo" as heard in [words] -- the closest run of one to three words,
    the longer on a tie -- or None."""
    best = None

    for at in range(len(words)):
        for count in (1, 2, 3):
            if at + count > len(words):
                break

            heard = "".join(words[at:at + count])
            heard = "j" + heard[1:] if heard[:1] == "g" else heard
            like = SequenceMatcher(None, heard, "jamendo").ratio() if heard[:1] == "j" else 0.0

            if like >= JAMENDO_LIKE and (best is None or like >= best[0]):
                best = (like, at, count)

        if best:
            return best[1], best[2]

    return None


def from_jamendo(what):
    """For "hip hop from jamendo", "jamendo jazz", "jamendo" (or as Whisper writes it, "jam endo"): what is
    wanted from Jamendo ("hip hop", or "" for anything); None when Jamendo was not asked for."""
    words = _words(what).split()
    found = _jamendo_at(words)

    if not found:
        return None

    at, count = found
    words = words[:at] + words[at + count:]
    return " ".join(word for word in words if word not in _JAMENDO_WORDS)


def _servers():
    named = [server.strip().rstrip("/") for server in (os.getenv("TOY_RADIO_SERVERS") or "").split(",") if server.strip()]
    return named or list(DEFAULT_SERVERS)


def _search(**wanted):
    """Stations for [wanted] (name= or tag=), from the first server that answers; [] when none do."""
    params = {**wanted, "codec": "MP3", "is_https": "true", "hidebroken": "true", "order": "clickcount",
              "reverse": "true", "limit": LIMIT}

    for server in _servers():
        try:
            response = requests.get(f"{server}/json/stations/search", params=params, timeout=TIMEOUT,
                                    headers={"User-Agent": USER_AGENT})
        except requests.RequestException as error:
            print(f"[toy-cloud] radio: {server} failed: {error}", flush=True)
            continue

        if response.status_code != 200:
            print(f"[toy-cloud] radio: {server} answered {response.status_code}", flush=True)
            continue

        try:
            found = response.json()
        except ValueError:
            continue

        return found if isinstance(found, list) else []

    return []


def _playable(station):
    url = str(station.get("url_resolved") or station.get("url") or "").strip()

    try:
        bitrate = int(station.get("bitrate") or 0)
    except (TypeError, ValueError):
        bitrate = 0

    return (url.startswith("https://") and len(url) <= 400 and not station.get("hls")
            and str(station.get("codec") or "").upper() == "MP3" and bitrate <= MAX_BITRATE
            and station.get("lastcheckok", 1) in (1, True, "1"))


def spoken_name(name):
    """A station's name as said: up to its first dash, bar, comma, colon or bracket, a few words."""
    name = re.split(r"\s[-|~]\s|[|,:(\[]|\s-\s", str(name or ""))[0]
    words = name.split()[:SPOKEN_WORDS]
    return " ".join(words) or "a station"


def _first_playable(stations, named=None):
    """The first station the toy can play; with [named], only one whose name as said has those words."""
    wanted = set(_words(named).split()) if named else set()

    for station in stations:
        if not isinstance(station, dict) or not _playable(station):
            continue

        said = spoken_name(station.get("name"))

        if wanted and not wanted <= set(_words(said).split()):
            continue

        return {"name": said, "url": str(station.get("url_resolved") or station.get("url")).strip()}

    return None


def find(what, genre_for=None):
    """{"name", "url"} of a station for [what] -- a station's name, else a genre -- or None.
    [genre_for] turns a request into a genre when neither finds anything (the model's): a tag, or
    (tag, whether an artist was named); the station found that way has "genre" (and "artist") too."""
    what = " ".join(str(what or "").split())[:80]

    if not what:
        return None

    genre = _FILLER.sub(" ", what.casefold())
    genre = " ".join(genre.split())

    if genre:
        station = _first_playable(_search(tag=genre, tagExact="true"))

        if station:
            return station

    station = _first_playable(_search(name=what), named=what)

    if station:
        return station

    if genre_for:
        found = genre_for(what)
        guess, artist = found if isinstance(found, tuple) else (found, False)
        guess = " ".join(str(guess or "").casefold().split())[:40]

        if guess and guess != genre:
            print(f"[toy-cloud] radio: {what!r} taken as the genre {guess!r}{' (an artist)' if artist else ''}",
                  flush=True)
            station = _first_playable(_search(tag=guess, tagExact="true"))

            if station:
                return {**station, "genre": guess, "artist": bool(artist)}

    return None
