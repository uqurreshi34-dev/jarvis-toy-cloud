"""Music on the toy: internet radio, by station or by genre ("Hey Jarvis, play some jazz",
"play Capital FM", "play a Quran station"), and pause, resume and stop.

Stations come from Radio Browser (radio-browser.info), a free public directory of about 50,000
stations, each broadcasting under its own licence -- how internet radios work. The toy plays the
stream itself (its chip decodes MP3), so only the station's address goes to it, in X-Jarvis-Play;
"pause", "resume" and "stop" go the same way. Nothing here is kept.

A request is looked up as a station's name first, then as a genre (a station tag), and only if both
find nothing does the model turn it into a genre ("something relaxing" -> "chillout"): no list of
genres or stations is kept here. Only MP3 streams over HTTPS that Radio Browser last found working,
not HLS, at most MAX_BITRATE kbps, most listened to first.

TOY_RADIO_SERVERS: Radio Browser's servers to ask, in turn (comma-separated), when the default
ones change.
"""

import os
import re

import requests

DEFAULT_SERVERS = ("https://de1.api.radio-browser.info", "https://de2.api.radio-browser.info",
                   "https://fi1.api.radio-browser.info")
USER_AGENT = "jarvis-toy-cloud/1.0 (+https://github.com/uqurreshi34-dev/jarvis-toy-cloud)"
TIMEOUT = (5, 10)
MAX_BITRATE = 192
LIMIT = 20

_PLAY = re.compile(
    r"^(?:(?:can|could|will|would)\s+you\s+)?(?:please\s+)?(?:play|put\s+on|stream|start\s+playing)\s+"
    r"(?:me\s+)?(?:some\s+|a\s+|an\s+|the\s+)?(.+?)(?:\s+please)?$")
_PAUSE = re.compile(r"^(?:please\s+)?pause(?:\s+(?:the\s+)?(?:music|radio|it))?(?:\s+please)?$")
_RESUME = re.compile(r"^(?:please\s+)?(?:resume|unpause|carry\s+on|continue|play|play\s+again|keep\s+playing)"
                     r"(?:\s+(?:the\s+)?(?:music|radio|it))?(?:\s+please)?$")
_STOP = re.compile(r"^(?:please\s+)?(?:stop|stop\s+playing|stop\s+the\s+(?:music|radio)|turn\s+(?:off\s+the\s+(?:music|radio)"
                   r"|the\s+(?:music|radio)\s+off))(?:\s+please)?$")

# Words that say "music" without being a station or a genre.
_FILLER = re.compile(r"\b(?:some|music|songs?|station|stations|radio|channel|please|a|an|the)\b")


def _words(text):
    text = str(text or "").casefold().replace("'", "").replace("\u2019", "")
    words = " ".join(re.sub(r"[^\w\s]", " ", text).split())
    return " ".join(word for word in words.split() if word not in ("jarvis", "sir", "hey"))


def asked(text):
    """("play", what), ("pause", None), ("resume", None) or ("stop", None); else None."""
    words = _words(text)

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


def _first_playable(stations):
    for station in stations:
        if isinstance(station, dict) and _playable(station):
            name = " ".join(str(station.get("name") or "a station").split())[:80]
            return {"name": name, "url": str(station.get("url_resolved") or station.get("url")).strip()}

    return None


def find(what, genre_for=None):
    """{"name", "url"} of a station for [what] -- a station's name, else a genre -- or None.
    [genre_for] turns a request into one genre word when neither finds anything (the model's)."""
    what = " ".join(str(what or "").split())[:80]

    if not what:
        return None

    station = _first_playable(_search(name=what))

    if station:
        return station

    genre = _FILLER.sub(" ", what.casefold())
    genre = " ".join(genre.split())

    if genre:
        station = _first_playable(_search(tag=genre))

        if station:
            return station

    if genre_for:
        guess = " ".join(str(genre_for(what) or "").casefold().split())[:40]

        if guess and guess != genre:
            print(f"[toy-cloud] radio: {what!r} taken as the genre {guess!r}", flush=True)
            return _first_playable(_search(tag=guess))

    return None
