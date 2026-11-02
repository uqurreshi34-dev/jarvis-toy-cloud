"""Music from Jamendo (jamendo.com): independent artists' songs, free to stream, for the toy beside
internet radio. "Hey Jarvis, play some hip hop from Jamendo", "play Jamendo jazz", "play Jamendo".

Jamendo's API (developer.jamendo.com, v3.0) gives each track's MP3 (96 kbps, `mp31`) at an HTTPS
address, which the toy plays as it plays a station. The cloud keeps a list of tracks for the toy and
gives it one address, /toy/music/<list>: each time the toy joins it, the cloud sends it on (a
redirect, which the toy follows as it follows a station's) to the next track, so when a song ends
and the toy joins again, as it does when a station drops, the next one plays. Lists are kept in
memory for LIST_HOURS, at most MAX_LISTS of them; their names cannot be guessed, as the toy cannot
send its token when joining.

Jamendo's API terms: free for non-commercial use only (commercial use: licensing@jamendo.com), and
each track's artist and Jamendo credited: "Hey Jarvis, what's playing" names the song, its artist and
Jamendo.

JAMENDO_CLIENT_ID: the client id from devportal.jamendo.com (free). Without it, Jamendo is not offered.
"""

import os
import random
import re
import secrets
import threading
import time

import requests

API = "https://api.jamendo.com/v3.0/tracks/"
TIMEOUT = (5, 10)
LIMIT = 50
LIST_HOURS = 12
MAX_LISTS = 50
# Songs only: Jamendo also has 5, 11, 30 and 60 second cuts of tracks, loops for videos, which played
# as a song ending at once, or "what's playing" naming a song not the one heard (2 November 2026).
SONG_SECONDS = (90, 900)

# Jamendo's own genres (its site's list), as its tags are written: "hip hop" -> "hiphop".
GENRES = frozenset({"pop", "rock", "electronic", "hiphop", "jazz", "indie", "filmscore", "classical",
                    "chillout", "ambient", "folk", "metal", "latin", "rnb", "reggae", "punk", "country",
                    "house", "blues"})

_lock = threading.Lock()
_lists = {}


def client_id():
    return (os.getenv("JAMENDO_CLIENT_ID") or "").strip()


def ready():
    return bool(client_id())


def genre_tag(words):
    """Jamendo's genre for [words] ("hip hop" -> "hiphop", "r&b" -> "rnb"), or None."""
    tag = re.sub(r"[\s&_-]+", "", str(words or "").casefold().replace("r and b", "rnb"))
    tag = "rnb" if tag == "rb" else tag
    return tag if tag in GENRES else None


def tracks(what=""):
    """Playable tracks for [what] -- a genre, else words searched for -- most listened to this month:
    [{"name", "artist", "url"}]; [] when there are none or Jamendo cannot be reached."""
    if not ready():
        return []

    params = {"client_id": client_id(), "format": "json", "limit": LIMIT, "audioformat": "mp31",
              "type": "single albumtrack", "durationbetween": f"{SONG_SECONDS[0]}_{SONG_SECONDS[1]}"}
    what = " ".join(str(what or "").split())[:80]
    tag = genre_tag(what)

    if tag:
        params.update(tags=tag, order="popularity_month")
    elif what:
        params.update(search=what, boost="popularity_month")
    else:
        params.update(order="popularity_week")

    try:
        response = requests.get(API, params=params, timeout=TIMEOUT)
        found = response.json() if response.status_code == 200 else {}
    except (requests.RequestException, ValueError) as error:
        print(f"[toy-cloud] jamendo: failed: {error}", flush=True)
        return []

    header = found.get("headers") or {}

    if header.get("status") not in (None, "success"):
        print(f"[toy-cloud] jamendo: {header.get('error_message') or header.get('status')}", flush=True)
        return []

    playable = []

    for track in found.get("results") or []:
        url = str(track.get("audio") or "").strip()

        try:
            seconds = int(track.get("duration") or 0)
        except (TypeError, ValueError):
            seconds = 0

        if url.startswith("https://") and len(url) <= 400 and (not seconds or SONG_SECONDS[0] <= seconds <= SONG_SECONDS[1]):
            playable.append({"name": str(track.get("name") or "a song").strip(),
                             "artist": str(track.get("artist_name") or "an artist").strip(), "url": url})

    return playable


def make_list(toy, found, now=None):
    """A list of [found] tracks for [toy], shuffled: its name, for /toy/music/<name>."""
    now = time.time() if now is None else now
    name = secrets.token_urlsafe(16)
    order = list(found)
    random.shuffle(order)

    with _lock:
        for old in [key for key, kept in _lists.items() if now - kept["made"] > LIST_HOURS * 3600]:
            del _lists[old]

        while len(_lists) >= MAX_LISTS:
            del _lists[min(_lists, key=lambda key: _lists[key]["made"])]

        _lists[name] = {"toy": toy, "tracks": order, "at": 0, "made": now}

    return name


def next_track(name, now=None):
    """(toy, the next track) of list [name], from the start again after its last; None when it is gone."""
    now = time.time() if now is None else now

    with _lock:
        kept = _lists.get(name)

        if not kept or now - kept["made"] > LIST_HOURS * 3600 or not kept["tracks"]:
            return None

        track = kept["tracks"][kept["at"] % len(kept["tracks"])]
        kept["at"] += 1
        return kept["toy"], track


def has_list(name, now=None):
    """Whether list [name] is still kept (not too old, and the cloud not restarted since)."""
    now = time.time() if now is None else now

    with _lock:
        kept = _lists.get(name)
        return bool(kept and kept["tracks"] and now - kept["made"] <= LIST_HOURS * 3600)


def forget_all():
    """Every list (for tests)."""
    with _lock:
        _lists.clear()
