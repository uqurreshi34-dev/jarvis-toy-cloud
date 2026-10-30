"""Where the toy is and what time it is there, and the weather: answered free, without a model.

Where: a guess from the toy's own internet connection (its public address, as Render sees it),
looked up once a day at ipapi.co, or ipwho.is when that is busy (both free, no key), and kept in
memory. TOY_PLACE, when set, wins
("Birmingham", or "Birmingham, GB"); TOY_TIMEZONE likewise ("Europe/London"). Telling the toy where
it lives ("Jarvis, I live in ...") comes with the memory database. Nothing about the place is kept
beyond memory, and the address itself is never logged.

The weather is Open-Meteo's (free, no key), as PC JARVIS's.
"""

import datetime
import ipaddress
import os
import re
import threading
import time
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests

from toy import store

IP_URL = "https://ipapi.co/{ip}/json/"
IP_URL_SPARE = "https://ipwho.is/{ip}"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = (4, 8)
WEATHER_TIMEOUT = (6, 15)
PLACE_KEEP_SECONDS = 24 * 3600

DEFAULT_TIMEZONE = "Europe/London"

_lock = threading.Lock()
_places = {}   # key (address or TOY_PLACE) -> (place dict, monotonic time)


# ---- where ------------------------------------------------------------------------------------

def _public(ip):
    try:
        return ipaddress.ip_address(ip).is_global
    except ValueError:
        return False


def _remember(key, place):
    with _lock:
        _places[key] = (place, time.monotonic())

    return place


def _kept(key):
    with _lock:
        found = _places.get(key)

    if found and time.monotonic() - found[1] < PLACE_KEEP_SECONDS:
        return found[0]

    return None


def _named_place(name):
    """TOY_PLACE looked up by name at Open-Meteo's geocoder: {"city", "lat", "lon", "timezone"}."""
    city, _, country = (part.strip() for part in name.partition(","))
    params = {"name": city, "count": 1, "language": "en", "format": "json"}

    if country:
        params["countryCode"] = country.upper()

    response = requests.get(GEOCODE_URL, params=params, timeout=TIMEOUT)
    found = (response.json().get("results") or [None])[0] if response.status_code == 200 else None

    if not found:
        return {"city": city, "lat": None, "lon": None, "timezone": None}

    return {"city": found.get("name") or city, "lat": found.get("latitude"), "lon": found.get("longitude"),
            "timezone": found.get("timezone")}


def place(ip=None, toy=None):
    """Where the toy is: {"city", "lat", "lon", "timezone"}, any of them None when not known. What it
    was told ("Jarvis, I live in Leeds") first, then TOY_PLACE, then a guess from its address."""
    if toy:
        told = store.get_place(toy)

        if told:
            return told

    named = (os.getenv("TOY_PLACE") or "").strip()
    key = f"name:{named}" if named else f"ip:{ip}"
    kept = _kept(key)

    if kept:
        return kept

    nothing = {"city": None, "lat": None, "lon": None, "timezone": None}

    try:
        if named:
            return _remember(key, _named_place(named))

        if not ip or not _public(ip):
            return nothing

        found = _from_address(IP_URL, ip) or _from_address(IP_URL_SPARE, ip)

        if not found:
            print("[toy-cloud] where: neither address lookup knew where the toy is", flush=True)
            return nothing

        return _remember(key, found)
    except (requests.RequestException, ValueError):
        return nothing


def _from_address(url, ip):
    """A place from one address lookup service, or None. Both answer city, latitude and longitude;
    ipapi.co gives the time zone as a name, ipwho.is inside "timezone": {"id": ...}."""
    service = url.split("/")[2]

    try:
        response = requests.get(url.format(ip=ip), timeout=TIMEOUT, headers={"User-Agent": "JARVIS-toy-cloud"})
        data = response.json() if response.status_code == 200 else {}
    except (requests.RequestException, ValueError) as error:
        print(f"[toy-cloud] where: {service} failed ({type(error).__name__})", flush=True)
        return None

    if response.status_code != 200:
        print(f"[toy-cloud] where: {service} answered {response.status_code}", flush=True)

    if not isinstance(data, dict) or data.get("error") or data.get("success") is False or not data.get("city"):
        return None

    timezone = data.get("timezone")

    if isinstance(timezone, dict):
        timezone = timezone.get("id")

    return {"city": data.get("city"), "lat": data.get("latitude"), "lon": data.get("longitude"),
            "timezone": timezone}


def zone(where):
    """The time zone to tell the time in: TOY_TIMEZONE, else the place's, else DEFAULT_TIMEZONE."""
    for name in ((os.getenv("TOY_TIMEZONE") or "").strip(), (where or {}).get("timezone"), DEFAULT_TIMEZONE):
        if name:
            try:
                return ZoneInfo(name)
            except (ZoneInfoNotFoundError, ValueError):
                continue

    return ZoneInfo("UTC")


def now(where=None):
    return datetime.datetime.now(zone(where))


# ---- the time and the date ----------------------------------------------------------------------

_TIME = re.compile(
    r"^(?:what(?:s| is)\s+the\s+time|what\s+time\s+is\s+it|what\s+time\s+(?:is\s+it\s+)?now|the\s+time"
    r"|tell\s+me\s+the\s+time|have\s+you\s+got\s+the\s+time|do\s+you\s+know\s+(?:what\s+)?the\s+time(?:\s+is)?)"
    r"(?:\s+(?:please|now|right\s+now|sir))*$"
)
_DATE = re.compile(
    r"^(?:what(?:s| is)\s+(?:the\s+date|today(?:s)?\s+date|the\s+day|today)|what\s+day\s+is\s+(?:it|today)"
    r"|what\s+date\s+is\s+(?:it|today)|which\s+day\s+is\s+it|tell\s+me\s+the\s+date)"
    r"(?:\s+(?:please|today|sir))*$"
)


def _normal(text):
    text = str(text or "").casefold().replace("'", "").replace("\u2019", "")
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def time_asked(text):
    """"time" or "date" when [text] asks for one, otherwise None."""
    words = _normal(text)

    if _TIME.match(words):
        return "time"

    if _DATE.match(words):
        return "date"

    return None


def _ordinal(day):
    suffix = "th" if 10 <= day % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def time_words(asked, moment):
    if asked == "time":
        hour = moment.hour % 12 or 12
        half = "am" if moment.hour < 12 else "pm"
        return f"It's {hour}:{moment.minute:02d} {half}, sir."

    return f"It's {moment.strftime('%A')} the {_ordinal(moment.day)} of {moment.strftime('%B')}, sir."


# ---- the weather -------------------------------------------------------------------------------

_WEATHER = re.compile(
    r"\b(?:weather|forecast|raining|rain|snowing|snow|sunny|temperature|how\s+(?:hot|cold|warm)"
    r"|umbrella|coat|jacket)\b"
)

# Open-Meteo's weather codes (WMO), in words.
_CODES = {
    0: "clear skies", 1: "mostly clear skies", 2: "some cloud", 3: "overcast skies",
    45: "fog", 48: "freezing fog", 51: "light drizzle", 53: "drizzle", 55: "heavy drizzle",
    56: "freezing drizzle", 57: "freezing drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "light showers", 81: "showers", 82: "heavy showers", 85: "snow showers",
    86: "heavy snow showers", 95: "thunderstorms", 96: "thunderstorms with hail", 99: "thunderstorms with hail",
}


def weather_asked(text):
    """"today" or "tomorrow" when [text] asks about the weather (here, not somewhere named), else None."""
    words = _normal(text)

    if not _WEATHER.search(words):
        return None

    # "the weather in Paris" is about somewhere else: left to the search.
    if re.search(r"\b(?:in|at|for)\s+(?!the\b|here\b|today\b|tomorrow\b|tonight\b|this\b)[a-z]", words):
        return None

    return "tomorrow" if "tomorrow" in words else "today"


# Open-Meteo's free plan counts requests per internet address, and Render's free servers share
# their addresses with thousands of other apps: it answered 429, "Daily API request limit exceeded",
# on the first day (30 October 2026). So it is rested for the day after a 429, and MET Norway's
# forecast (free, no key, counted by who asks rather than from where: its terms ask for a User-Agent
# naming the app) is asked instead. Each place's forecast is kept WEATHER_KEEP_SECONDS.
MET_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"
MET_AGENT = "jarvis-toy-cloud/1.0 github.com/uqurreshi34-dev/jarvis-toy-cloud"
WEATHER_KEEP_SECONDS = 15 * 60
OPEN_METEO_REST_SECONDS = 6 * 3600

_weather = {}                 # (lat, lon) -> (forecast, monotonic time)
_open_meteo_rests_until = 0.0

# MET Norway's symbols ("lightrainshowers_day"), in words.
_SYMBOLS = {
    "clearsky": "clear skies", "fair": "mostly clear skies", "partlycloudy": "some cloud",
    "cloudy": "overcast skies", "fog": "fog",
    "lightrain": "light rain", "rain": "rain", "heavyrain": "heavy rain",
    "lightrainshowers": "light showers", "rainshowers": "showers", "heavyrainshowers": "heavy showers",
    "lightsleet": "light sleet", "sleet": "sleet", "heavysleet": "heavy sleet",
    "lightsleetshowers": "sleet showers", "sleetshowers": "sleet showers", "heavysleetshowers": "sleet showers",
    "lightsnow": "light snow", "snow": "snow", "heavysnow": "heavy snow",
    "lightsnowshowers": "snow showers", "snowshowers": "snow showers", "heavysnowshowers": "heavy snow showers",
}
_WET = ("rain", "sleet", "snow", "shower", "drizzle", "thunder")


def _sky_from_symbol(symbol):
    symbol = str(symbol or "").split("_")[0]

    if "thunder" in symbol:
        return "thunderstorms"

    return _SYMBOLS.get(symbol, "mixed weather")


def _from_open_meteo(lat, lon):
    """{"now", "days": [{"high", "low", "sky", "rain"}, ...]} or None (and rested after a 429)."""
    global _open_meteo_rests_until

    if time.monotonic() < _open_meteo_rests_until:
        return None

    try:
        response = requests.get(WEATHER_URL, timeout=WEATHER_TIMEOUT, params={
            "latitude": lat, "longitude": lon, "timezone": "auto", "forecast_days": 2,
            "current": "temperature_2m,weather_code",
            "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        })
    except requests.RequestException as error:
        print(f"[toy-cloud] weather: Open-Meteo failed ({type(error).__name__}: {error})", flush=True)
        return None

    if response.status_code == 429:
        _open_meteo_rests_until = time.monotonic() + OPEN_METEO_REST_SECONDS
        print("[toy-cloud] weather: Open-Meteo's daily limit is spent on Render's shared address; "
              "MET Norway instead for a while", flush=True)
        return None

    if response.status_code != 200:
        print(f"[toy-cloud] weather: Open-Meteo answered {response.status_code}: {response.text[:200]}", flush=True)
        return None

    try:
        data = response.json()
        daily = data["daily"]
        days = [{"high": round(daily["temperature_2m_max"][day]), "low": round(daily["temperature_2m_min"][day]),
                 "sky": _CODES.get(daily["weather_code"][day], "mixed weather"),
                 "rain": (daily.get("precipitation_probability_max") or [None, None])[day], "wet": False}
                for day in (0, 1)]
        now_temp = (data.get("current") or {}).get("temperature_2m")
    except (ValueError, KeyError, IndexError, TypeError) as error:
        print(f"[toy-cloud] weather: Open-Meteo's forecast was not as expected ({type(error).__name__})", flush=True)
        return None

    return {"now": now_temp, "days": days}


def _from_met_norway(lat, lon, zone_name):
    """The same from MET Norway's compact forecast: highs, lows and the sky worked out per local day."""
    try:
        response = requests.get(MET_URL, timeout=WEATHER_TIMEOUT, headers={"User-Agent": MET_AGENT},
                                params={"lat": lat, "lon": lon})
    except requests.RequestException as error:
        print(f"[toy-cloud] weather: MET Norway failed ({type(error).__name__}: {error})", flush=True)
        return None

    if response.status_code not in (200, 203):
        print(f"[toy-cloud] weather: MET Norway answered {response.status_code}: {response.text[:200]}", flush=True)
        return None

    try:
        series = response.json()["properties"]["timeseries"]
    except (ValueError, KeyError, TypeError):
        print("[toy-cloud] weather: MET Norway's forecast was not as expected", flush=True)
        return None

    where_zone = zone({"timezone": zone_name})
    today = datetime.datetime.now(where_zone).date()
    days = []

    for offset in (0, 1):
        date = today + datetime.timedelta(days=offset)
        temps, skies, wet = [], [], False

        for step in series:
            try:
                at = datetime.datetime.fromisoformat(step["time"].replace("Z", "+00:00")).astimezone(where_zone)
                data = step["data"]
                temperature = data["instant"]["details"]["air_temperature"]
            except (KeyError, ValueError, TypeError):
                continue

            if at.date() != date:
                continue

            temps.append(temperature)
            symbol = ((data.get("next_6_hours") or data.get("next_1_hours") or {}).get("summary") or {}).get("symbol_code")

            if symbol and 6 <= at.hour <= 18:
                skies.append(symbol)

            wet = wet or any(word in str(symbol or "") for word in _WET)

        if not temps:
            if offset == 0:
                return None

            break

        sky = _sky_from_symbol(max(set(skies), key=skies.count)) if skies else "mixed weather"
        days.append({"high": round(max(temps)), "low": round(min(temps)), "sky": sky, "rain": None, "wet": wet})

    try:
        now_temp = series[0]["data"]["instant"]["details"]["air_temperature"]
    except (KeyError, IndexError, TypeError):
        now_temp = None

    return {"now": now_temp, "days": days}


def forecast(where):
    """The forecast where the toy is, from whichever service answers, kept a quarter of an hour."""
    try:
        lat, lon = round(float(where["lat"]), 4), round(float(where["lon"]), 4)
    except (KeyError, TypeError, ValueError):
        return None

    with _lock:
        kept = _weather.get((lat, lon))

    if kept and time.monotonic() - kept[1] < WEATHER_KEEP_SECONDS:
        return kept[0]

    found = _from_open_meteo(lat, lon) or _from_met_norway(lat, lon, where.get("timezone"))

    if found:
        with _lock:
            _weather[(lat, lon)] = (found, time.monotonic())

    return found


def weather_words(when, where):
    """The weather where the toy is, said; None when no service could say."""
    if not where or where.get("lat") is None or where.get("lon") is None:
        return None

    found = forecast(where)
    day = 1 if when == "tomorrow" else 0

    if not found or len(found["days"]) <= day:
        return None

    today_or_tomorrow = found["days"][day]
    city = where.get("city") or "here"
    sky = today_or_tomorrow["sky"]

    if when == "today":
        now_temp = found.get("now")
        start = f"It's {round(now_temp)} degrees in {city} now, sir, with {sky} today" if now_temp is not None \
            else f"In {city} today, sir, expect {sky}"
    else:
        start = f"Tomorrow in {city}, sir, expect {sky}"

    said = f"{start}: a high of {today_or_tomorrow['high']} and a low of {today_or_tomorrow['low']}"
    rain = today_or_tomorrow.get("rain")

    if isinstance(rain, (int, float)) and rain >= 30:
        said += f", with a {round(rain)} percent chance of rain"
    elif today_or_tomorrow.get("wet") and not any(word in sky for word in _WET):
        said += ", with some rain at times"

    return said + "."


_LIVE_IN = re.compile(r"^(?:i|we)\s+(?:live|am\s+living|are\s+living)\s+in\s+([a-z][a-z .'-]{1,60})$"
                      r"|^(?:my|our)\s+home\s+is\s+in\s+([a-z][a-z .'-]{1,60})$"
                      r"|^(?:you\s+are|youre)\s+in\s+([a-z][a-z .'-]{1,60})$")


def lives_in_asked(text):
    """The town in "I live in Leeds" / "you're in Leeds", or None."""
    found = _LIVE_IN.match(_normal(text).replace("jarvis ", ""))

    if not found:
        return None

    town = next(group for group in found.groups() if group).strip()
    return " ".join(part.capitalize() for part in town.split())


def learn_place(toy, town):
    """Looks [town] up and keeps it as where [toy] lives. What to say."""
    try:
        found = _named_place(town)
    except (requests.RequestException, ValueError):
        found = None

    if not found or found.get("lat") is None:
        return f"I couldn't find {town} on the map, sir. Try saying it another way."

    if not store.set_place(toy, found):
        return "I couldn't save that just now, sir. Try me again in a moment."

    return f"Noted, sir. I'll give you the time and weather for {found['city']}."


UNKNOWN_PLACE = "I don't know where we are yet, sir. Tell me: Jarvis, I live in, and your town."


def forget():
    global _open_meteo_rests_until

    with _lock:
        _places.clear()
        _weather.clear()

    _open_meteo_rests_until = 0.0
