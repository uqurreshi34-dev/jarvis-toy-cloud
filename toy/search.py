"""Live answers: a web search (Tavily, as PC JARVIS's research uses), for what a model cannot know --
today's news, scores, league tables, prices, who holds a job now.

Asked when the model says the question needs it (brain.LIVE, or brain.SEARCH): no list of words
is kept here. The results' short texts go to the model with the question, and it
answers from them in one to three spoken sentences.
"""

import os
import threading
import time

import requests

TAVILY_URL = "https://api.tavily.com/search"
TIMEOUT = (5, 15)
MAX_RESULTS = 5
MAX_SNIPPET = 600

# A bad key or a spent allowance does not recover in minutes: left alone this long.
REST_SECONDS = 600.0

_lock = threading.Lock()
_resting_until = 0.0


def search(query, fresh=False):
    """[(title, text), ...] for [query], or [] when Tavily is not set up, resting, or failed. A thorough
    search ("advanced": fuller passages from each page); [fresh] keeps to the past week's news."""
    global _resting_until

    key = (os.getenv("TAVILY_API_KEY") or "").strip()

    with _lock:
        resting = time.monotonic() < _resting_until

    if not key or resting or not str(query or "").strip():
        return []

    try:
        body = {
            "query": str(query).strip(),
            "search_depth": "advanced",
            "max_results": MAX_RESULTS,
            "include_answer": True,
            "safe_search": True,
        }

        if fresh:
            body.update(topic="news", time_range="week")

        response = requests.post(TAVILY_URL, timeout=TIMEOUT, headers={"Authorization": f"Bearer {key}"}, json=body)
    except requests.RequestException as error:
        print(f"[toy-cloud] search failed: {error}", flush=True)
        return []

    if response.status_code in (401, 429, 432, 433):
        with _lock:
            _resting_until = time.monotonic() + REST_SECONDS

        print(f"[toy-cloud] search unavailable ({response.status_code}), resting", flush=True)
        return []

    if response.status_code != 200:
        print(f"[toy-cloud] search answered {response.status_code}", flush=True)
        return []

    try:
        data = response.json()
    except ValueError:
        return []

    found = []
    answer = str(data.get("answer") or "").strip()

    if answer:
        found.append(("Summary", answer[:MAX_SNIPPET]))

    for item in data.get("results") or []:
        title = str(item.get("title") or "").strip()
        text = " ".join(str(item.get("content") or "").split())[:MAX_SNIPPET]

        if title and text:
            found.append((title, text))

    return found


def rest_over():
    """For tests: Tavily may be asked again."""
    global _resting_until

    with _lock:
        _resting_until = 0.0
