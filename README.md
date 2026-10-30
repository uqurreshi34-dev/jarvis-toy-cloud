# jarvis-toy-cloud

The JARVIS toy's brain on the internet, so the toy works with the PC switched off.

The toy (DFRobot DFR1154 "toy cam") keeps its eyes and reflexes on the chip: camera, person and
face finding, listening, light ring, servo. What it hears goes to this service, which turns the
speech into text (Groq Whisper), answers it (Groq, then Gemini, then DeepSeek, or the free
built-in replies for thanks and affection) and sends the answer back as speech (Google Chirp 3
HD) for the toy's own speaker.

PC JARVIS keeps everything else: reports, trading, files, face names.

## Endpoints

- `GET /health`: `{"ok": true}`. For Render's health check and UptimeRobot.
- `POST /toy/talk`: the same contract as PC JARVIS's `phone.py`. A WAV in (16 kHz mono 16-bit)
  with `X-Jarvis-Token`, `X-Jarvis-Camera` and `X-Jarvis-Toy` (`button` or `sound`); raw PCM out
  (`audio/L16; rate=16000`) with `X-Jarvis-Mood` (`affection` or `normal`), or 204 to stay silent.

## Settings (Render, Environment)

Keys are only ever set in Render, never in this repo.

| Name | What for |
|---|---|
| `TOY_TOKEN` | The toy's password. Without it every request is refused. |
| `GROQ_API_KEY` | Speech to text, and the first brain |
| `GEMINI_API_KEY`, `GEMINI_MODEL` | Second brain |
| `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL` | Paid last resort |
| `LLM_PROVIDER` | Which brain is asked first (`groq`, `gemini` or `deepseek`) |
| `GOOGLE_TTS_API_KEY` | The voice |
| `TAVILY_API_KEY` | Web search (a later chunk) |
| `PYTHON_VERSION` | `3.12.7` |

Optional, with defaults: `GROQ_MODEL` (openai/gpt-oss-20b), `GROQ_WHISPER_MODEL`
(whisper-large-v3-turbo), `GOOGLE_TTS_VOICE` (en-GB-Chirp3-HD-Achird), `GOOGLE_TTS_RATE`,
`GOOGLE_TTS_MONTHLY_CHARACTERS` (950000, under Chirp 3 HD's free million).

Render: build `pip install -r requirements.txt`, start
`gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 60`, health check `/health`.

## PC mode and cloud mode

"Jarvis, switch to PC mode" / "Jarvis, switch to cloud mode", said to the toy: the answer carries
`X-Jarvis-Switch` and the toy keeps the choice. In PC mode with the PC off, the toy falls back here
with `X-Jarvis-Fallback: pc`, and the first answer in ten minutes says so. PC JARVIS's `phone.py`
knows the same words (keep `toy/modes.py` and its `TOY_SWITCH` alike).

## What the toy sees

- `POST /camera/seen`: what the toy's own detection is sure of (names and boxes, never a picture).
  Answers `{"ok": true, "say": true}` when something has come into view.
- `POST /toy/announce`: the words for it ("I can see someone, sir."), as PCM, or 204.
- `POST /camera/face`: a face the toy found. Checked and let go for now; recognising it comes with
  the memory database. `TOY_DETECT_IGNORE` names things never said ("dining table,sink").

## The time, the weather, and live questions

- "What time is it", "what's the date": answered free, in the toy's own time zone.
- "What's the weather", "will it rain tomorrow": Open-Meteo, free, where the toy is.
- Where the toy is: a guess from its internet connection (ipapi.co, or ipwho.is), kept a day.
  `TOY_PLACE` ("Leeds, GB") and `TOY_TIMEZONE` ("Europe/London"), when set, win.
- Live questions ("who's top of the Premier League"): a Tavily search first when the question plainly
  needs one, or when the model says it does (it answers just SEARCH); then answered from the results.
  The model is always told today's date and the town.

## Faces and memory

- "Jarvis, learn my face, I'm Sarah": the toy is asked for a few faces (`X-Jarvis-Faces`), each is
  fingerprinted here (OpenCV's YuNet and SFace, fetched on first use and checked by SHA-256), and
  kept as Sarah's. Anyone can, each by their own name. A known face coming into view is greeted
  by name, once in ten minutes. A stranger's face is compared and let go, never kept.
- "Who am I", "who do you know", "forget my face", "forget Sarah", "forget everyone".
- "Jarvis, I live in Leeds": kept, and used for the time and the weather before any guess.
- "What can you see", "is anyone there": from the toy's last report, and who, if it knows them.
- Kept in Postgres at `DATABASE_URL` (Neon's free plan). Without it, in memory until restart.

## Trying it from the PC

```
$env:TOY_TOKEN = 'your token'
python tools/try_talk.py
```

## Tests

```
python -m pytest -q
```

## Roadmap

1. Talk: done here.
1b. PC mode and cloud mode; what the toy sees, said on the toy: done here.
2. Memory that survives restarts, faces by name, what it can see: done here.
3. Web search, the time and the weather: done here (telling it where it lives comes with memory).
4. Announcements, including ones PC JARVIS sends.
5. Hearing "Jarvis" on the chip itself.
