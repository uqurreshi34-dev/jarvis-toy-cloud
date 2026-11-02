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
  `/health?detail=1` adds the memory, the face models (`ready`, or what they are doing and for how
  long), and which process answered with its parent: open it in a browser, no laptop needed.
  Every face the toy sends is logged as `faces (process): ...`, learning or recognising, and a face
  waits at most 20 s for busy models instead of hanging the toy.
  The models are loaded by the first request each process answers, never at import: Render's gunicorn
  imports the app in a parent process and forks the worker, and a worker forked mid-load inherited
  a lock nobody could release.
- `POST /toy/talk`: the same contract as PC JARVIS's `phone.py`. A WAV in (16 kHz mono 16-bit)
  with `X-Jarvis-Token`, `X-Jarvis-Camera` and `X-Jarvis-Toy` (`button`, `sound`, or `wake` when
  the toy heard "Jarvis" on its own chip, so the name need not be in the audio); raw PCM out
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
| `JAMENDO_CLIENT_ID` | Music from Jamendo (free, devportal.jamendo.com); without it, radio only |

Optional, with defaults: `GROQ_MODEL` (openai/gpt-oss-20b), `GROQ_WHISPER_MODEL`
(whisper-large-v3-turbo), `GOOGLE_TTS_VOICE` (en-GB-Chirp3-HD-Achird), `GOOGLE_TTS_RATE`,
`GOOGLE_TTS_MONTHLY_CHARACTERS` (950000, under Chirp 3 HD's free million).

Render: build `pip install -r requirements.txt && python -m toy.faces`, start
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
- Someone coming into view is held up to `FACE_WAIT` (5 s) for their face, which the toy sends a
  moment after its report: a known face makes it one sentence, "I can see you, Umer." (or "Good
  evening, Umer." when it is time for a greeting); a stranger's face, or the toy's next report with
  no face, lets "I can see someone, sir." go as it was.
- The toy also looks for a face every 10 s in cloud mode, person or not, and sends one in view
  each minute: someone known back after `BACK_AFTER` (90 s) out of sight hears "I can see you,
  Umer." even when greeted lately, and "who am I" always has a recent face.
- Greetings come only as someone comes into view (first seen, or back after `BACK_AFTER`), never
  minutes later while they sit there.
- `POST /camera/face`: a face the toy found. Checked and let go for now; recognising it comes with
  the memory database. `TOY_DETECT_IGNORE` names things never said ("dining table,sink").

## The time, the weather, and live questions

- "What time is it", "what's the date": answered free, in the toy's own time zone.
- "What's the weather", "will it rain tomorrow": Open-Meteo, free, where the toy is.
- Where the toy is: a guess from its internet connection (ipapi.co, or ipwho.is), kept a day.
  `TOY_PLACE` ("Leeds, GB") and `TOY_TIMEZONE` ("Europe/London"), when set, win.
- Every other question goes to the model (Groq, then Gemini, then DeepSeek). It first decides for
  itself, with no list of words or leagues kept here, whether a right answer needs live information
  (any league's table, news, prices, who holds a job now): if so, a Tavily search, and the answer
  comes from the results. Each decision is logged ("live, searching" or "known"). The model writes
  its own search words (with the season or year), Tavily searches thoroughly ("advanced"), and when
  those pages do not answer it, the past week's news is searched once more; each search is logged
  with how many pages it found.
  The model is always told today's date and the town.

## Faces and memory

- "Jarvis, learn my face, I'm Sarah": the toy is asked for a few faces (`X-Jarvis-Faces`), each is
  fingerprinted here (OpenCV's YuNet and SFace, fetched on first use and checked by SHA-256), and
  kept as Sarah's. Anyone can, each by their own name. A known face coming into view is greeted
  by name, once in ten minutes. A stranger's face is compared and let go, never kept.
- Said casually too: "learned my face" (as Whisper often hears it), "can you remember me, I'm Sarah",
  "I'm Sarah, learn my face please", "remember my face, Sarah".
- Someone it knew in the last minute coming into view again is still them: no "I can see someone".
- "Who am I", "who do you know", "forget my face", "forget Sarah", "forget everyone".
- "Jarvis, I live in Leeds": kept, and used for the time and the weather before any guess.
- "What can you see", "is anyone there": from the toy's last report, and who, if it knows them.
- Kept in Postgres at `DATABASE_URL` (Neon's free plan). Without it, in memory until restart.
- "Hey Jarvis, mute" is kept there too (`toy_muted`), so a muted toy stays muted when Render restarts,
  until "Hey Jarvis, unmute". Every talk answer, silence included, says `X-Jarvis-Muted: 1` or `0`, and
  `/health` asked with the toy's token and name says `"muted"`: the toy's ring pulses red while muted.

## Music

- "Hey Jarvis, play some jazz" / "play Capital FM" / "play a Quran station" / "play something relaxing":
  internet radio from Radio Browser's free directory (`toy/radio.py`), a genre first (an exact tag),
  then a station's name, and only then the model's genre for the words; a name is said short. Only MP3 over HTTPS, working, not HLS, at most
  192 kbps, most listened to first. The toy gets the address in `X-Jarvis-Play` and plays the stream
  itself; "pause", "resume" and "stop" go the same way. `TOY_RADIO_SERVERS` overrides the servers.
- A station is found by name only when the words are in its name as said, not in a name listing artists
  to be found. "Play Kendrick Lamar": an artist the model has to place gets a station of their genre,
  and the toy says it cannot choose their songs on the radio; a station named for them ("Bob Marley
  Radio") plays as itself.
- Jamendo (`toy/jamendo.py`): "Hey Jarvis, play some hip hop from Jamendo" / "play Jamendo jazz" /
  "play Jamendo": independent artists' songs, free to stream, 96 kbps MP3 over HTTPS. Whisper is told
  the name, and what it writes for "JAM-endo" ("jam endo", "jamaindo", "gemendo") is taken as it. The toy is given
  `/toy/music/<list>`, which sends it on to the next song each time it joins (as a station that drops
  is joined again), so songs follow one another; a list is kept in memory for 12 hours, and its name
  cannot be guessed. A request no station matches at all plays from Jamendo instead. Resuming moves
  on to the next song. Needs `JAMENDO_CLIENT_ID` (free, devportal.jamendo.com); without it, it says so.
  Jamendo's API is free for non-commercial use only (commercial: licensing@jamendo.com), crediting
  each song's artist and Jamendo.
- "Hey Jarvis, what's playing" / "who is this" / "what song is this": the Jamendo song and its artist,
  or the station.

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
