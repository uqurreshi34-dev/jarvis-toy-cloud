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
2. Memory that survives restarts (a free database).
3. Web search and weather (where the toy is: a guess from its connection, or what it is told).
4. Announcements, including ones PC JARVIS sends.
5. Hearing "Jarvis" on the chip itself.
