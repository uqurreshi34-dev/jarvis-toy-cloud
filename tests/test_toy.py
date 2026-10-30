"""Tests for the toy cloud. No network: hearing, the models and the voice are stood in for."""

import array
import base64
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app as toy_app  # noqa: E402
from toy import brain, social, voice, wake  # noqa: E402

TOKEN = "test-token"
SPEECH = b"RIFF" + b"\0" * 40 + b"\1\0" * 16000   # a header and half a second of samples


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TOY_TOKEN", TOKEN)
    toy_app._toys.clear()
    heard = {"text": ""}
    monkeypatch.setattr(toy_app.hearing, "transcribe", lambda wav: heard["text"])
    monkeypatch.setattr(toy_app.brain, "answer", lambda text: f"answer to {text}")
    monkeypatch.setattr(toy_app.voice, "pcm16", lambda text: text.encode())
    test_client = toy_app.app.test_client()
    test_client.heard = heard
    return test_client


def talk(client, said, how="sound", token=TOKEN, body=SPEECH, toy="toy cam"):
    client.heard["text"] = said
    return client.post("/toy/talk", data=body, headers={
        "X-Jarvis-Token": token, "Content-Type": "audio/wav",
        "X-Jarvis-Toy": how, "X-Jarvis-Camera": toy})


def test_health_is_open_and_says_nothing_secret(client):
    response = client.get("/health")
    assert response.status_code == 200 and response.get_json() == {"ok": True}


def test_wrong_or_missing_token_is_refused(client, monkeypatch):
    assert talk(client, "jarvis hello", token="nope").status_code == 403
    monkeypatch.setenv("TOY_TOKEN", "")
    assert talk(client, "jarvis hello", token="").status_code == 403


def test_only_wav_and_not_too_large(client):
    response = client.post("/toy/talk", data=SPEECH, headers={
        "X-Jarvis-Token": TOKEN, "Content-Type": "audio/mpeg"})
    assert response.status_code == 400
    assert talk(client, "jarvis hi", body=b"x" * (toy_app.MAX_TALK_BYTES + 1)).status_code == 413


def test_hands_free_needs_his_name(client):
    assert talk(client, "what time is it").status_code == 204
    response = talk(client, "Jarvis, what time is it")
    assert response.status_code == 200 and response.data == b"answer to what time is it"
    assert response.headers["Content-Type"].startswith("audio/L16; rate=16000")
    assert response.headers["X-Jarvis-Mood"] == "normal"


def test_name_alone_gets_yes_sir_and_a_follow_up_needs_no_name(client):
    assert talk(client, "Jarvis.").data == b"Yes, sir?"
    assert talk(client, "tell me a joke").data == b"answer to tell me a joke"
    assert talk(client, "and another").data == b"answer to and another"


def test_follow_up_window_is_per_toy(client):
    talk(client, "Jarvis.", toy="toy cam")
    assert talk(client, "tell me a joke", toy="other toy").status_code == 204


def test_button_needs_no_name_and_does_not_open_a_follow_up(client):
    assert talk(client, "tell me a joke", how="button").data == b"answer to tell me a joke"
    assert talk(client, "and another").status_code == 204


def test_button_complains_where_hands_free_stays_quiet(client):
    assert talk(client, "", how="button").data == b"I couldn't make out what you said, sir."
    assert talk(client, "").status_code == 204
    assert talk(client, "um", how="button").data == b"I only caught a stray word, sir."
    assert talk(client, "jarvis hi", how="button", body=b"RIFF" + b"\0" * 100).data.startswith(b"That was too short")
    assert talk(client, "jarvis hi", body=b"RIFF" + b"\0" * 100).status_code == 204


def test_mute_and_unmute(client):
    assert talk(client, "Jarvis, mute").data.startswith(b"Muted, sir.")
    assert talk(client, "Jarvis, what time is it").status_code == 204
    assert talk(client, "Jarvis.").status_code == 204
    assert talk(client, "Jarvis, unmute please").data == b"I'm listening again, sir."
    assert talk(client, "Jarvis, what time is it").status_code == 200


def test_affection_is_warm_free_and_red(client, monkeypatch):
    monkeypatch.setattr(toy_app.brain, "answer", lambda text: pytest.fail("no model call for affection"))
    response = talk(client, "Jarvis, I love you")
    assert response.headers["X-Jarvis-Mood"] == "affection"
    assert response.data.decode() in social.POOLS["love"]
    thanks = talk(client, "Jarvis, thank you")
    assert thanks.headers["X-Jarvis-Mood"] == "normal" and thanks.data.decode() in social.POOLS["welcome"]


def test_a_failure_is_spoken_not_a_crash(client, monkeypatch):
    def broken(wav):
        raise RuntimeError("groq down")

    monkeypatch.setattr(toy_app.hearing, "transcribe", broken)
    response = talk(client, "", how="button")
    assert response.status_code == 200 and response.data == b"Something went wrong there, sir."


def test_no_audio_is_503(client, monkeypatch):
    monkeypatch.setattr(toy_app.voice, "pcm16", lambda text: None)
    assert talk(client, "Jarvis, hello").status_code == 503


@pytest.mark.parametrize("said, command", [
    ("Jarvis, what time is it", "what time is it"),
    ("hey jarvis how are you", "how are you"),
    ("Jovis, hello", "hello"),
    ("Jarvis.", ""),
    ("travis what time is it", None),
    ("open an issue on the jarvis repo", None),
    ("what time is it", None),
])
def test_addressed_command(said, command):
    assert wake.addressed_command(said) == command


def test_social_kinds():
    assert social.kind("I love you so much") == "love"
    assert social.kind("you're my best friend") == "friend"
    assert social.kind("I missed you") == "missed"
    assert social.kind("you're so cool") == "fond"
    assert social.kind("thanks a lot") == "gratitude"
    assert social.kind("I love you, now play some music") is None
    first = social.reply("love")
    assert social.reply("love") != first


def test_provider_order_follows_llm_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    assert brain.order() == ["gemini", "groq", "deepseek"]
    monkeypatch.setenv("LLM_PROVIDER", "nonsense")
    assert brain.order() == ["groq", "gemini", "deepseek"]


def test_brain_falls_through_to_the_next_provider(monkeypatch):
    asked = []

    def ask(name, question):
        asked.append(name)
        if name == "groq":
            raise RuntimeError("rate limited")
        return "Hello, sir."

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setattr(brain, "_ask", ask)
    assert brain.answer("hi") == "Hello, sir." and asked == ["groq", "gemini"]


def test_brain_apologises_when_all_fail(monkeypatch):
    monkeypatch.setattr(brain, "_ask", lambda name, question: None)
    assert brain.answer("hi").startswith("I can't reach my thinking")


def test_spoken_strips_markdown():
    assert brain.spoken("**Hello** sir, see https://x.y\n- one") == "Hello sir, see one"


def test_brain_request_shape(monkeypatch):
    sent = {}

    class Reply:
        status_code = 200

        @staticmethod
        def json():
            return {"choices": [{"message": {"content": "Good evening, sir."}}]}

    def post(url, json, timeout, headers):
        sent.update(url=url, body=json, headers=headers)
        return Reply()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    monkeypatch.setattr(brain.requests, "post", post)
    assert brain._ask("deepseek", "hi") == "Good evening, sir."
    assert sent["url"] == "https://api.deepseek.com/chat/completions"
    assert sent["body"]["model"] == "deepseek-flash"
    assert sent["body"]["thinking"] == {"type": "disabled"}
    assert sent["body"]["messages"][0]["role"] == "system"
    assert sent["headers"]["Authorization"] == "Bearer k"


def test_missing_key_skips_provider(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert brain._ask("gemini", "hi") is None


def test_voice_asks_google_for_16k_linear_and_raises_the_peak(monkeypatch):
    sent = {}
    samples = array.array("h", [0, 1000, -2000, 500])
    wav = b"RIFF" + b"\0" * 32 + b"data" + b"\0\0\0\0" + samples.tobytes()

    class Reply:
        status_code = 200

        @staticmethod
        def json():
            return {"audioContent": base64.b64encode(wav).decode()}

    def post(url, json, headers, timeout):
        sent.update(url=url, body=json, headers=headers)
        return Reply()

    monkeypatch.setenv("GOOGLE_TTS_API_KEY", "g")
    monkeypatch.delenv("GOOGLE_TTS_VOICE", raising=False)
    monkeypatch.setenv("GOOGLE_TTS_RATE", "1.1")
    monkeypatch.setattr(voice.requests, "post", post)
    pcm = voice.pcm16("Hello, sir.")
    out = array.array("h")
    out.frombytes(pcm)
    assert sent["body"]["audioConfig"] == {"audioEncoding": "LINEAR16", "sampleRateHertz": 16000,
                                          "speakingRate": 1.1}
    assert sent["body"]["voice"] == {"languageCode": "en-GB", "name": "en-GB-Chirp3-HD-Achird"}
    assert sent["headers"]["X-Goog-Api-Key"] == "g"
    assert len(out) == 4 and max(abs(s) for s in out) == int(0.95 * 32767)


def test_voice_monthly_guard(monkeypatch):
    monkeypatch.setenv("GOOGLE_TTS_API_KEY", "g")
    monkeypatch.setenv("GOOGLE_TTS_MONTHLY_CHARACTERS", "5")
    voice._used.update(month="", characters=0)
    monkeypatch.setattr(voice.requests, "post", lambda *a, **k: pytest.fail("over the limit, nothing sent"))
    assert voice.pcm16("Hello, sir.") is None
    voice._used.update(month="", characters=0)


def test_bad_voice_name_falls_back(monkeypatch):
    monkeypatch.setenv("GOOGLE_TTS_VOICE", "Puck; drop table")
    assert voice.voice_name() == voice.DEFAULT_VOICE
