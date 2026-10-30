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


# ---- PC mode and cloud mode ---------------------------------------------------------------------

from toy import modes, seen  # noqa: E402


@pytest.mark.parametrize("said, mode", [
    ("switch to PC mode", "pc"),
    ("Switch to P.C. mode.", "pc"),
    ("go back to pc mode please", "pc"),
    ("computer mode", "pc"),
    ("switch to cloud mode", "cloud"),
    ("Cloud mode.", "cloud"),
    ("change to the cloud mode", "cloud"),
    ("what is cloud mode", None),
    ("switch on the pc", None),
    ("tell me about pc mode and cloud mode", None),
])
def test_switch_asked(said, mode):
    assert modes.switch_asked(said) == mode


def test_switch_is_told_to_the_toy(client):
    response = talk(client, "Jarvis, switch to PC mode")
    assert response.headers["X-Jarvis-Switch"] == "pc" and response.data.startswith(b"Switching to PC mode")
    response = talk(client, "switch to cloud mode", how="button")
    assert response.headers["X-Jarvis-Switch"] == "cloud"
    assert "X-Jarvis-Switch" not in talk(client, "Jarvis, hello").headers


def test_muted_toy_does_not_switch(client):
    talk(client, "Jarvis, mute")
    assert talk(client, "Jarvis, switch to PC mode").status_code == 204


def fall_back(client, said):
    client.heard["text"] = said
    return client.post("/toy/talk", data=SPEECH, headers={
        "X-Jarvis-Token": TOKEN, "Content-Type": "audio/wav", "X-Jarvis-Toy": "sound",
        "X-Jarvis-Camera": "toy cam", "X-Jarvis-Fallback": "pc"})


def test_fallback_is_said_once_then_not_again(client):
    first = fall_back(client, "Jarvis, hello").data.decode()
    assert first.startswith("Defaulting to the cloud, sir") and first.endswith("answer to hello")
    assert fall_back(client, "Jarvis, hello again").data == b"answer to hello again"


def test_fallback_says_nothing_when_not_addressed(client):
    assert fall_back(client, "what time is it").status_code == 204
    assert fall_back(client, "Jarvis, hi").data.decode().startswith("Defaulting to the cloud")


def test_fallback_note_comes_back_after_a_while(client):
    toy = toy_app._toy("toy cam")
    assert toy_app._fallback_note(toy, now=1000.0)
    assert not toy_app._fallback_note(toy, now=1000.0 + 60)
    assert toy_app._fallback_note(toy, now=1000.0 + toy_app.FALLBACK_REMIND_SECONDS + 1)


# ---- what the toy sees ------------------------------------------------------------------------

def report(*labels):
    return {"name": "toy cam", "objects": [{"label": label, "score": 0.8, "box": [0, 0, 1, 1]} for label in labels]}


def see(client, payload, toy="toy cam", token=TOKEN):
    return client.post("/camera/seen", json=payload, headers={"X-Jarvis-Token": token, "X-Jarvis-Camera": toy})


def announce(client, toy="toy cam"):
    return client.post("/toy/announce", headers={"X-Jarvis-Token": TOKEN, "X-Jarvis-Camera": toy})


@pytest.fixture
def fresh_sight():
    seen.forget()
    yield
    seen.forget()


def test_someone_coming_into_view_is_said_once(client, fresh_sight):
    assert see(client, report("person")).get_json() == {"ok": True, "say": True}
    assert announce(client).data == b"I can see someone, sir."
    assert announce(client).status_code == 204
    assert see(client, report("person")).get_json()["say"] is False


def test_more_of_them_is_news(client, fresh_sight):
    see(client, report("person", "cup"))
    assert announce(client).data == b"I can see someone and a cup, sir."
    see(client, report("person", "cup", "cup", "cup"))
    assert announce(client).data == b"I can see three cups, sir."


def test_back_after_being_away_is_news(fresh_sight):
    assert seen.record("t", report("person"), now=100.0)
    assert not seen.record("t", report("person"), now=110.0)
    assert seen.record("t", report("person"), now=110.0 + seen.AWAY_SECONDS + 1)


def test_ignored_things_are_never_said(client, fresh_sight, monkeypatch):
    monkeypatch.setenv("TOY_DETECT_IGNORE", "dining table, sink")
    assert see(client, report("dining table")).get_json()["say"] is False


def test_stale_announcement_is_dropped(fresh_sight):
    seen.record("t", report("person"), now=100.0)
    assert seen.take("t", now=100.0 + seen.PENDING_SECONDS + 1) is None


def test_seen_is_checked(client, fresh_sight):
    assert see(client, report("person"), token="nope").status_code == 403
    assert see(client, {"objects": "lots"}).status_code == 400
    assert see(client, {"objects": [{"label": "x" * 41}]}).status_code == 400


def test_muted_toy_announces_nothing(client, fresh_sight):
    talk(client, "Jarvis, mute")
    assert see(client, report("person")).get_json()["say"] is False
    assert announce(client).status_code == 204


def test_words():
    assert seen.words({"person": 2}) == "I can see two people, sir."
    assert seen.words({"bottle": 1, "person": 1, "cup": 2}) == "I can see someone, a bottle and two cups, sir."


def test_face_is_taken_and_let_go(client):
    headers = {"X-Jarvis-Token": TOKEN, "X-Jarvis-Camera": "toy cam", "Content-Type": "image/jpeg"}
    assert client.post("/camera/face", data=b"\xff\xd8jpeg", headers=headers).get_json() == {"ok": True}
    assert client.post("/camera/face", data=b"x" * (toy_app.MAX_FACE_BYTES + 1), headers=headers).status_code == 413
    headers["Content-Type"] = "image/png"
    assert client.post("/camera/face", data=b"png", headers=headers).status_code == 400
    headers["X-Jarvis-Token"] = "nope"
    assert client.post("/camera/face", data=b"\xff\xd8", headers=headers).status_code == 403
