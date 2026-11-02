"""Speech requests must authenticate and must never generate or save a plan."""

from io import BytesIO
from unittest.mock import Mock
import wave

from fastapi.testclient import TestClient
import pytest

from config.config import AppConfig
from src.api.auth import create_access_token
from src.api.main import create_app
from src.retrieval.base import MissingRetriever
from src.speech import MAX_AUDIO_BYTES


def recording():
    output = BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0" * 32000)
    return output.getvalue()


@pytest.fixture
def speech_client(tmp_path):
    transcriber = Mock()
    transcriber.transcribe.return_value = "Je veux un programme pour les jambes."
    generator = Mock()
    secret = "test-speech-auth-secret-at-least-32-characters"
    app = create_app(
        AppConfig(database_path=str(tmp_path / "speech.sqlite3"), auth_secret=secret),
        retriever=MissingRetriever(), generator=generator, transcriber=transcriber,
    )
    headers = {"Authorization": "Bearer " + create_access_token("student", secret), "Content-Type": "audio/wav"}
    with TestClient(app) as client:
        yield client, headers, transcriber, generator


def test_recording_returns_only_editable_text(speech_client):
    client, headers, transcriber, generator = speech_client
    audio = recording()
    result = client.post("/api/v1/speech/transcribe", content=audio, headers=headers)
    assert result.status_code == 200, result.text
    assert result.json() == {"text": "Je veux un programme pour les jambes."}
    transcriber.transcribe.assert_called_once_with(audio)
    generator.generate.assert_not_called()
    assert client.get("/api/v1/students/student/plans", headers=headers).json() == []


def test_recording_requires_authentication(speech_client):
    client, _, transcriber, _ = speech_client
    result = client.post("/api/v1/speech/transcribe", content=recording(), headers={"Content-Type": "audio/wav"})
    assert result.status_code == 401
    transcriber.transcribe.assert_not_called()


@pytest.mark.parametrize("content,content_type,status", [
    (b"", "audio/wav", 422), (b"not wav", "audio/wav", 422),
    (b"mp3", "audio/mpeg", 415), (b"0" * (MAX_AUDIO_BYTES + 1), "audio/wav", 413),
])
def test_invalid_recording_never_reaches_transcriber(speech_client, content, content_type, status):
    client, headers, transcriber, _ = speech_client
    result = client.post("/api/v1/speech/transcribe", content=content, headers={**headers, "Content-Type": content_type})
    assert result.status_code == status
    transcriber.transcribe.assert_not_called()


@pytest.mark.parametrize("failure,status", [(ValueError("No speech was detected."), 422),
                                           (RuntimeError("Speech transcription is unavailable."), 503)])
def test_transcription_errors_do_not_generate_a_plan(speech_client, failure, status):
    client, headers, transcriber, generator = speech_client
    transcriber.transcribe.side_effect = failure
    result = client.post("/api/v1/speech/transcribe", content=recording(), headers=headers)
    assert result.status_code == status
    generator.generate.assert_not_called()
