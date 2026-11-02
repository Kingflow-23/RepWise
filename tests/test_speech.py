"""Local speech validation and lazy transcription without model downloads."""

from io import BytesIO
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import wave

import pytest

from src.speech import WhisperTranscriber, validate_recording


def wav_recording(*, seconds=1, channels=1, sample_width=2):
    data = BytesIO()
    with wave.open(data, "wb") as recording:
        recording.setnchannels(channels)
        recording.setsampwidth(sample_width)
        recording.setframerate(16000)
        recording.writeframes(b"\0" * int(seconds * 16000 * channels * sample_width))
    return data.getvalue()


def test_pcm_microphone_recording_is_valid():
    validate_recording(wav_recording())


@pytest.mark.parametrize(
    "audio",
    [
        b"",
        b"not wav",
        wav_recording(seconds=0),
        wav_recording(sample_width=1),
        wav_recording(channels=3),
        wav_recording(seconds=121),
        wav_recording()[:-10],
    ],
)
def test_invalid_recordings_are_rejected(audio):
    with pytest.raises(ValueError):
        validate_recording(audio)


def test_model_is_loaded_once_and_transcribes_in_original_language(monkeypatch):
    model = Mock()
    model.transcribe.side_effect = lambda *_args, **_kwargs: (
        iter(
            [
                SimpleNamespace(text=" Je veux "),
                SimpleNamespace(text=" travailler les jambes. "),
            ]
        ),
        None,
    )
    constructor = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=constructor)
    )
    transcriber = WhisperTranscriber("base")
    constructor.assert_not_called()
    for _ in range(2):
        assert (
            transcriber.transcribe(wav_recording()) == "Je veux travailler les jambes."
        )
    constructor.assert_called_once_with("base", device="cpu", compute_type="int8")
    assert model.transcribe.call_args.kwargs == {
        "task": "transcribe",
        "beam_size": 5,
        "vad_filter": True,
        "condition_on_previous_text": False,
    }
    assert isinstance(model.transcribe.call_args.args[0], BytesIO)


def test_silence_does_not_create_a_prompt(monkeypatch):
    model = Mock()
    model.transcribe.return_value = (iter([]), None)
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        SimpleNamespace(WhisperModel=Mock(return_value=model)),
    )
    with pytest.raises(ValueError, match="No speech"):
        WhisperTranscriber().transcribe(wav_recording())


def test_model_failure_is_reported_without_transcript(monkeypatch):
    constructor = Mock(side_effect=RuntimeError("model unavailable"))
    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=constructor)
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        WhisperTranscriber().transcribe(wav_recording())


def test_invalid_audio_does_not_load_the_model(monkeypatch):
    constructor = Mock()
    monkeypatch.setitem(
        sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=constructor)
    )
    with pytest.raises(ValueError):
        WhisperTranscriber().transcribe(b"invalid")
    constructor.assert_not_called()
