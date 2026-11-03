"""Local speech-to-text for short microphone recordings; no audio is persisted."""

from __future__ import annotations

from io import BytesIO
import logging
from threading import Lock
from typing import Protocol
import wave

logger = logging.getLogger(__name__)
MAX_AUDIO_BYTES = 5 * 1024 * 1024
MAX_AUDIO_SECONDS = 120


class SpeechTranscriber(Protocol):
    def transcribe(self, audio: bytes) -> str:
        """Return editable text from a WAV recording without submitting a plan."""


def validate_recording(audio: bytes) -> None:
    """Accept non-empty, complete PCM16 WAV audio lasting at most two minutes."""
    if not audio:
        raise ValueError("The recording is empty. Please record your request again.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("The recording is too large. Record up to two minutes.")
    try:
        with wave.open(BytesIO(audio), "rb") as recording:
            if (
                recording.getcomptype() != "NONE"
                or recording.getsampwidth() != 2
                or recording.getnchannels() not in (1, 2)
                or not 8000 <= recording.getframerate() <= 48000
            ):
                raise ValueError("Use a 16-bit WAV microphone recording.")
            frames = recording.getnframes()
            if not frames:
                raise ValueError(
                    "The recording is empty. Please record your request again."
                )
            if frames / recording.getframerate() > MAX_AUDIO_SECONDS:
                raise ValueError("The recording is too long. Record up to two minutes.")
            expected = frames * recording.getnchannels() * recording.getsampwidth()
            if len(recording.readframes(frames)) != expected:
                raise ValueError("The recording is incomplete. Please record it again.")
    except (wave.Error, EOFError) as error:
        raise ValueError(
            "The recording is not a valid WAV file. Please record it again."
        ) from error


class WhisperTranscriber:
    """Lazily load multilingual Whisper and reuse it for local transcription.

    The first call downloads the configured model if it is not cached. The
    default base model uses CPU/int8. A lock protects loading and decoding;
    audio and transcripts are never written to disk or logged by this service.
    """

    def __init__(self, model_name: str = "base"):
        self.model_name = model_name
        self._model = None
        self._lock = Lock()

    def transcribe(self, audio: bytes) -> str:
        validate_recording(audio)
        with self._lock:
            try:
                if self._model is None:
                    from faster_whisper import WhisperModel

                    self._model = WhisperModel(
                        self.model_name, device="cpu", compute_type="int8"
                    )
                segments, _ = self._model.transcribe(
                    BytesIO(audio),
                    task="transcribe",
                    beam_size=5,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )
                text = " ".join(segment.text.strip() for segment in segments).strip()
            except ImportError as error:
                raise RuntimeError(
                    "Speech transcription is not installed. Install the project requirements and restart the API."
                ) from error
            except Exception as error:
                logger.exception("Local speech transcription failed")
                raise RuntimeError(
                    "Speech transcription is unavailable. Please try again or type your request."
                ) from error
        if not text:
            raise ValueError(
                "No speech was detected. Please record your request again."
            )
        return text
