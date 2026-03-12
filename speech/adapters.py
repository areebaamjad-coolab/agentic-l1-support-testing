from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class SpeechToText(Protocol):
    def transcribe(self, wav_path: str) -> str: ...


class TextToSpeech(Protocol):
    def speak(self, text: str) -> None: ...


@dataclass
class Pyttsx3TTS:
    def speak(self, text: str) -> None:
        import pyttsx3  # type: ignore

        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()


@dataclass
class NoopTTS:
    def speak(self, text: str) -> None:
        return

