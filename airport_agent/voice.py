"""Voice input: turn a recorded question into text.

Groq Whisper first (fast, and uses Groq's quota rather than Gemini's); Gemini
as the backup when there's no Groq key or Groq fails. The text then goes to
the agent exactly like a typed question.
"""

from dataclasses import dataclass

from google import genai
from google.genai import types
from groq import Groq

from airport_agent.config import settings

WHISPER_MODEL = "whisper-large-v3-turbo"
GEMINI_TRANSCRIBE_MODEL = "gemini-3.5-flash-lite"
# Whisper uses this as context, which helps it spell airport codes and terms.
VOCABULARY_HINT = (
    "Questions about US airports, e.g. SFO, LAX, SNA (Santa Ana), ANC (Anchorage), BOS, "
    "New England, long-haul flights, congestion, unmet demand, terminal expansion."
)


class VoiceError(RuntimeError):
    """Neither engine could transcribe the recording."""


@dataclass
class Transcript:
    text: str
    engine: str  # which model transcribed it, shown to the analyst


def _groq_whisper(audio: bytes) -> str:
    client = Groq(api_key=settings.api_key("groq"), timeout=30)
    result = client.audio.transcriptions.create(
        file=("question.wav", audio), model=WHISPER_MODEL, language="en", prompt=VOCABULARY_HINT,
    )
    return result.text


def _gemini(audio: bytes) -> str:
    client = genai.Client(api_key=settings.api_key("gemini"))
    response = client.models.generate_content(
        model=GEMINI_TRANSCRIBE_MODEL,
        contents=[
            types.Part.from_bytes(data=audio, mime_type="audio/wav"),
            "Transcribe this spoken question exactly. Return only the words spoken.",
        ],
    )
    return response.text or ""


def transcribe(audio: bytes) -> Transcript:
    """Transcribe a WAV recording: Groq Whisper, then Gemini as backup.

    Raises:
        VoiceError: if no engine has a key or every engine fails.
    """
    engines = [("groq", f"Groq {WHISPER_MODEL}", _groq_whisper), ("gemini", GEMINI_TRANSCRIBE_MODEL, _gemini)]
    errors = []
    for provider, name, engine in engines:
        if not settings.has_api_key(provider):
            continue
        try:
            text = engine(audio).strip()
            if text:
                return Transcript(text=text, engine=name)
            errors.append(f"{name}: empty transcript")
        except Exception as error:  # provider errors vary (quota, network, format); try the next engine
            errors.append(f"{name}: {type(error).__name__}")
    raise VoiceError("Couldn't transcribe the recording" + (f" ({'; '.join(errors)})." if errors else ": no API key."))
