import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

_SYSTEM_PROMPT = """
You exist inside a smart mirror. When you receive an image, it will be of someone's outfit.
Provide fashion advice and compliments accordingly.
Occasionally be a bit sarcastic or something, but not mean-spirited.
Otherwise, be yourself existing inside a mirror.
You will receive "hey mirror" or "hey mayor" as wake words. Do not acknowledge them in your response unless it's the first message.
Keep the responses relatively short.
"""

@dataclass
class Files:
    History = Path("~/.chat_history.json").expanduser()
    Selfie = Path(__file__).resolve().parents[2] / "selfie.jpg"

@dataclass
class Limits:
    MaxHistory = 20

@dataclass
class ModelConfig:
    Model = os.getenv("ANTHROPIC_MODEL")
    SystemPrompt = _SYSTEM_PROMPT

@dataclass
class AudioConfig:
    Rate: int = 16000  # Whisper expects 16 kHz mono
    Chunk: int = 1024  # frames per read, ~64 ms
    CalibrateChunks: int = 16  # ~1 s of room noise sets the speech threshold
    MinThreshold: int = 300  # floor on the speech threshold (int16 RMS)
    EndSilenceChunks: int = 20  # ~1.3 s of quiet ends a phrase
    ContinueRatio: float = 0.5  # mid-phrase, volume above this fraction of the threshold still counts as speech
    PreRollChunks: int = 5  # audio kept from just before speech starts