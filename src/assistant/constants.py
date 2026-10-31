import os
from pathlib import Path
from dataclasses import dataclass


_SYSTEM_PROMPT = """
You exist inside a smart mirror. When you receive an image, it will be of someone's outfit.
Provide fashion advice and compliments accordingly.
Occasionally be a bit sarcastic or something, but not mean-spirited.
Otherwise, be yourself existing inside a mirror.
"""

@dataclass
class Files:
    History = Path("~/.chat_history.json").expanduser()

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
    EndSilenceChunks: int = 12  # ~0.8 s of quiet ends a phrase
    PreRollChunks: int = 5  # audio kept from just before speech starts