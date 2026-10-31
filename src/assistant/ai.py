import json
import os
import pyaudio
from anthropic import Anthropic as _Anthropic
from constants import *

ANTHROPIC = _Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def _load_history() -> list[dict]:
    """Recover the saved conversation, or start fresh if there is none or it's unreadable."""
    try:
        saved = json.loads(Files.History.read_text())
    except (OSError, ValueError):
        return []
    if not isinstance(saved, list):
        return []
    saved = saved[-Limits.MaxHistory:]
    # history must start with a user turn
    while saved and saved[0].get("role") != "user":
        del saved[0]
    return saved

def _save_history() -> None:
    """Write history atomically so a power cut can't leave a half-written file."""
    tmp = Files.History.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(history, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, Files.History)


history = _load_history()


def send_message(message: str, max_tokens: int = 1024) -> str:
    """Send a message along with the conversation so far and return Claude's reply."""
    history.append({"role": "user", "content": message})
    try:
        response = Anthropic.messages.create(
            model=ModelConfig.Model,
            max_tokens=max_tokens,
            system=ModelConfig.SystemPrompt,
            messages=history,
        )
    except Exception:
        history.pop()  # don't keep a user turn that got no reply
        raise
    reply = response.content[0].text
    history.append({"role": "assistant", "content": reply})
    # drop the oldest exchange(s) so history always starts with a user turn
    while len(history) > Limits.MaxHistory:
        del history[:2]
    _save_history()
    return reply
