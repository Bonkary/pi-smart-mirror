import base64
import io
import json
import os
import pyaudio
from anthropic import Anthropic as _Anthropic
from constants import *
from dotenv import load_dotenv
from PIL import Image, ImageOps

load_dotenv()



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
        json.dump(HISTORY, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, Files.History)


HISTORY = _load_history()
ANTHROPIC = _Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def _selfie_block() -> dict:
    """Load the selfie as an API image block, shrunk to fit the API's size limit."""
    with Image.open(Files.Selfie) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img.thumbnail((400, 400))  # the API downscales anything larger anyway
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.b64encode(buf.getvalue()).decode(),
        },
    }

def send_message(message: str, max_tokens: int = 1024, image: bool = False) -> str:
    """Send a message along with the conversation so far and return Claude's reply.

    With image=True the selfie is attached to this message only; the saved history
    keeps just a text note about it so the photo isn't re-sent or stored every time.
    """
    # the history only ever holds text; the photo is added to the outgoing request alone
    HISTORY.append({"role": "user", "content": f"[sent a photo] {message}" if image else message})
    messages = HISTORY
    if image:
        messages = HISTORY[:-1] + [{
            "role": "user",
            "content": [_selfie_block(), {"type": "text", "text": message}],
        }]
    try:
        response = ANTHROPIC.messages.create(
            model=ModelConfig.Model,
            max_tokens=max_tokens,
            system=ModelConfig.SystemPrompt,
            messages=messages,
        )
    except Exception:
        HISTORY.pop()  # don't keep a user turn that got no reply
        raise
    # the reply can start with a thinking block, so keep only the text blocks
    reply = "".join(block.text for block in response.content if block.type == "text")
    HISTORY.append({"role": "assistant", "content": reply})
    # drop the oldest exchange(s) so history always starts with a user turn
    while len(HISTORY) > Limits.MaxHistory:
        del HISTORY[:2]
    _save_history()
    return reply
