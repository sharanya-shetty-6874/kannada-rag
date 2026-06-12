# tts_service.py

import os
import uuid
import re
import edge_tts

from src.config import AUDIO_DIR

# Ensure audio output folder exists
os.makedirs(AUDIO_DIR, exist_ok=True)

# Stable Kannada voice
VOICE = "kn-IN-SapnaNeural"


def clean_tts_text(text: str) -> str:
    """
    Clean text before sending to TTS.
    Helps avoid weird pauses / broken OCR noise.
    """
    if not text:
        return ""

    # Normalize spaces/newlines
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    # Remove repeated junk punctuation
    text = re.sub(r"[•▪■●◦]+", " ", text)
    text = re.sub(r"[_=]{3,}", " ", text)

    return text.strip()


async def generate_kannada_voice(text: str) -> str:
    """
    Generate Kannada MP3 from text using Edge TTS.
    Returns absolute file path.
    """
    text = clean_tts_text(text)

    if not text or len(text) < 3:
        raise ValueError("Empty or invalid TTS text")

    # Prevent extremely long TTS requests
    max_chars = 2500
    if len(text) > max_chars:
        text = text[:max_chars]

    filename = f"kannada_{uuid.uuid4().hex}.mp3"
    path = os.path.join(AUDIO_DIR, filename)

    communicate = edge_tts.Communicate(
        text=text,
        voice=VOICE
    )

    await communicate.save(path)

    # Safety check
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError("Edge TTS returned empty audio")

    return path