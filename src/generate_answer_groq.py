"""
src/generate_answer_groq.py — KANNADA ANSWER GENERATION VIA GROQ

Changes:
  ✔ RateLimitError raised separately so retry agent won't loop on 429
  ✔ avg_quality param → tells LLM when context is OCR-noisy
  ✔ Stronger anti-hallucination prompt
  ✔ Kannada ratio check preserved
"""

import re
from groq import Groq
from dotenv import load_dotenv
from src.config import GROQ_API_KEY, GROQ_MODEL

load_dotenv()

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY missing in .env")

client = Groq(api_key=GROQ_API_KEY)


class RateLimitError(Exception):
    """Raised when Groq returns 429. Signals orchestrator to stop retrying."""
    def __init__(self, wait_seconds: int = 0):
        self.wait_seconds = wait_seconds
        super().__init__(f"Groq rate limit hit. Retry after {wait_seconds}s.")


def _kannada_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if '\u0C80' <= c <= '\u0CFF') / max(len(text), 1)


def clean_context(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _kannada_ratio(line) > 0.15 or re.search(r"[A-Za-z]{5,}", line):
            lines.append(line)
    return "\n".join(lines).strip()


def build_prompt(question: str, context: str, avg_quality: float = 1.0) -> str:
    quality_note = ""
    if avg_quality < 0.5:
        quality_note = (
            "⚠️ ಗಮನಿಸಿ: ಈ context OCR ಮೂಲಕ ಸಂಗ್ರಹಿಸಲಾಗಿದೆ. "
            "ಕೆಲವು ಅಕ್ಷರ ತಪ್ಪಾಗಿರಬಹುದು — ಎಚ್ಚರಿಕೆಯಿಂದ ಓದಿ ಉತ್ತರಿಸಿ.\n"
        )

    return f"""ನೀವು ಕನ್ನಡ RAG ಸಹಾಯಕ.

{quality_note}⚠️ ಅತ್ಯಂತ ಮುಖ್ಯ ನಿಯಮಗಳು:
1. context ಹೊರಗಿನ ಮಾಹಿತಿ ಬಳಸಬೇಡಿ
2. ಊಹಿಸಬೇಡಿ (NO guessing)
3. context ನಲ್ಲಿ ಇಲ್ಲದಿದ್ದರೆ: "ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ" ಎಂದು ಮಾತ್ರ ಹೇಳಿ
4. ಸರಳ ಮತ್ತು ಸ್ಪಷ್ಟ ಕನ್ನಡ ಬಳಸಿ
5. ಸಂಪೂರ್ಣ ವಾಕ್ಯಗಳಲ್ಲಿ ಉತ್ತರ ನೀಡಿ

-------------------- CONTEXT --------------------
{context}
-------------------------------------------------

ಪ್ರಶ್ನೆ: {question}

👉 ಕನ್ನಡದಲ್ಲಿ ಸರಿಯಾದ ಉತ್ತರ ನೀಡಿ:"""


def _parse_retry_seconds(message: str) -> int:
    """Extract wait time in seconds from Groq 429 message."""
    match = re.search(r'try again in (\d+)m(\d+(?:\.\d+)?)s', message)
    if match:
        return int(match.group(1)) * 60 + int(float(match.group(2)))
    match = re.search(r'try again in (\d+(?:\.\d+)?)s', message)
    if match:
        return int(float(match.group(1)))
    return 0


def generate_kannada_answer(question: str, context: str,
                             avg_quality: float = 1.0) -> str:
    """
    Returns a Kannada answer string.
    Raises RateLimitError if Groq 429 — caller must NOT retry.
    """
    question = (question or "").strip()
    context  = clean_context(context)

    if not question:
        return "ಪ್ರಶ್ನೆ ಖಾಲಿಯಾಗಿದೆ."
    if not context:
        return "ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ"

    prompt = build_prompt(question, context, avg_quality)

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict Kannada RAG assistant. "
                        "Answer ONLY from the provided context. "
                        "Never hallucinate. Output in Kannada."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_completion_tokens=1200,
        )
        answer = response.choices[0].message.content.strip()

    except Exception as e:
        error_str = str(e).lower()

        # ── Detect 429 rate limit ─────────────────────────────
        if "429" in error_str or "rate_limit_exceeded" in error_str:
            wait = _parse_retry_seconds(str(e))
            print(f"⏳ Groq rate limit hit. Retry after {wait}s.")
            raise RateLimitError(wait_seconds=wait)

        # ── Detect non-retriable network / connection errors ──
        non_retriable = (
            "connection error" in error_str
            or "connectionerror" in error_str
            or "name or service not known" in error_str
            or "failed to establish" in error_str
            or "network is unreachable" in error_str
            or "timed out" in error_str
            or "timeout" in error_str
            or "ssl" in error_str
        )
        if non_retriable:
            print(f"🌐 Groq connection error (non-retriable): {e}")
            raise RateLimitError(wait_seconds=0)   # reuse abort flag

        print("❌ Groq error:", e)
        return "⚠️ ಉತ್ತರ ರಚಿಸುವಾಗ ಸಮಸ್ಯೆ ಉಂಟಾಯಿತು."

    if not answer:
        return "ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ"

    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

    if _kannada_ratio(answer) < 0.1:
        return "ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ"

    return answer