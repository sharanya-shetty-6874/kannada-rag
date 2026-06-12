"""
src/agents/answer_agent.py — ANSWER GENERATION AGENT  (Groq / Llama edition)
"""

from src.config import MAX_CONTEXT_CHARS
from src.generate_answer_groq import generate_kannada_answer, RateLimitError
from typing import List, Dict


class AnswerAgent:

    def run(self, state) -> object:
        if not state.ranked_chunks:
            state.answer  = "ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ"
            state.context = ""
            return state

        state.context     = self._build_context(state.ranked_chunks)
        state.avg_quality = sum(
            ch.get("lang_quality", 1.0) for ch in state.ranked_chunks
        ) / len(state.ranked_chunks)

        if not state.context.strip():
            state.answer = "ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ"
            return state

        try:
            state.answer = generate_kannada_answer(
                state.original_question,
                state.context,
                state.avg_quality,
            )

        except RateLimitError as e:
            if e.wait_seconds > 0:
                wait_min = e.wait_seconds // 60 + 1
                state.answer = (
                    f"⏳ Groq rate limit hit. Please wait ~{wait_min} minutes and try again."
                )
            else:
                state.answer = (
                    "⚠️ Groq connection error. Please check your internet / API key."
                )
            state.abort_retries = True
            state.eval_quality  = "error"
            state.confidence    = 0.0

        return state

    @staticmethod
    def _build_context(chunks: List[Dict]) -> str:
        parts = []
        total = 0
        for ch in chunks:
            header = (
                f"[ಪುಟ {ch.get('page', '?')} | "
                f"{ch.get('chunk_type', '?')} | "
                f"ಗುಣಮಟ್ಟ:{ch.get('lang_quality', 1.0):.2f}]"
            )
            block = f"{header}\n{ch['text']}"
            if total + len(block) > MAX_CONTEXT_CHARS:
                break
            parts.append(block)
            total += len(block)
        return "\n\n".join(parts)