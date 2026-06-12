"""
src/agents/evaluation_agent.py — ANSWER EVALUATION AGENT

NEW — replaces the lightweight _eval_answer() in rag_pipeline.py.

Checks:
  ✔ Kannada ratio (language faithfulness)
  ✔ Context overlap (grounding check)
  ✔ Hallucination risk scoring (answer contains facts not in context)
  ✔ Answer completeness (too short = suspect)
  ✔ Confidence score (0–1, composite)
  ✔ should_retry flag with reason
"""

import re
from typing import List


# ── Thresholds ────────────────────────────────────────────────
_MIN_KANNADA_RATIO    = 0.10   # answer must be ≥10% Kannada
_MIN_CONTEXT_OVERLAP  = 0.06   # answer words must overlap context
_MIN_ANSWER_LEN       = 20     # chars
_MIN_CONFIDENCE       = 0.40   # below this → retry


class EvaluationAgent:

    def run(self, state) -> object:
        answer  = state.answer or ""
        context = state.context or ""

        # ── 1. Kannada ratio ──────────────────────────────────
        kn_ratio = self._kannada_ratio(answer)
        state.kannada_ratio = kn_ratio

        # ── 2. Context overlap ────────────────────────────────
        overlap = self._context_overlap(answer, context)
        state.context_overlap = overlap

        # ── 3. Hallucination risk ─────────────────────────────
        hallucination_risk = self._hallucination_risk(answer, context)
        state.hallucination_risk = hallucination_risk

        # ── 4. Completeness ───────────────────────────────────
        completeness = self._completeness_score(answer)

        # ── 5. Composite confidence ───────────────────────────
        # Weighted composite
        confidence = (
            0.35 * min(kn_ratio / 0.35, 1.0)          # Kannada presence
          + 0.30 * min(overlap / 0.15, 1.0)            # grounding
          + 0.20 * completeness                         # not too short
          + 0.15 * (1.0 if hallucination_risk == "low" else
                    0.4 if hallucination_risk == "medium" else 0.0)
        )
        state.confidence = round(min(confidence, 1.0), 3)

        # ── 6. Quality label ──────────────────────────────────
        if state.confidence >= 0.60:
            state.eval_quality = "good"
        elif state.confidence >= 0.35:
            state.eval_quality = "low"
        else:
            state.eval_quality = "no_match"

        # ── 7. Retry decision ─────────────────────────────────
        state.should_retry = False
        state.retry_reason = ""

        if kn_ratio < _MIN_KANNADA_RATIO:
            state.should_retry = True
            state.retry_reason = "low_kannada_ratio"

        elif overlap < _MIN_CONTEXT_OVERLAP and state.chunks_used > 0:
            state.should_retry = True
            state.retry_reason = "low_context_overlap"

        elif hallucination_risk == "high":
            state.should_retry = True
            state.retry_reason = "hallucination_risk_high"

        elif state.confidence < _MIN_CONFIDENCE and state.chunks_used > 0:
            state.should_retry = True
            state.retry_reason = "low_confidence"

        # Specific "not found" patterns — don't retry, these are valid
        _NOT_FOUND = [
            "ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ",
            "ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ",
        ]
        if any(nf in answer for nf in _NOT_FOUND):
            state.should_retry  = False
            state.eval_quality  = "no_match"
            state.confidence    = 0.0

        return state

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _kannada_ratio(text: str) -> float:
        if not text:
            return 0.0
        return sum(1 for c in text if '\u0C80' <= c <= '\u0CFF') / max(len(text), 1)

    @staticmethod
    def _context_overlap(answer: str, context: str) -> float:
        if not answer or not context:
            return 0.0
        ans_words = set(answer.lower().split())
        ctx_words = set(context.lower().split())
        if not ans_words:
            return 0.0
        return len(ans_words & ctx_words) / len(ans_words)

    @staticmethod
    def _hallucination_risk(answer: str, context: str) -> str:
        """
        Heuristic hallucination detection:
        - Extract number-like and proper noun tokens from answer
        - Check if they appear in context
        - High ratio of answer-only tokens → high risk
        """
        if not answer or not context:
            return "low"

        ctx_lower = context.lower()

        # Numbers in the answer
        numbers_in_answer = re.findall(r'\b\d+[\d,.]*\b', answer)
        ungrounded_numbers = [
            n for n in numbers_in_answer
            if n not in ctx_lower
        ]

        # English proper-noun-like tokens in answer
        en_tokens = re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', answer)
        ungrounded_en = [t for t in en_tokens if t.lower() not in ctx_lower]

        total_suspects = len(ungrounded_numbers) + len(ungrounded_en)

        if total_suspects >= 5:
            return "high"
        elif total_suspects >= 2:
            return "medium"
        return "low"

    @staticmethod
    def _completeness_score(answer: str) -> float:
        if not answer:
            return 0.0
        length = len(answer)
        if length < _MIN_ANSWER_LEN:
            return 0.1
        # Penalize very short answers (likely "not found" responses)
        if length < 60:
            return 0.4
        # Reward answers with multiple sentences
        sentences = len(re.findall(r'[.!?।]', answer))
        sentence_bonus = min(sentences * 0.1, 0.3)
        return min(0.7 + sentence_bonus, 1.0)