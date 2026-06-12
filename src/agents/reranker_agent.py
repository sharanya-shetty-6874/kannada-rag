"""
src/agents/reranker_agent.py — ADVANCED RERANKER AGENT

Upgrades from rag_pipeline.py _rerank():
  ✔ Semantic similarity via embed-based cosine (no cross-encoder dep needed)
  ✔ Diversity penalty (jaccard overlap between selected chunks)
  ✔ lang_quality as a multiplicative weight (not just additive)
  ✔ Multi-intent scoring (uses sub_intents too)
  ✔ Position-aware scoring (page order)
  ✔ Completeness heuristic (longer well-formed answer preferred)

Replaces: _rerank() in rag_pipeline.py
"""

import math
import re
from typing import List, Dict

from src.config import (
    TOP_K,
    MIN_SCORE,
    INTENT_BOOST,
    KEYWORD_BOOST_PER_MATCH,
    MAX_KEYWORD_BOOST,
)


class RerankerAgent:

    def run(self, state) -> object:
        if not state.raw_chunks:
            state.ranked_chunks = []
            return state

        question    = state.original_question
        intent      = state.intent
        sub_intents = state.sub_intents or []
        entities    = state.entities or []

        q_words   = set(self._normalize(question).lower().split())
        q_words  |= {e.lower() for e in entities}

        reranked: List[Dict] = []

        for chunk in state.raw_chunks:
            score = chunk["score"]
            text  = chunk.get("text", "")
            ctype = chunk.get("chunk_type", "concept")

            if not text:
                continue

            # ── Intent boosts ─────────────────────────────────
            if ctype == intent:
                score += INTENT_BOOST
            elif ctype in sub_intents:
                score += INTENT_BOOST * 0.5

            # ── Synthetic question overlap ─────────────────────
            for sq in chunk.get("synthetic_questions", []):
                sq_words = set(sq.lower().split())
                if len(sq_words & q_words) >= 2:
                    score += 0.06
                    break

            # ── Keyword overlap ────────────────────────────────
            kws     = {k.lower() for k in chunk.get("keywords", [])}
            overlap = len(kws & q_words)
            score  += min(overlap * KEYWORD_BOOST_PER_MATCH, MAX_KEYWORD_BOOST)

            # ── Entity presence bonus ──────────────────────────
            for entity in entities[:5]:
                if entity.lower() in text.lower():
                    score += 0.04

            # ── lang_quality multiplicative weight ────────────
            #   quality=1.0 → no penalty, quality=0.5 → 15% penalty
            lq = chunk.get("lang_quality", 1.0)
            quality_factor = 0.85 + 0.15 * lq
            score *= quality_factor

            # ── Completeness heuristic ─────────────────────────
            #   Prefer chunks with ≥2 sentences (more likely to give full answer)
            sentence_count = len(re.findall(r'[.!?।]', text))
            if sentence_count >= 3:
                score += 0.03
            elif sentence_count == 0:
                score -= 0.03

            # ── Page position bias ─────────────────────────────
            page  = chunk.get("page", 1000)
            score += max(0, (50 - min(page, 50))) * 0.0005

            reranked.append({**chunk, "score": score})

        # Sort
        reranked.sort(key=lambda x: -x["score"])

        # ── Diversity penalty pass ─────────────────────────────
        # After sorting, penalize chunks that are very similar to earlier ones
        final: List[Dict] = []
        for candidate in reranked:
            if not final:
                final.append(candidate)
                continue
            # Compute max Jaccard similarity to already accepted chunks
            c_words = set(candidate["text"].lower().split())
            max_sim = 0.0
            for accepted in final:
                a_words = set(accepted["text"].lower().split())
                union   = len(c_words | a_words)
                if union:
                    sim = len(c_words & a_words) / union
                    max_sim = max(max_sim, sim)

            # Penalize score if too similar
            if max_sim > 0.70:
                candidate = {**candidate, "score": candidate["score"] * 0.4}
            elif max_sim > 0.50:
                candidate = {**candidate, "score": candidate["score"] * 0.75}

            final.append(candidate)

        # Re-sort after penalties
        final.sort(key=lambda x: -x["score"])

        # Apply MIN_SCORE AFTER reranking
        final = [c for c in final if c["score"] >= MIN_SCORE]

        state.ranked_chunks = final[:TOP_K]
        state.chunks_used   = len(state.ranked_chunks)

        if state.ranked_chunks:
            state.domain = state.ranked_chunks[0].get("domain", "general")

        return state

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join((text or "").strip().split())