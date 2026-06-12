"""
src/agents/retry_agent.py — RETRY / CORRECTION AGENT

NEW — governs the self-improving loop in orchestrator.py.

Strategy per retry_reason:
  low_kannada_ratio     → simplify query, increase top_k
  low_context_overlap   → expand query, shift to dense-heavy
  hallucination_risk_high → restrict context (quality filter)
  low_confidence        → increase retrieval depth + rewrite aggressively

Anti-infinite-loop: max_retries is enforced by orchestrator.
Each retry mutates state in a different direction than the last.
"""

import unicodedata
from src.config import DENSE_WEIGHT, SPARSE_WEIGHT, TOP_K


_RETRY_STRATEGIES = {
    # reason → (dense_delta, sparse_delta, top_k_delta, rewrite_mode)
    "low_kannada_ratio":      ( 0.10, -0.10,  2, "simplify"),
    "low_context_overlap":    ( 0.05, -0.05,  3, "expand"),
    "hallucination_risk_high":( 0.15, -0.15,  0, "restrict"),
    "low_confidence":         ( 0.05, -0.05,  3, "expand"),
}

_FALLBACK_REWRITES = [
    # Round 1: broaden
    lambda q: f"ಈ ಬಗ್ಗೆ ಮಾಹಿತಿ: {q}",
    # Round 2: keyword extraction only
    lambda q: " ".join(
        w for w in q.split() if len(w) >= 3
    ),
]


class RetryAgent:

    def run(self, state) -> object:
        reason  = state.retry_reason or "low_confidence"
        attempt = state.retry_count   # already incremented by orchestrator

        # ── 1. Adjust retrieval weights ───────────────────────
        d_delta, s_delta, k_delta, rw_mode = _RETRY_STRATEGIES.get(
            reason, (0.05, -0.05, 2, "expand")
        )

        # Scale delta with retry count (first retry = mild, second = aggressive)
        scale = 1.0 + (attempt - 1) * 0.5

        new_dense  = min(0.90, max(0.30, state.dense_weight  + d_delta * scale))
        new_sparse = 1.0 - new_dense

        # Store as override on state (picked up by RetrievalAgent)
        state._override_dense_w  = new_dense
        state._override_sparse_w = new_sparse

        # ── 2. Increase retrieval depth ───────────────────────
        state.top_k = min(TOP_K + k_delta * attempt, TOP_K * 3)

        # ── 3. Rewrite query ──────────────────────────────────
        state.rewritten_query = self._rewrite(
            state.original_question,
            state.intent,
            state.entities,
            rw_mode,
            attempt,
        )

        # For retrieval agent: overwrite decomposed_queries with new rewrite
        state.decomposed_queries = [state.rewritten_query]

        # ── 4. For hallucination risk: raise quality threshold ─
        if reason == "hallucination_risk_high":
            # Filter ranked_chunks to high-quality only
            if state.ranked_chunks:
                state.ranked_chunks = [
                    ch for ch in state.ranked_chunks
                    if ch.get("lang_quality", 1.0) >= 0.40
                ]

        state.log(
            f"🔁 RetryAgent strategy: dense={new_dense:.2f} sparse={new_sparse:.2f} "
            f"top_k={state.top_k} rewrite_mode={rw_mode}"
        )

        return state

    # ── Rewrite strategies ────────────────────────────────────

    @staticmethod
    def _rewrite(question: str, intent: str, entities, mode: str, attempt: int) -> str:
        q = unicodedata.normalize("NFC", question)

        if mode == "simplify":
            # Keep only the most important words
            words = q.split()
            if len(words) > 5:
                q = " ".join(words[:5])
            return f"intent: {intent} | {q}"

        elif mode == "expand":
            # Add entity emphasis + broader terms
            ent_str = " ".join(entities[:4]) if entities else ""
            broad   = "ಮಾಹಿತಿ ವಿವರ explain describe overview"
            return f"intent: {intent} | entities: {ent_str} | {broad} | {q}"

        elif mode == "restrict":
            # High-precision: only keep entities + intent
            ent_str = " ".join(entities[:3]) if entities else q[:40]
            return f"intent: {intent} | {ent_str}"

        else:
            # Generic: use fallback rewrite for the attempt number
            idx = min(attempt - 1, len(_FALLBACK_REWRITES) - 1)
            return _FALLBACK_REWRITES[idx](q)