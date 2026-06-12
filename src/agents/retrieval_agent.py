"""
src/agents/retrieval_agent.py — ADAPTIVE HYBRID RETRIEVAL AGENT

Upgrades from rag_pipeline.py:
  ✔ Dynamic dense/sparse weights based on query intent
  ✔ Semantic filtering (removes irrelevant chunks before reranking)
  ✔ Diversity selection (MMR - Maximal Marginal Relevance) to avoid duplicates
  ✔ Multi-query retrieval: if decomposed_queries > 1, merges results
  ✔ Embedding cache (no re-encoding same query twice)
  ✔ FAISS warm loading (uses lru_cache from build_faiss_index)

Replaces:
  - _hybrid_search() in rag_pipeline.py
  - _load_index() in rag_pipeline.py (delegates to build_faiss_index)
"""

import os
import numpy as np
from functools import lru_cache
from typing import Dict, List, Tuple

from src.config import (
    INDEX_DIR, TOP_K, MIN_SCORE,
    DENSE_WEIGHT, SPARSE_WEIGHT,
)
from src.bm25_store import BM25Store


# ── Per-intent adaptive weights ──────────────────────────────
#   (dense, sparse)
_INTENT_WEIGHTS: Dict[str, Tuple[float, float]] = {
    "definition":  (0.75, 0.25),   # dense-heavy: semantics matter
    "procedure":   (0.65, 0.35),
    "benefits":    (0.65, 0.35),
    "comparison":  (0.60, 0.40),   # balanced: both match keywords & semantics
    "list":        (0.50, 0.50),   # balanced
    "example":     (0.70, 0.30),
    "formula":     (0.45, 0.55),   # sparse-heavy: formulae are keyword-dense
    "cause":       (0.65, 0.35),
    "story":       (0.70, 0.30),
    "explanation": (0.70, 0.30),
    "concept":     (0.65, 0.35),
    "negation":    (0.55, 0.45),
    "quantity":    (0.50, 0.55),   # sparse-heavy: numbers/counts
}

_EMBED_CACHE: Dict[str, np.ndarray] = {}


class RetrievalAgent:

    def __init__(self, pdf_name: str):
        self.pdf_name = os.path.splitext(pdf_name.strip())[0]
        self._faiss_index = None
        self._bm25        = None
        self._df          = None
        self._loaded      = False

    # ── Lazy index loading ────────────────────────────────────

    def _ensure_loaded(self):
        if self._loaded:
            return
        from src.build_faiss_index import _load_index
        self._faiss_index, self._bm25, self._df = _load_index(self.pdf_name)
        self._loaded = True

    # ── Embedding with cache ──────────────────────────────────

    @staticmethod
    def _embed(text: str) -> np.ndarray:
        if text in _EMBED_CACHE:
            return _EMBED_CACHE[text]
        from src.build_faiss_index import get_embedder
        model = get_embedder()
        emb = model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        _EMBED_CACHE[text] = emb
        if len(_EMBED_CACHE) > 256:          # cap cache size
            oldest = next(iter(_EMBED_CACHE))
            del _EMBED_CACHE[oldest]
        return emb

    # ── Core hybrid search ────────────────────────────────────

    def _single_hybrid_search(
        self, embed_text: str, raw_query: str,
        dense_w: float, sparse_w: float, top_k: int
    ) -> Dict[int, float]:

        q_emb = self._embed(embed_text)

        # Dense (FAISS)
        scores_arr, ids_arr = self._faiss_index.search(q_emb, top_k * 4)
        dense: Dict[int, float] = {
            int(idx): float(sc)
            for sc, idx in zip(scores_arr[0], ids_arr[0])
            if idx != -1
        }

        # Sparse (BM25)
        sparse: Dict[int, float] = {}
        if self._bm25 is not None:
            raw_bm25 = self._bm25.search(raw_query, top_k * 4)
            max_bm25 = max((s for _, s in raw_bm25), default=1e-9)
            max_bm25 = max(max_bm25, 1e-9)
            sparse = {idx: sc / max_bm25 for idx, sc in raw_bm25}

        # Combine
        all_ids  = set(dense) | set(sparse)
        combined = {
            idx: dense_w * dense.get(idx, 0.0) + sparse_w * sparse.get(idx, 0.0)
            for idx in all_ids
        }
        return combined

    # ── Semantic filter ───────────────────────────────────────

    def _semantic_filter(
        self, combined: Dict[int, float], threshold: float
    ) -> Dict[int, float]:
        """Remove chunks below adaptive threshold."""
        if not combined:
            return combined
        max_score = max(combined.values())
        adaptive_threshold = max(threshold, max_score * 0.35)
        return {idx: sc for idx, sc in combined.items() if sc >= adaptive_threshold}

    # ── MMR diversity selection ───────────────────────────────

    def _mmr_select(
        self, candidates: List[Dict], top_k: int, lambda_: float = 0.7
    ) -> List[Dict]:
        """
        Maximal Marginal Relevance: select diverse, relevant chunks.
        lambda_ = relevance weight (1.0 = pure relevance, 0.0 = pure diversity)
        """
        if len(candidates) <= top_k:
            return candidates

        # Use pre-computed embeddings if available, else use score as proxy
        selected = []
        remaining = list(candidates)

        # Pick the best chunk first
        selected.append(remaining.pop(0))

        while len(selected) < top_k and remaining:
            best_idx   = -1
            best_score = -999.0

            for i, cand in enumerate(remaining):
                relevance = cand["score"]

                # Diversity penalty: max cosine sim to already selected chunks
                # We approximate similarity via text overlap (no embedding cost)
                max_sim = 0.0
                cand_words = set(cand["text"].lower().split())
                for sel in selected:
                    sel_words = set(sel["text"].lower().split())
                    if cand_words | sel_words:
                        sim = len(cand_words & sel_words) / len(cand_words | sel_words)
                        max_sim = max(max_sim, sim)

                mmr_score = lambda_ * relevance - (1 - lambda_) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx   = i

            if best_idx >= 0:
                selected.append(remaining.pop(best_idx))

        return selected

    # ── Main run ─────────────────────────────────────────────

    def run(self, state) -> object:
        self._ensure_loaded()

        intent     = state.intent
        dense_w, sparse_w = _INTENT_WEIGHTS.get(intent, (DENSE_WEIGHT, SPARSE_WEIGHT))

        # Override weights if retry has adjusted them
        if hasattr(state, "_override_dense_w"):
            dense_w  = state._override_dense_w
            sparse_w = state._override_sparse_w

        state.dense_weight  = dense_w
        state.sparse_weight = sparse_w

        top_k = state.top_k

        # Multi-query retrieval for compound/complex queries
        queries_to_run = state.decomposed_queries or [state.original_question]

        merged: Dict[int, float] = {}

        for sub_q in queries_to_run:
            # Build embed text for this sub-query
            from src.agents.query_agent import _QUERY_EXPANSION, _SYNONYMS
            import unicodedata
            q_nfc = unicodedata.normalize("NFC", sub_q)
            expansion = _QUERY_EXPANSION.get(intent, "")
            embed_text = f"intent: {intent} | expansion: {expansion} | {q_nfc}"

            combined = self._single_hybrid_search(
                embed_text, sub_q, dense_w, sparse_w, top_k
            )
            # Merge with max-score fusion
            for idx, sc in combined.items():
                merged[idx] = max(merged.get(idx, 0.0), sc)

        # Semantic filter
        filtered = self._semantic_filter(merged, MIN_SCORE * 0.7)

        if not filtered:
            state.raw_chunks = []
            return state

        # Build chunk dicts
        df = self._df
        chunk_list: List[Dict] = []
        for idx, score in filtered.items():
            if idx < 0 or idx >= len(df):
                continue
            row  = df.iloc[int(idx)].to_dict()
            text = row.get("text", "")
            if not text:
                continue
            chunk_list.append({
                "idx":          idx,
                "text":         text,
                "score":        score,
                "page":         int(row.get("page", 1000)),
                "chunk_type":   row.get("chunk_type", "concept"),
                "lang_quality": float(row.get("lang_quality", 1.0)),
                "domain":       row.get("domain", "general"),
                "difficulty":   row.get("difficulty", "basic"),
                "keywords":     row.get("keywords", []),
                "synthetic_questions": row.get("synthetic_questions", []),
            })

        # Sort by score
        chunk_list.sort(key=lambda x: -x["score"])

        # MMR diversity selection
        state.raw_chunks = self._mmr_select(chunk_list, top_k * 2, lambda_=0.72)

        return state