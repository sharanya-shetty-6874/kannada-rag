"""
src/compare/compare_orchestrator.py

Runs one configuration combo (embed_model × retrieval_mode × llm_model)
through the full pipeline and returns structured results.

CHANGES v2:
  ✔ Added gemma2-9b-it, mixtral-8x7b-32768 to available LLMs
  ✔ Added LLM_COMPARISON_COMBOS (KN-BERT Hybrid × all 4 LLMs) for paper-style table
  ✔ Added NO_RAG_MODELS list for baseline evaluation
  ✔ run_no_rag() — calls LLM with empty context for ungrounded baseline
  ✔ All original combos preserved (backward compatible)
"""

import os
import time
import re
import numpy as np
from typing import Dict, List, Optional

from src.config import (
    EMBED_MODEL,
    MAIN_MODELS_DIR,
    BACKEND_MODELS_DIR,
    INDEX_DIR,
    find_local_model,
)
from src.build_faiss_index import (
    E5_MODEL,
    E5_SUBDIR,
    get_embedder    as _get_knbert_embedder,
    get_e5_embedder as _get_e5_embedder_singleton,
)


# ══════════════════════════════════════════════════════════════
#  COMBO CONFIG
# ══════════════════════════════════════════════════════════════

class ComboConfig:
    def __init__(self, embed_model: str, retrieval: str, llm_model: str):
        self.embed_model = embed_model
        self.retrieval   = retrieval
        self.llm_model   = llm_model

    @property
    def embed_model_id(self) -> str:
        return EMBED_MODEL if self.embed_model == "kannada-bert" else E5_MODEL

    @property
    def is_e5(self) -> bool:
        return self.embed_model == "e5"

    @property
    def is_small_llm(self) -> bool:
        name = self.llm_model.lower()
        return any(tag in name for tag in ("8b", "9b", "7b", "20b", "scout", "qwen3-32b"))

    @property
    def label(self) -> str:
        return f"{self.embed_model} | {self.retrieval} | {self.llm_model}"

    @property
    def short_label(self) -> str:
        """Short label for table columns (LLM name only)."""
        return self.llm_model.split("-versatile")[0].split("-instant")[0]

    def to_dict(self) -> dict:
        return {
            "embed_model": self.embed_model,
            "retrieval":   self.retrieval,
            "llm_model":   self.llm_model,
            "label":       self.label,
        }


# ══════════════════════════════════════════════════════════════
#  TOKEN BUDGETS  (Groq free tier limits)
# ══════════════════════════════════════════════════════════════
#   llama-3.1-8b-instant  → 6,000  TPM → keep prompt < 4,500 tokens
#   llama-3.3-70b-versatile → 12,000 TPM → up to 9,000 tokens
#   gemma2-9b-it          → 15,000 TPM → safe at 3,000 chars
#   mixtral-8x7b-32768    → 5,000  TPM → keep tight at 1,800 chars

_LLM_LIMITS = {
    "small": {"max_context_chars": 1_800, "max_completion_tokens": 400},
    "large": {"max_context_chars": 3_000, "max_completion_tokens": 800},
}


# ══════════════════════════════════════════════════════════════
#  EMBEDDER GETTER
# ══════════════════════════════════════════════════════════════

def _get_embedder(config: ComboConfig):
    if config.is_e5:
        return _get_e5_embedder_singleton()
    return _get_knbert_embedder()


# ══════════════════════════════════════════════════════════════
#  INDEX LOADER
# ══════════════════════════════════════════════════════════════

def _load_index_for_combo(pdf_name: str, config: ComboConfig):
    import faiss
    import pandas as pd
    from src.bm25_store import BM25Store

    base_path = os.path.join(INDEX_DIR, pdf_name)
    path      = os.path.join(base_path, E5_SUBDIR) if config.is_e5 else base_path
    faiss_path   = os.path.join(path, "index.faiss")
    parquet_path = os.path.join(path, "chunks.parquet")

    if not os.path.exists(faiss_path):
        if config.is_e5:
            raise FileNotFoundError(
                f"E5 index not found for '{pdf_name}'. "
                f"Fix: POST /rebuild_e5_index/{pdf_name}"
            )
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

    faiss_index = faiss.read_index(faiss_path)
    df          = pd.read_parquet(parquet_path)
    bm25        = None

    try:
        from src.bm25_store import BM25Store
        b = BM25Store(path)
        b.load()
        bm25 = b
    except FileNotFoundError:
        pass

    return faiss_index, bm25, df


# ══════════════════════════════════════════════════════════════
#  RETRIEVAL HELPERS
# ══════════════════════════════════════════════════════════════

def _embed_query(text: str, config: ComboConfig) -> np.ndarray:
    model = _get_embedder(config)
    if config.is_e5:
        text = f"query: {text}"
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)


def _dense_search(faiss_index, df, query_emb, top_k: int) -> List[Dict]:
    scores_arr, ids_arr = faiss_index.search(query_emb, top_k * 3)
    results = []
    for sc, idx in zip(scores_arr[0], ids_arr[0]):
        if idx == -1:
            continue
        row  = df.iloc[int(idx)].to_dict()
        text = row.get("text", "")
        if not text:
            continue
        results.append({
            "idx":          int(idx),
            "text":         text,
            "score":        float(sc),
            "page":         int(row.get("page", 0)),
            "chunk_type":   row.get("chunk_type", "concept"),
            "lang_quality": float(row.get("lang_quality", 1.0)),
            "domain":       row.get("domain", "general"),
            "keywords":     row.get("keywords", []),
        })
    results.sort(key=lambda x: -x["score"])
    return results[:top_k]


def _hybrid_search(faiss_index, bm25, df, query_emb, raw_query, top_k,
                   dense_w=0.65, sparse_w=0.35) -> List[Dict]:
    scores_arr, ids_arr = faiss_index.search(query_emb, top_k * 4)
    dense = {int(idx): float(sc)
             for sc, idx in zip(scores_arr[0], ids_arr[0]) if idx != -1}

    sparse: Dict[int, float] = {}
    if bm25 is not None:
        raw_bm25 = bm25.search(raw_query, top_k * 4)
        max_bm25 = max((s for _, s in raw_bm25), default=1e-9)
        sparse   = {idx: sc / max(max_bm25, 1e-9) for idx, sc in raw_bm25}

    all_ids  = set(dense) | set(sparse)
    combined = {
        idx: dense_w * dense.get(idx, 0.0) + sparse_w * sparse.get(idx, 0.0)
        for idx in all_ids
    }

    results = []
    for idx, score in sorted(combined.items(), key=lambda x: -x[1])[:top_k * 2]:
        if idx < 0 or idx >= len(df):
            continue
        row  = df.iloc[int(idx)].to_dict()
        text = row.get("text", "")
        if not text:
            continue
        results.append({
            "idx":          int(idx),
            "text":         text,
            "score":        float(score),
            "page":         int(row.get("page", 0)),
            "chunk_type":   row.get("chunk_type", "concept"),
            "lang_quality": float(row.get("lang_quality", 1.0)),
            "domain":       row.get("domain", "general"),
            "keywords":     row.get("keywords", []),
        })
    results.sort(key=lambda x: -x["score"])
    return results[:top_k]


def _build_context(chunks: List[Dict], max_chars: int) -> str:
    parts, total = [], 0
    for ch in chunks:
        block = f"[ಪುಟ {ch.get('page', '?')}]\n{ch['text']}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


# ══════════════════════════════════════════════════════════════
#  RATE LIMIT HELPERS
# ══════════════════════════════════════════════════════════════

def _parse_wait_seconds(error_str: str) -> Optional[int]:
    m = re.search(r'try again in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s', error_str)
    if m:
        return int(int(m.group(1) or 0) * 60 + float(m.group(2) or 0)) + 1
    return None

def _is_tpd_error(err: str) -> bool:
    return "tokens per day" in err.lower() or "TPD" in err

def _is_tpm_error(err: str) -> bool:
    return "tokens per minute" in err.lower() or "TPM" in err


# ══════════════════════════════════════════════════════════════
#  LLM ANSWER GENERATION
# ══════════════════════════════════════════════════════════════

def _generate_answer(question: str, context: str, llm_model: str,
                     max_completion_tokens: int) -> str:
    from groq import Groq
    from src.config import GROQ_API_KEY

    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY missing"

    client = Groq(api_key=GROQ_API_KEY)

    if context.strip():
        prompt = (
            f"ನೀವು ಕನ್ನಡ RAG ಸಹಾಯಕ.\n"
            f"⚠️ ಕೇವಲ context ಆಧಾರದ ಮೇಲೆ ಕನ್ನಡದಲ್ಲಿ ಉತ್ತರಿಸಿ.\n\n"
            f"-------------------- CONTEXT --------------------\n"
            f"{context}\n"
            f"-------------------------------------------------\n\n"
            f"ಪ್ರಶ್ನೆ: {question}\n\n"
            f"👉 ಕನ್ನಡ ಉತ್ತರ:"
        )
    else:
        # No-RAG mode: answer from parametric knowledge
        prompt = (
            f"ನೀವು ಒಬ್ಬ ಸಹಾಯಕ ಶಿಕ್ಷಕ.\n"
            f"ಕೆಳಗಿನ ಪ್ರಶ್ನೆಗೆ ಕನ್ನಡದಲ್ಲಿ 2-3 ವಾಕ್ಯಗಳಲ್ಲಿ ಉತ್ತರಿಸಿ.\n\n"
            f"ಪ್ರಶ್ನೆ: {question}\n\n"
            f"👉 ಕನ್ನಡ ಉತ್ತರ:"
        )

    try:
        resp = client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system",
                 "content": "Answer in Kannada only. Be concise."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_completion_tokens=max_completion_tokens,
        )
        return resp.choices[0].message.content.strip()

    except Exception as e:
        err = str(e) if str(e).strip() else repr(e)
        if _is_tpd_error(err):
            wait = _parse_wait_seconds(err)
            ts   = f"{wait//60}m {wait%60}s" if wait else "~24h"
            return f"⏳ Daily limit for '{llm_model}'. Retry after {ts}."
        if _is_tpm_error(err):
            return f"⏳ Per-minute limit for '{llm_model}'. Wait ~1 min."
        return f"⚠️ LLM error: {err}"


# ══════════════════════════════════════════════════════════════
#  INTERNAL METRICS (used by compare panel)
# ══════════════════════════════════════════════════════════════

def _kannada_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if '\u0C80' <= c <= '\u0CFF') / max(len(text), 1)


def _confidence_score(answer: str, context: str, chunks: List[Dict]) -> float:
    kn_r    = _kannada_ratio(answer)
    ans_w   = set(answer.lower().split())
    ctx_w   = set(context.lower().split())
    overlap = len(ans_w & ctx_w) / max(len(ans_w), 1)
    length  = min(len(answer) / 200, 1.0)
    return round(min(0.40 * min(kn_r/0.35, 1.0) + 0.35 * min(overlap/0.15, 1.0) + 0.25 * length, 1.0), 3)


def _hallucination_risk(answer: str, context: str) -> str:
    if not answer or not context:
        return "low"
    ctx_lower = context.lower()
    suspects  = sum(1 for n in re.findall(r'\b\d+[\d,.]*\b', answer) if n not in ctx_lower)
    suspects += sum(1 for t in re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', answer) if t.lower() not in ctx_lower)
    return "high" if suspects >= 5 else "medium" if suspects >= 2 else "low"


# ══════════════════════════════════════════════════════════════
#  MAIN RUNNER — RAG
# ══════════════════════════════════════════════════════════════

def run_combo(question: str, pdf_name: str, config: ComboConfig, top_k: int = 5) -> Dict:
    t0     = time.time()
    limits = _LLM_LIMITS["small"] if config.is_small_llm else _LLM_LIMITS["large"]

    try:
        faiss_index, bm25, df = _load_index_for_combo(pdf_name, config)
        query_emb             = _embed_query(question, config)

        if config.retrieval == "hybrid" and bm25 is not None:
            chunks = _hybrid_search(faiss_index, bm25, df, query_emb, question, top_k)
        else:
            chunks = _dense_search(faiss_index, df, query_emb, top_k)

        if not chunks:
            return {
                "config": config.to_dict(),
                "answer": "ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ",
                "confidence": 0.0, "hallucination_risk": "low",
                "chunks_used": 0, "kannada_ratio": 0.0,
                "latency_ms": round((time.time() - t0) * 1000, 1),
                "top_chunk_score": 0.0, "retrieved_texts": [], "error": None,
            }

        context = _build_context(chunks, max_chars=limits["max_context_chars"])
        answer  = _generate_answer(question, context, config.llm_model, limits["max_completion_tokens"])

        return {
            "config":             config.to_dict(),
            "answer":             answer,
            "confidence":         _confidence_score(answer, context, chunks),
            "hallucination_risk": _hallucination_risk(answer, context),
            "chunks_used":        len(chunks),
            "kannada_ratio":      round(_kannada_ratio(answer), 3),
            "latency_ms":         round((time.time() - t0) * 1000, 1),
            "top_chunk_score":    round(chunks[0]["score"], 3) if chunks else 0.0,
            "retrieved_texts":    [c["text"] for c in chunks],   # for Recall@k
            "error":              None,
        }

    except Exception as e:
        err = str(e) if str(e).strip() else repr(e)
        return {
            "config": config.to_dict(),
            "answer": f"⚠️ Error: {err}",
            "confidence": 0.0, "hallucination_risk": "low",
            "chunks_used": 0, "kannada_ratio": 0.0,
            "latency_ms": round((time.time() - t0) * 1000, 1),
            "top_chunk_score": 0.0, "retrieved_texts": [], "error": err,
        }


# ══════════════════════════════════════════════════════════════
#  NO-RAG RUNNER — ungrounded baseline
# ══════════════════════════════════════════════════════════════

def run_no_rag(question: str, llm_model: str = "llama-3.3-70b-versatile") -> Dict:
    """
    Call LLM with NO retrieval context.
    Used as ungrounded baseline to quantify RAG improvement.
    Mirrors the other team's Table 9 'Without RAG' column.
    """
    limits = _LLM_LIMITS["small"] if any(t in llm_model.lower() for t in ("8b","9b","7b")) else _LLM_LIMITS["large"]
    t0     = time.time()

    answer = _generate_answer(question, "", llm_model, limits["max_completion_tokens"])

    return {
        "llm_model":          llm_model,
        "answer":             answer,
        "confidence":         0.0,   # no context to score against
        "hallucination_risk": "unknown",
        "chunks_used":        0,
        "kannada_ratio":      round(_kannada_ratio(answer), 3),
        "latency_ms":         round((time.time() - t0) * 1000, 1),
        "retrieved_texts":    [],
        "error":              None if not answer.startswith(("⚠️","⏳")) else answer,
    }


# ══════════════════════════════════════════════════════════════
#  COMBO LISTS
# ══════════════════════════════════════════════════════════════

# Original 8 combos (2 embed × 2 retrieval × 2 LLMs)
ALL_COMBOS = [
    ComboConfig("kannada-bert", "hybrid", "llama-3.3-70b-versatile"),
    ComboConfig("kannada-bert", "hybrid", "llama-3.1-8b-instant"),
    ComboConfig("kannada-bert", "dense",  "llama-3.3-70b-versatile"),
    ComboConfig("kannada-bert", "dense",  "llama-3.1-8b-instant"),
    ComboConfig("e5",           "hybrid", "llama-3.3-70b-versatile"),
    ComboConfig("e5",           "hybrid", "llama-3.1-8b-instant"),
    ComboConfig("e5",           "dense",  "llama-3.3-70b-versatile"),
    ComboConfig("e5",           "dense",  "llama-3.1-8b-instant"),
]

# Extended: Best embedding/retrieval (KN-BERT Hybrid) × 4 LLMs
# Mirrors the other team's Table 7 "model comparison" approach
LLM_COMPARISON_COMBOS = [
    ComboConfig("kannada-bert", "hybrid", "llama-3.1-8b-instant"),
    ComboConfig("kannada-bert", "hybrid", "meta-llama/llama-4-scout-17b-16e-instruct"),
    ComboConfig("kannada-bert", "hybrid", "qwen/qwen3-32b"),
    ComboConfig("kannada-bert", "hybrid", "llama-3.3-70b-versatile"),
]

# No-RAG models (used for Table 9 comparison)
NO_RAG_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]