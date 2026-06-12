"""
backend/rag_pipeline.py — AGENT-BASED KANNADA RAG PIPELINE

This file is now a thin wrapper around src/core/orchestrator.py.

All logic has been moved to agents:
  - classify_question()  → src/agents/query_agent.py
  - rewrite_query()      → src/agents/query_agent.py
  - _hybrid_search()     → src/agents/retrieval_agent.py
  - _rerank()            → src/agents/reranker_agent.py
  - answer generation    → src/agents/answer_agent.py
  - _eval_answer()       → src/agents/evaluation_agent.py
  - retry loop           → src/agents/retry_agent.py
  - orchestration        → src/core/orchestrator.py

get_rag_answer() is kept for backward compatibility with main.py.
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.core.orchestrator import Orchestrator

# ── Orchestrator cache (one per PDF) ────────────────────────
_ORCHESTRATORS: dict = {}


def get_rag_answer(question: str, pdf_name: str) -> dict:
    """
    Entry point for main.py.
    Returns:
        {
            "answer":            str,
            "intent":            str,
            "sub_intents":       List[str],
            "entities":          List[str],
            "quality":           str,
            "confidence":        float,
            "hallucination_risk":str,
            "chunks_used":       int,
            "domain":            str,
            "retry_count":       int,
            "latency_ms":        float,
            "pipeline_log":      List[str],
        }
    """
    if not question:
        return {
            "answer":            "ಪ್ರಶ್ನೆ ಖಾಲಿಯಾಗಿದೆ.",
            "intent":            "none",
            "sub_intents":       [],
            "entities":          [],
            "quality":           "none",
            "confidence":        0.0,
            "hallucination_risk":"low",
            "chunks_used":       0,
            "domain":            "none",
            "retry_count":       0,
            "latency_ms":        0.0,
            "pipeline_log":      [],
        }

    pdf_base = os.path.splitext(pdf_name.strip())[0]

    # Cache orchestrators per PDF (avoids reloading FAISS each request)
    if pdf_base not in _ORCHESTRATORS:
        try:
            _ORCHESTRATORS[pdf_base] = Orchestrator(pdf_base)
        except Exception as e:
            print(f"❌ Orchestrator init error: {e}")
            return {
                "answer":            "🚫 ಇಂಡೆಕ್ಸ್ ಲಭ್ಯವಿಲ್ಲ.",
                "intent":            "none",
                "sub_intents":       [],
                "entities":          [],
                "quality":           "none",
                "confidence":        0.0,
                "hallucination_risk":"low",
                "chunks_used":       0,
                "domain":            "none",
                "retry_count":       0,
                "latency_ms":        0.0,
                "pipeline_log":      [str(e)],
            }

    return _ORCHESTRATORS[pdf_base].run(question)