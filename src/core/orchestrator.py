"""
src/core/orchestrator.py — AGENT ORCHESTRATOR FOR KANNADA RAG

Manages the full pipeline:
  User Query → QueryAgent → RetrievalAgent → RerankerAgent
             → AnswerAgent → EvaluationAgent → RetryAgent (if needed)

Each agent is a class with a .run() method.
The orchestrator passes a shared PipelineState dict between agents.
"""

import time
import logging
from typing import Optional
from dataclasses import dataclass, field
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# ── Shared pipeline state ────────────────────────────────────

@dataclass
class PipelineState:
    # Input
    original_question:  str = ""
    pdf_name:           str = ""

    # Query understanding
    intent:             str = "concept"
    sub_intents:        List[str] = field(default_factory=list)
    entities:           List[str] = field(default_factory=list)
    decomposed_queries: List[str] = field(default_factory=list)
    rewritten_query:    str = ""
    ambiguity_score:    float = 0.0
    query_complexity:   str = "simple"   # simple | compound | complex

    # Retrieval
    dense_weight:       float = 0.65
    sparse_weight:      float = 0.35
    top_k:              int = 5
    raw_chunks:         List[Dict] = field(default_factory=list)

    # Reranking
    ranked_chunks:      List[Dict] = field(default_factory=list)

    # Answer
    context:            str = ""
    answer:             str = ""
    avg_quality:        float = 1.0

    # Evaluation
    confidence:         float = 0.0
    kannada_ratio:      float = 0.0
    context_overlap:    float = 0.0
    hallucination_risk: str = "low"     # low | medium | high
    eval_quality:       str = "good"    # good | low | no_match
    should_retry:       bool = False

    # Retry tracking
    retry_count:        int = 0
    max_retries:        int = 2
    retry_reason:       str = ""
    abort_retries:      bool = False  # set True on rate-limit / non-retriable errors

    # Metadata
    domain:             str = "general"
    chunks_used:        int = 0
    latency_ms:         float = 0.0
    pipeline_log:       List[str] = field(default_factory=list)

    def log(self, msg: str):
        logger.info(msg)
        self.pipeline_log.append(msg)


# ── Orchestrator ─────────────────────────────────────────────

class Orchestrator:

    def __init__(self, pdf_name: str):
        from src.agents.query_agent     import QueryAgent
        from src.agents.retrieval_agent import RetrievalAgent
        from src.agents.reranker_agent  import RerankerAgent
        from src.agents.answer_agent    import AnswerAgent
        from src.agents.evaluation_agent import EvaluationAgent
        from src.agents.retry_agent     import RetryAgent

        self.pdf_name   = pdf_name
        self.query_ag   = QueryAgent()
        self.retrieval_ag = RetrievalAgent(pdf_name)
        self.reranker_ag  = RerankerAgent()
        self.answer_ag    = AnswerAgent()
        self.eval_ag      = EvaluationAgent()
        self.retry_ag     = RetryAgent()

    def run(self, question: str) -> dict:
        t0 = time.time()

        state = PipelineState(
            original_question=question,
            pdf_name=self.pdf_name,
        )

        state.log("🚀 Orchestrator: pipeline start")

        # ── Step 1: Query Understanding ──────────────────────
        state = self.query_ag.run(state)
        state.log(f"🧠 QueryAgent: intent={state.intent} complexity={state.query_complexity}")

        # ── Step 2: Retrieval (with adaptive weights) ────────
        state = self.retrieval_ag.run(state)
        state.log(f"🔍 RetrievalAgent: {len(state.raw_chunks)} chunks retrieved")

        if not state.raw_chunks:
            state.answer = "ಪ್ರಶ್ನೆಗೆ ಸಂಬಂಧಿಸಿದ ಮಾಹಿತಿ ಕಂಡುಬಂದಿಲ್ಲ"
            state.eval_quality = "no_match"
            return self._final_response(state, t0)

        # ── Step 3: Reranking ────────────────────────────────
        state = self.reranker_ag.run(state)
        state.log(f"📊 RerankerAgent: top score={state.ranked_chunks[0]['score']:.3f}" if state.ranked_chunks else "⚠️ No ranked chunks")

        # ── Step 4: Answer Generation ────────────────────────
        state = self.answer_ag.run(state)
        state.log(f"💬 AnswerAgent: answer_len={len(state.answer)}")

        # ── Step 5: Evaluation + Retry Loop ──────────────────
        state = self.eval_ag.run(state)
        state.log(f"🧪 EvaluationAgent: confidence={state.confidence:.2f} quality={state.eval_quality}")

        while state.should_retry and state.retry_count < state.max_retries and not state.abort_retries:
            state.retry_count += 1
            state.log(f"🔁 RetryAgent: attempt {state.retry_count} reason={state.retry_reason}")

            state = self.retry_ag.run(state)        # rewrites query, adjusts weights
            state = self.retrieval_ag.run(state)    # re-retrieves
            state = self.reranker_ag.run(state)     # re-ranks
            state = self.answer_ag.run(state)       # re-answers
            state = self.eval_ag.run(state)         # re-evaluates

            state.log(f"   → after retry {state.retry_count}: confidence={state.confidence:.2f}")

        return self._final_response(state, t0)

    def _final_response(self, state: PipelineState, t0: float) -> dict:
        state.latency_ms = round((time.time() - t0) * 1000, 1)
        state.log(f"✅ Done in {state.latency_ms}ms")

        return {
            "answer":          state.answer,
            "intent":          state.intent,
            "sub_intents":     state.sub_intents,
            "entities":        state.entities,
            "quality":         state.eval_quality,
            "confidence":      round(state.confidence, 3),
            "hallucination_risk": state.hallucination_risk,
            "chunks_used":     state.chunks_used,
            "domain":          state.domain,
            "retry_count":     state.retry_count,
            "latency_ms":      state.latency_ms,
            "pipeline_log":    state.pipeline_log,
        }