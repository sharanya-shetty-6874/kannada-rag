"""
backend/compare_endpoint.py

Add this to main.py to get the /compare endpoint.
It runs selected combos SEQUENTIALLY and returns structured
results for the frontend ComparePanel.

HOW TO INTEGRATE INTO main.py:
1. Copy this file into your backend/ folder
2. Add this import at the top of main.py:
       from compare_endpoint import router as compare_router
3. Add this line after app = FastAPI(...):
       app.include_router(compare_router)
That's it — existing code is untouched.
"""

import asyncio
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.compare.compare_orchestrator import ComboConfig, run_combo, ALL_COMBOS

router = APIRouter()


# ── Request/response models ───────────────────────────────────

class ComboRequest(BaseModel):
    embed_model: str   # "kannada-bert" | "e5"
    retrieval:   str   # "hybrid" | "dense"
    llm_model:   str   # "llama-3.3-70b-versatile" | "llama-3.1-8b-instant"


class CompareRequest(BaseModel):
    question: str
    pdf:      str
    combos:   Optional[List[ComboRequest]] = None   # None = run all 8


# ── /compare endpoint ─────────────────────────────────────────

@router.post("/compare")
async def compare(req: CompareRequest):
    """
    Runs selected combos SEQUENTIALLY to avoid thread race conditions
    when loading embedding models. Results are returned all at once
    after all combos finish.
    """
    question = (req.question or "").strip()
    pdf_name = os.path.splitext((req.pdf or "").strip())[0]

    if not question:
        return {"error": "Empty question", "results": []}

    # Build combo list
    if req.combos:
        configs = [
            ComboConfig(
                embed_model=c.embed_model,
                retrieval=c.retrieval,
                llm_model=c.llm_model,
            )
            for c in req.combos
        ]
    else:
        configs = ALL_COMBOS

    # Run sequentially in a single thread — avoids meta tensor race condition
    # when multiple threads try to load the same model simultaneously.
    loop = asyncio.get_event_loop()

    def _run_all_sequential():
        results = []
        for cfg in configs:
            result = run_combo(question, pdf_name, cfg)
            results.append(result)
        return results

    results = await loop.run_in_executor(None, _run_all_sequential)

    return {
        "question": question,
        "pdf":      pdf_name,
        "results":  list(results),
    }


# ── /compare/combos — list all available combos ───────────────

@router.get("/compare/combos")
def list_combos():
    """Returns all 8 predefined combos for the frontend checkboxes."""
    return {
        "combos": [c.to_dict() for c in ALL_COMBOS],
        "embed_models":    ["kannada-bert", "e5"],
        "retrieval_modes": ["hybrid", "dense"],
        "llm_models":      ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    }