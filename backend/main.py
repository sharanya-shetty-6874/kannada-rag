"""
backend/main.py — AGENT-BASED KANNADA RAG API

Changes vs original:
  ✔ /query returns richer response (confidence, hallucination_risk, sub_intents, entities)
  ✔ /upload_pdf clears orchestrator cache on re-upload
  ✔ /pipeline_status endpoint (shows last pipeline log)
  ✔ Async OCR + indexing (non-blocking upload)
  ✔ /compare endpoint added (runs 8 pipeline combos in parallel)
  ✔ Background indexing now builds BOTH KN-BERT and E5 indexes so the
    compare panel can use the correct embedding space for each model
  ✔ All other endpoints unchanged
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

import shutil
import asyncio
from fastapi import FastAPI, UploadFile, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from rag_pipeline import get_rag_answer, _ORCHESTRATORS
from src.ingest_pdf import process_pdf
from src.build_faiss_index import build_index_for_pdf, build_e5_index_for_pdf
from src.config import DATA_RAW_DIR, AUDIO_DIR

# ── Compare router ────────────────────────────────────────────
from compare_endpoint import router as compare_router

try:
    from tts_service import generate_kannada_voice
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False


app = FastAPI(title="🌸 Kannada Agent RAG API")

# ── Register compare router ───────────────────────────────────
app.include_router(compare_router)

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Upload status tracker ─────────────────────────────────────
_UPLOAD_STATUS: dict = {}


class QueryRequest(BaseModel):
    question: str
    pdf: str


# ── Health ────────────────────────────────────────────────────

@app.get("/ping")
def ping():
    return {"status": "ok", "message": "Kannada Agent RAG running 🚀"}


@app.get("/")
def root():
    return {"message": "🌸 Kannada Agent RAG API. Visit /docs"}


# ── Upload + Index (async background) ─────────────────────────

def _process_and_index(filename: str, base: str):
    """
    Runs in background thread — non-blocking for the API.

    Step 1: OCR / text extraction
    Step 2: Build KN-BERT index  (main pipeline)
    Step 3: Build E5 index       (compare panel)

    E5 indexing is attempted after KN-BERT finishes. If the E5 model
    is not available locally, it logs a warning and continues — the main
    pipeline is unaffected.
    """
    try:
        # ── Step 1: OCR ──────────────────────────────────────
        _UPLOAD_STATUS[base] = {"status": "processing", "message": "PDF processing..."}
        process_pdf(filename)

        # ── Step 2: KN-BERT index ────────────────────────────
        _UPLOAD_STATUS[base] = {"status": "indexing", "message": "Building KN-BERT index..."}
        build_index_for_pdf(base)

        # Evict stale orchestrator so next query picks up fresh index
        _ORCHESTRATORS.pop(base, None)

        # ── Step 3: E5 index (compare panel) ─────────────────
        _UPLOAD_STATUS[base] = {
            "status":  "indexing_e5",
            "message": "Building E5 index for compare panel...",
        }
        build_e5_index_for_pdf(base)

        _UPLOAD_STATUS[base] = {
            "status":  "ready",
            "message": f"{filename} processed & indexed (KN-BERT + E5)",
        }
        print(f"✅ Background indexing done: {base}")

    except Exception as e:
        _UPLOAD_STATUS[base] = {"status": "error", "message": str(e)}
        print(f"❌ Background indexing error: {e}")


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks):
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF allowed")

        filename = os.path.basename(file.filename)
        pdf_path = os.path.join(DATA_RAW_DIR, filename)

        with open(pdf_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        base = os.path.splitext(filename)[0]
        _UPLOAD_STATUS[base] = {"status": "queued", "message": "Upload received, processing..."}

        # Non-blocking: process + index (KN-BERT + E5) in background
        background_tasks.add_task(_process_and_index, filename, base)

        return {
            "status":   "accepted",
            "message":  f"{filename} upload accepted. Processing in background.",
            "pdf_name": base,
            "poll_url": f"/upload_status/{base}",
        }

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Upload error:", e)
        raise HTTPException(status_code=500, detail="Upload failed")


@app.get("/upload_status/{pdf_base}")
def upload_status(pdf_base: str):
    return _UPLOAD_STATUS.get(
        pdf_base,
        {"status": "unknown", "message": "No upload found for this PDF"}
    )


# ── Rebuild E5 index on demand ────────────────────────────────
# Useful for PDFs that were uploaded before E5 indexing was added.

@app.post("/rebuild_e5_index/{pdf_base}")
async def rebuild_e5_index(pdf_base: str, background_tasks: BackgroundTasks):
    """
    Rebuilds (or builds for the first time) the E5 index for an existing PDF.
    Triggers in the background; poll /upload_status/{pdf_base} for progress.
    """
    _UPLOAD_STATUS[pdf_base] = {
        "status":  "indexing_e5",
        "message": "Rebuilding E5 index...",
    }

    def _do_e5(base: str):
        try:
            build_e5_index_for_pdf(base)
            _UPLOAD_STATUS[base] = {
                "status":  "ready",
                "message": "E5 index rebuilt successfully",
            }
        except Exception as e:
            _UPLOAD_STATUS[base] = {"status": "error", "message": str(e)}

    background_tasks.add_task(_do_e5, pdf_base)
    return {
        "status":   "accepted",
        "message":  f"E5 index rebuild started for '{pdf_base}'.",
        "poll_url": f"/upload_status/{pdf_base}",
    }


# ── Query ─────────────────────────────────────────────────────

@app.post("/query")
async def query(req: QueryRequest):
    try:
        question = (req.question or "").strip()
        pdf_name = os.path.splitext(req.pdf)[0].strip()

        if not question:
            raise HTTPException(status_code=400, detail="Empty question")

        # Check if PDF is still being indexed
        status = _UPLOAD_STATUS.get(pdf_name, {})
        if status.get("status") in ("queued", "processing", "indexing", "indexing_e5"):
            raise HTTPException(
                status_code=202,
                detail=f"PDF is still being indexed: {status.get('message', '')}"
            )

        print(f"\n❓ Q: {question}")
        print(f"📘 PDF: {pdf_name}")

        result = get_rag_answer(question, pdf_name)

        return {
            "answer":             result.get("answer", "ಕ್ಷಮಿಸಿ, ಮಾಹಿತಿ ದೊರೆಯಲಿಲ್ಲ."),
            "intent":             result.get("intent", ""),
            "sub_intents":        result.get("sub_intents", []),
            "entities":           result.get("entities", []),
            "quality":            result.get("quality", ""),
            "confidence":         result.get("confidence", 0.0),
            "hallucination_risk": result.get("hallucination_risk", "low"),
            "chunks_used":        result.get("chunks_used", 0),
            "domain":             result.get("domain", ""),
            "retry_count":        result.get("retry_count", 0),
            "latency_ms":         result.get("latency_ms", 0.0),
        }

    except HTTPException:
        raise
    except Exception as e:
        print("❌ Query error:", e)
        raise HTTPException(status_code=500, detail="Query failed")


# ── Pipeline debug log ─────────────────────────────────────────

@app.post("/query_debug")
async def query_debug(req: QueryRequest):
    """Same as /query but also returns full pipeline_log for debugging."""
    try:
        question = (req.question or "").strip()
        pdf_name = os.path.splitext(req.pdf)[0].strip()

        if not question:
            raise HTTPException(status_code=400, detail="Empty question")

        result = get_rag_answer(question, pdf_name)
        result["pipeline_log"] = result.get("pipeline_log", [])
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── TTS ────────────────────────────────────────────────────────

@app.post("/tts")
async def tts(request: Request):
    if not TTS_AVAILABLE:
        raise HTTPException(status_code=500, detail="TTS not available")
    try:
        body = await request.json()
        text = (body.get("text") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text")

        audio_path = await generate_kannada_voice(text)
        filename   = os.path.basename(audio_path)
        base_url   = str(request.base_url).rstrip("/")

        return {"url": f"{base_url}/audio/{filename}"}

    except HTTPException:
        raise
    except Exception as e:
        print("❌ TTS error:", e)
        raise HTTPException(status_code=500, detail="TTS failed")