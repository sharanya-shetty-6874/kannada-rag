"""
src/config.py — KANNADA RAG CONFIG
LLM: Groq (llama-3.3-70b-versatile)
"""

import os
import pathlib
from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).resolve().parent.parent / ".env")
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

# ── Directories ──────────────────────────────────────────────
DATA_RAW_DIR       = str(BASE_DIR / "data" / "raw")
DATA_PROCESSED_DIR = str(BASE_DIR / "data" / "processed")
AUDIO_DIR          = str(BASE_DIR / "data" / "audio")
INDEX_DIR          = str(BASE_DIR / "index" / "faiss")

# ── Local model directories ──────────────────────────────────
# l3cube  → <project_root>/models/
# e5      → <project_root>/backend/models/
MAIN_MODELS_DIR    = str(BASE_DIR / "models")
BACKEND_MODELS_DIR = str(BASE_DIR / "backend" / "models")

# ── Embedding model ──────────────────────────────────────────
EMBED_MODEL = "l3cube-pune/kannada-sentence-bert-nli"

# ── Force offline mode — use locally cached models only ──────
# This prevents any attempt to reach huggingface.co at runtime.
# Set BEFORE any transformers / sentence-transformers import.
os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ["TRANSFORMERS_OFFLINE"]  = "1"

# Point HuggingFace cache to the main models dir (l3cube lives here)
os.environ["HF_HOME"]               = MAIN_MODELS_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = MAIN_MODELS_DIR

# ── Indic NLP resources ──────────────────────────────────────
INDIC_NLP_RESOURCES = str(BASE_DIR / "src" / "indic_resources")

# ── Offline / model cache ────────────────────────────────────
# SentenceTransformer cache also points to the main models dir
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", MAIN_MODELS_DIR)

# ── Chunking ─────────────────────────────────────────────────
CHUNK_SIZE        = 900
CHUNK_OVERLAP     = 2
MIN_CHUNK_CHARS   = 80
MIN_KANNADA_RATIO = 0.20

# ── OCR ──────────────────────────────────────────────────────
OCR_DPI               = 400
OCR_MIN_KANNADA_CHARS = 30
OCR_MAX_WORKERS       = 4
OCR_TIMEOUT           = 300

# ── Retrieval ────────────────────────────────────────────────
TOP_K             = 5
MIN_SCORE         = 0.30
MAX_CONTEXT_CHARS = 3200

# ── Reranking weights ────────────────────────────────────────
INTENT_BOOST            = 0.18
KEYWORD_BOOST_PER_MATCH = 0.02
MAX_KEYWORD_BOOST       = 0.08

# ── Hybrid retrieval weights ─────────────────────────────────
DENSE_WEIGHT  = 0.65
SPARSE_WEIGHT = 0.35

# ── Groq LLM ─────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── Create dirs ───────────────────────────────────────────────
for _p in [DATA_RAW_DIR, DATA_PROCESSED_DIR, AUDIO_DIR, INDEX_DIR]:
    os.makedirs(_p, exist_ok=True)

# ── Local model path resolver ─────────────────────────────────

def find_local_model(model_id: str, search_dirs: list) -> str:
    """
    Searches for a locally saved model and returns its path.

    Handles all common folder structures:
      • direct name:       models/multilingual-e5-small/
      • org/name:          models/intfloat/multilingual-e5-small/
      • HF hub cache:      models/models--intfloat--multilingual-e5-small/snapshots/<hash>/
      • ST cache (dashes): models/intfloat_multilingual-e5-small/

    Returns the resolved local path if found, otherwise returns model_id
    unchanged (so the caller can still attempt a hub load).
    """
    parts     = model_id.split("/")
    org       = parts[0] if len(parts) > 1 else ""
    name      = parts[-1]
    hf_cache  = model_id.replace("/", "--")  # "intfloat--multilingual-e5-small"
    st_cache  = model_id.replace("/", "_")   # "intfloat_multilingual-e5-small"

    def _has_model_files(p: str) -> bool:
        if not os.path.isdir(p):
            return False
        files = os.listdir(p)
        return any(f in files for f in ("config.json", "tokenizer_config.json", "tokenizer.json"))

    for base in search_dirs:
        if not base or not os.path.isdir(base):
            continue

        candidates = [
            os.path.join(base, name),                # multilingual-e5-small/
            os.path.join(base, org, name),            # intfloat/multilingual-e5-small/
            os.path.join(base, st_cache),             # intfloat_multilingual-e5-small/
            os.path.join(base, hf_cache),             # intfloat--multilingual-e5-small/
            os.path.join(base, f"models--{hf_cache}"), # models--intfloat--multilingual-e5-small/
        ]

        for candidate in candidates:
            if _has_model_files(candidate):
                print(f"✅ Found local model at: {candidate}")
                return candidate
            # HuggingFace hub cache stores weights inside snapshots/<hash>/
            snapshots_dir = os.path.join(candidate, "snapshots")
            if os.path.isdir(snapshots_dir):
                snaps = sorted(os.listdir(snapshots_dir))
                for snap in snaps:
                    snap_path = os.path.join(snapshots_dir, snap)
                    if _has_model_files(snap_path):
                        print(f"✅ Found local model (hub cache) at: {snap_path}")
                        return snap_path

    print(f"⚠️  Local model not found for '{model_id}' in {search_dirs}")
    return model_id  # fall back — will fail if offline


print("📦 Config loaded")
print(f"   RAW:             {DATA_RAW_DIR}")
print(f"   PROCESSED:       {DATA_PROCESSED_DIR}")
print(f"   INDEX:           {INDEX_DIR}")
print(f"   EMBED:           {EMBED_MODEL}")
print(f"   MAIN_MODELS:     {MAIN_MODELS_DIR}")
print(f"   BACKEND_MODELS:  {BACKEND_MODELS_DIR}")
print(f"   LLM:             Groq / {GROQ_MODEL}")
print(f"   OFFLINE:         HF_HUB_OFFLINE=1")