"""
src/build_faiss_index.py — FAISS + BM25 INDEXING FOR KANNADA RAG

Changes vs original:
  ✔ _load_index() exposed (used by RetrievalAgent)
  ✔ Embedder singletons for both KN-BERT and E5 (no double-load)
  ✔ BM25 index built alongside FAISS
  ✔ New chunk fields (domain, difficulty, lang_quality) preserved
  ✔ Embedding cache cleared on re-build
  ✔ Uses find_local_model() — no HuggingFace network call
  ✔ build_e5_index_for_pdf() writes E5 index into pdf_dir/e5/ subfolder
    so compare panel can use the correct embedding space per model
"""

import os
import json
import numpy as np
import pandas as pd
import faiss
import threading
from functools import lru_cache

from src.config import (
    DATA_PROCESSED_DIR,
    INDEX_DIR,
    EMBED_MODEL,
    MAIN_MODELS_DIR,
    BACKEND_MODELS_DIR,
    find_local_model,
)
from src.bm25_store import BM25Store

# ── E5 model constants ────────────────────────────────────────
E5_MODEL  = "intfloat/multilingual-e5-small"
E5_SUBDIR = "e5"   # lives at  index/faiss/{pdf}/e5/

# ── KN-BERT singleton ─────────────────────────────────────────
_knbert_embedder      = None
_knbert_embedder_lock = threading.Lock()

# ── E5 singleton ──────────────────────────────────────────────
_e5_embedder          = None
_e5_embedder_lock     = threading.Lock()

os.makedirs(INDEX_DIR, exist_ok=True)


# ── KN-BERT loader (main pipeline) ────────────────────────────

def get_embedder():
    """
    Returns the KN-BERT singleton (l3cube-pune/kannada-sentence-bert-nli).
    Loaded from MAIN_MODELS_DIR — no network call.
    """
    global _knbert_embedder
    if _knbert_embedder is not None:
        return _knbert_embedder
    with _knbert_embedder_lock:
        if _knbert_embedder is None:
            from sentence_transformers import SentenceTransformer
            local_path = find_local_model(EMBED_MODEL, search_dirs=[MAIN_MODELS_DIR])
            print(f"🔹 Loading KN-BERT embedder from: {local_path}")
            _knbert_embedder = SentenceTransformer(local_path, device="cpu")
    return _knbert_embedder


# ── E5 loader (compare panel only) ────────────────────────────

def get_e5_embedder():
    """
    Returns the E5 singleton (intfloat/multilingual-e5-small).
    Loaded from BACKEND_MODELS_DIR — no network call.
    """
    global _e5_embedder
    if _e5_embedder is not None:
        return _e5_embedder
    with _e5_embedder_lock:
        if _e5_embedder is None:
            import torch
            import torch.nn.functional as F
            from transformers import AutoTokenizer, AutoModel

            local_path = find_local_model(
                E5_MODEL,
                search_dirs=[BACKEND_MODELS_DIR, MAIN_MODELS_DIR],
            )
            if local_path == E5_MODEL:
                raise FileNotFoundError(
                    f"E5 model not found locally.\n"
                    f"  Searched: {BACKEND_MODELS_DIR}, {MAIN_MODELS_DIR}\n"
                    f"  Expected: multilingual-e5-small/ or "
                    f"models--intfloat--multilingual-e5-small/"
                )

            print(f"🔹 Loading E5 embedder from: {local_path}")
            tokenizer = AutoTokenizer.from_pretrained(
                local_path, local_files_only=True
            )
            raw_model = AutoModel.from_pretrained(
                local_path,
                local_files_only=True,
                low_cpu_mem_usage=False,
            )
            raw_model.eval()

            class _E5Embedder:
                def __init__(self, tok, mod):
                    self.tokenizer = tok
                    self.model     = mod

                def encode(self, texts, convert_to_numpy=True,
                           normalize_embeddings=True, batch_size=32, **kw):
                    if isinstance(texts, str):
                        texts = [texts]
                    all_embs = []
                    for i in range(0, len(texts), batch_size):
                        batch  = texts[i: i + batch_size]
                        inputs = self.tokenizer(
                            batch, return_tensors="pt",
                            padding=True, truncation=True, max_length=512,
                        )
                        with torch.no_grad():
                            out = self.model(**inputs)
                        emb = out.last_hidden_state.mean(dim=1)
                        if normalize_embeddings:
                            emb = F.normalize(emb, p=2, dim=1)
                        all_embs.append(emb.numpy() if convert_to_numpy else emb)
                    return np.vstack(all_embs) if convert_to_numpy else torch.cat(all_embs)

            _e5_embedder = _E5Embedder(tokenizer, raw_model)
            print("✅ E5 embedder ready")
    return _e5_embedder


# ── Index loader (LRU-cached, used by RetrievalAgent) ─────────

@lru_cache(maxsize=8)
def _load_index(pdf_name: str):
    """
    Loads KN-BERT FAISS + BM25 + DataFrame.
    Called by src/agents/retrieval_agent.py.
    """
    path = os.path.join(INDEX_DIR, pdf_name)

    faiss_path   = os.path.join(path, "index.faiss")
    parquet_path = os.path.join(path, "chunks.parquet")

    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"chunks.parquet not found: {parquet_path}")

    faiss_index = faiss.read_index(faiss_path)
    df          = pd.read_parquet(parquet_path)

    bm25 = BM25Store(path)
    try:
        bm25.load()
    except FileNotFoundError:
        print(f"⚠️ BM25 index not found for {pdf_name} — using dense-only")
        bm25 = None

    return faiss_index, bm25, df


# ── Helpers ───────────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return " ".join(text.strip().split())


def _build_embed_text(row: dict) -> str:
    embed = row.get("embed_text", "")
    if embed and len(embed.strip()) > 30:
        return _normalize_text(embed)
    parts = [f"intent: {row.get('chunk_type', 'concept')}"]
    sq = row.get("synthetic_questions", [])
    if sq:
        parts.append("questions: " + " ".join(sq[:5]))
    kw = row.get("keywords", [])
    if kw:
        parts.append("keywords: " + ", ".join(kw[:10]))
    parts.append(row.get("text", ""))
    return _normalize_text(" | ".join(parts))


def _choose_faiss_index(embs: np.ndarray):
    n, dim = embs.shape
    if n < 2000:
        print("⚡ FAISS: IndexFlatIP (exact)")
        return faiss.IndexFlatIP(dim), "flat"
    elif n < 15000:
        print("⚡ FAISS: HNSW")
        idx = faiss.IndexHNSWFlat(dim, 32)
        idx.hnsw.efConstruction = 40
        idx.hnsw.efSearch       = 64
        return idx, "hnsw"
    else:
        print("⚡ FAISS: IVF (large dataset)")
        quantizer = faiss.IndexFlatIP(dim)
        nlist     = max(32, min(int(np.sqrt(n)), 512))
        idx       = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        print("🔧 Training IVF...")
        idx.train(embs)
        idx.nprobe = min(10, nlist)
        return idx, "ivf"


# ── Generic index writer ──────────────────────────────────────

def _build_index_into(
    pdf_base:    str,
    out_dir:     str,
    model,
    model_name:  str,
    e5_prefix:   bool = False,
) -> bool:
    """
    Reads processed JSONL, encodes with `model`, saves FAISS+BM25 into out_dir.
    e5_prefix=True → prepends "passage: " to each chunk (E5 training requirement).
    Returns True on success.
    """
    jsonl_path = os.path.join(DATA_PROCESSED_DIR, f"{pdf_base}.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"❌ Missing: {jsonl_path}")
        return False

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                rows.append(json.loads(line.strip()))
            except Exception as e:
                print(f"⚠️ Skipping line {i}: {e}")

    if not rows:
        print("❌ No chunks found")
        return False

    df = pd.DataFrame(rows)
    df["text"] = df["text"].fillna("").astype(str).apply(_normalize_text)
    df = df[df["text"].str.len() > 50].reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    if before != len(df):
        print(f"🧹 Removed {before - len(df)} duplicates")

    if df.empty:
        print("❌ No valid chunks after cleaning")
        return False

    df["embed_text"] = df.apply(lambda r: _build_embed_text(r.to_dict()), axis=1)

    texts = (
        ["passage: " + t for t in df["embed_text"].tolist()]
        if e5_prefix
        else df["embed_text"].tolist()
    )

    print(f"🔹 Encoding {len(df)} chunks with {model_name}...")
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    if embeddings is None or len(embeddings) == 0:
        print("❌ Embedding failed")
        return False

    faiss_index, index_type = _choose_faiss_index(embeddings)
    faiss_index.add(embeddings)

    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(faiss_index, os.path.join(out_dir, "index.faiss"))
    np.save(os.path.join(out_dir, "idmap.npy"), df.index.values)

    save_cols = [c for c in df.columns if c != "embed_text"]
    df[save_cols].to_parquet(os.path.join(out_dir, "chunks.parquet"), index=False)

    with open(os.path.join(out_dir, "index_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "index_type":      index_type,
            "embedding_model": model_name,
            "num_chunks":      len(df),
        }, f, ensure_ascii=False, indent=2)

    bm25 = BM25Store(out_dir)
    bm25.build(df["text"].tolist())

    print(f"✅ Index ready: {out_dir}")
    print(f"   chunks={len(df)}  faiss={index_type}  bm25=✓  model={model_name}")
    return True


# ── Public builders ───────────────────────────────────────────

def build_index_for_pdf(pdf_base: str):
    """
    Builds the KN-BERT index (main pipeline).
    Saves to: index/faiss/{pdf_base}/
    """
    pdf_base = os.path.splitext((pdf_base or "").strip())[0]
    out_dir  = os.path.join(INDEX_DIR, pdf_base)
    print(f"\n📦 Building KN-BERT index: {pdf_base}")

    success = _build_index_into(
        pdf_base, out_dir,
        model=get_embedder(),
        model_name=EMBED_MODEL,
        e5_prefix=False,
    )
    if success:
        _load_index.cache_clear()


def build_e5_index_for_pdf(pdf_base: str):
    """
    Builds the E5 index for the compare panel.
    Saves to: index/faiss/{pdf_base}/e5/
    Safe to call in a background thread after build_index_for_pdf().
    """
    pdf_base = os.path.splitext((pdf_base or "").strip())[0]
    out_dir  = os.path.join(INDEX_DIR, pdf_base, E5_SUBDIR)
    print(f"\n📦 Building E5 index: {pdf_base}")

    try:
        model = get_e5_embedder()
    except FileNotFoundError as e:
        print(f"⚠️ Skipping E5 index — {e}")
        return

    _build_index_into(
        pdf_base, out_dir,
        model=model,
        model_name=E5_MODEL,
        e5_prefix=True,
    )


# ── Build all ─────────────────────────────────────────────────

def build_all():
    files = [f for f in os.listdir(DATA_PROCESSED_DIR) if f.endswith(".jsonl")]
    if not files:
        print("⚠️ No processed files found")
        return
    for f in files:
        base = f.replace(".jsonl", "")
        build_index_for_pdf(base)
        build_e5_index_for_pdf(base)


if __name__ == "__main__":
    build_all()