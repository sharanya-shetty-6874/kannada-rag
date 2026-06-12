"""
src/ingest_pdf.py — SMART PDF INGESTION FOR KANNADA RAG

Fixes vs original:
  ✔ pymupdf4llm loaded ONCE per PDF (not per page)
  ✔ quality_score tracked per page
  ✔ passes through new chunk fields (domain, difficulty, lang_quality)
"""

import os
import json
import fitz

from src.config import DATA_RAW_DIR, DATA_PROCESSED_DIR, OCR_MIN_KANNADA_CHARS
from src.ocr_utils import ocr_page, validate_kannada_ocr
from src.utils_text import clean_text, chunk_text


# ── Kannada quality helpers ──────────────────────────────────

def _kannada_count(text: str) -> int:
    return sum(1 for c in text if '\u0C80' <= c <= '\u0CFF')


def _has_broken_kannada(text: str) -> bool:
    words = [w for w in text.split() if _kannada_count(w) > 0]
    if not words:
        return False
    return (sum(len(w) for w in words) / len(words)) < 2.5


def _quality_score(text: str) -> float:
    """Returns 0.0–1.0 Kannada quality score."""
    if not text or len(text.strip()) < 50:
        return 0.0
    kn = _kannada_count(text)
    ratio = kn / max(len(text), 1)
    broken_penalty = 0.4 if _has_broken_kannada(text) else 0.0
    return max(0.0, ratio - broken_penalty)


def _is_meaningful(text: str, min_score: float = 0.15) -> bool:
    if not text or len(text.strip()) < 100:
        return False
    if _kannada_count(text) < OCR_MIN_KANNADA_CHARS:
        return False
    return _quality_score(text) >= min_score


# ── Pre-load markdown for entire PDF ─────────────────────────

def _load_markdown_pages(pdf_path: str) -> dict:
    """
    Returns {page_0_index: markdown_text} for all pages.
    Called ONCE per PDF — not per page.
    """
    md_pages = {}
    try:
        import pymupdf4llm
        import fitz as _fitz
        doc = _fitz.open(pdf_path)
        total = len(doc)
        doc.close()
        for page_0 in range(total):
            try:
                md = pymupdf4llm.to_markdown(
                    pdf_path,
                    pages=[page_0],
                    show_progress=False
                ) or ""
                md_pages[page_0] = md
            except Exception:
                md_pages[page_0] = ""
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️ Markdown pre-load failed: {e}")
    return md_pages


# ── Per-page extraction ──────────────────────────────────────

def extract_page_text(pdf_path: str, page: fitz.Page,
                      page_num: int, md_pages: dict):
    """
    Tier 1: fitz
    Tier 2: pre-loaded markdown
    Tier 3: OCR
    """
    page_0 = page_num - 1

    # Tier 1: fitz
    try:
        text_fitz = page.get_text("text") or ""
    except Exception:
        text_fitz = ""

    if _is_meaningful(text_fitz):
        print(f"  page {page_num}: ✅ fitz  (score={_quality_score(text_fitz):.2f})")
        return text_fitz, "fitz"

    # Tier 2: markdown (pre-loaded)
    text_md = md_pages.get(page_0, "")
    if _is_meaningful(text_md):
        print(f"  page {page_num}: ✅ markdown  (score={_quality_score(text_md):.2f})")
        return text_md, "markdown"

    # Tier 3: OCR
    print(f"  page {page_num}: 🔥 forcing OCR")
    try:
        ocr_text = ocr_page(pdf_path, page_num)
    except Exception as e:
        print(f"  page {page_num}: ❌ OCR error: {e}")
        return "", "failed"

    if validate_kannada_ocr(ocr_text):
        print(f"  page {page_num}: ✅ OCR accepted")
        return ocr_text, "ocr"

    print(f"  page {page_num}: ⚠️ OCR garbage — skipped")
    return "", "failed"


# ── Main process function ────────────────────────────────────

def process_pdf(filename: str):
    pdf_path = os.path.join(DATA_RAW_DIR, filename)
    base     = os.path.splitext(filename)[0]
    out_path = os.path.join(DATA_PROCESSED_DIR, f"{base}.jsonl")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n📄 Processing: {filename}")

    doc        = fitz.open(pdf_path)
    md_pages   = _load_markdown_pages(pdf_path)
    all_chunks = []
    chunk_id   = 0
    stats      = {"fitz": 0, "markdown": 0, "ocr": 0, "failed": 0}

    for i in range(len(doc)):
        page_num = i + 1
        page     = doc[i]

        raw_text, method = extract_page_text(pdf_path, page, page_num, md_pages)
        stats[method] = stats.get(method, 0) + 1

        if not raw_text.strip():
            continue

        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        chunks = chunk_text(cleaned)

        for ch in chunks:
            ch.update({
                "chunk_id":       f"{base}_p{page_num}_c{chunk_id}",
                "page":           page_num,
                "source":         filename,
                "extract_method": method,
            })
            all_chunks.append(ch)
            chunk_id += 1

    doc.close()

    if not all_chunks:
        print("⚠️ No valid chunks extracted")
        return

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"\n✅ Done — {len(all_chunks)} chunks saved to {out_path}")
    print("📊 Method stats:", stats)