"""
src/utils_text.py — CLEANING, CHUNKING, ENRICHMENT FOR KANNADA RAG

Fixes vs original:
  ✔ CHUNK_OVERLAP actually implemented (sliding window)
  ✔ domain detection added
  ✔ difficulty detection added
  ✔ lang_quality score added
"""

import unicodedata
import regex as re
from typing import List, Dict, Any

from indicnlp import common, loader
from indicnlp.tokenize.sentence_tokenize import sentence_split

from src.config import (
    INDIC_NLP_RESOURCES,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MIN_CHUNK_CHARS,
    MIN_KANNADA_RATIO,
)

common.set_resources_path(INDIC_NLP_RESOURCES)
try:
    loader.load()
    print("✅ Indic NLP loaded")
except Exception as e:
    print("❌ Indic NLP failed:", e)


# ── Kannada helpers ──────────────────────────────────────────

def normalize_kannada(text: str) -> str:
    return unicodedata.normalize("NFC", text) if text else ""


def kannada_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if '\u0C80' <= c <= '\u0CFF') / max(len(text), 1)


def has_kannada(text: str, min_chars: int = 20) -> bool:
    return sum(1 for c in text if '\u0C80' <= c <= '\u0CFF') >= min_chars


# ── Cleaning ────────────────────────────────────────────────

VALID_KEEP = re.compile(r"[^\p{Kannada}\p{Latin}\p{N}\p{P}\s]+", flags=re.UNICODE)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = normalize_kannada(text)
    text = text.replace("\r", "\n")
    text = re.sub(r'[\x00-\x1F]', ' ', text)
    text = VALID_KEEP.sub(" ", text)
    text = re.sub(r'https?://\S+', ' ', text)
    text = re.sub(r'\b\d{6,}\b', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    lines = text.split("\n")
    filtered = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if kannada_ratio(line) > MIN_KANNADA_RATIO or re.search(r'[A-Za-z]{5,}', line):
            filtered.append(line)
    return "\n".join(filtered).strip()


# ── Sentence splitting ───────────────────────────────────────

def split_sentences_kn(text: str) -> List[str]:
    try:
        sents = sentence_split(text, lang="kn")
        return [s.strip() for s in sents if s.strip()]
    except Exception:
        return [s.strip() for s in re.split(r'(?<=[.?!।])\s+', text) if s.strip()]


# ── Chunk validation ─────────────────────────────────────────

def is_valid_chunk(text: str) -> bool:
    if not text:
        return False
    if len(text) < MIN_CHUNK_CHARS:
        return False
    if kannada_ratio(text) < MIN_KANNADA_RATIO:
        return False
    return True


# ── Intent / chunk_type detection ───────────────────────────

_INTENT_PATTERNS = [
    ("definition", r"(ಎಂದರೇನು|ವ್ಯಾಖ್ಯಾನ|definition|meaning|ಅರ್ಥ)"),
    ("procedure",  r"(ಹೇಗೆ|steps|procedure|process|ಯಾವ ರೀತಿ)"),
    ("benefits",   r"(ಪ್ರಯೋಜನ|ಲಾಭ|uses|advantages)"),
    ("comparison", r"(ವ್ಯತ್ಯಾಸ|difference|compare|vs)"),
    ("list",       r"(ಪಟ್ಟಿ|ಪ್ರಕಾರಗಳು|types|components)"),
    ("example",    r"(ಉದಾಹರಣೆ|example)"),
    ("formula",    r"(ಸೂತ್ರ|formula|=)"),
    ("cause",      r"(ಏಕೆ|reason|why|because|ಕಾರಣ)"),
    ("story",      r"(ಒಮ್ಮೆ|ಆಗ|ಎಂದ|ಕಥೆ|story)"),
    ("explanation",r"(ವಿವರಣೆ|explain|describe)"),
]


def detect_chunk_type(text: str) -> str:
    t = text.lower()
    for name, pattern in _INTENT_PATTERNS:
        if re.search(pattern, t):
            return name
    return "concept"


# ── Domain detection ─────────────────────────────────────────

_DOMAIN_PATTERNS = [
    ("education", r"(ಪಾಠ|ಅಧ್ಯಾಯ|ವಿದ್ಯಾರ್ಥಿ|lesson|chapter|ಪಠ್ಯ)"),
    ("story",     r"(ಒಮ್ಮೆ|ಕಥೆ|ರಾಜ|ರಾಣಿ|ಅರಮನೆ|once upon)"),
    ("science",   r"(ವಿಜ್ಞಾನ|ಪ್ರಯೋಗ|science|experiment|ಸೂತ್ರ|ಧಾತು)"),
    ("news",      r"(ಸುದ್ದಿ|ಸರ್ಕಾರ|ನ್ಯಾಯಾಲಯ|ಚುನಾವಣೆ|minister)"),
    ("history",   r"(ಇತಿಹಾಸ|ಕ್ರಿ\.ಶ|ಕ್ರಿ\.ಪೂ|ಯುದ್ಧ|history|century)"),
]


def detect_domain(text: str) -> str:
    t = text.lower()
    for domain, pattern in _DOMAIN_PATTERNS:
        if re.search(pattern, t):
            return domain
    return "general"


# ── Difficulty detection ─────────────────────────────────────

def detect_difficulty(text: str) -> str:
    sentences = [s for s in re.split(r'[।.!?]', text) if s.strip()]
    if not sentences:
        return "basic"
    avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
    if avg_words < 8:
        return "basic"
    elif avg_words < 16:
        return "intermediate"
    return "advanced"


# ── Language quality score ───────────────────────────────────

def lang_quality_score(text: str) -> float:
    """
    0.0 = garbage / mostly non-Kannada
    1.0 = clean high-density Kannada
    """
    if not text:
        return 0.0
    kn_r = kannada_ratio(text)
    words = text.split()
    if not words:
        return 0.0
    avg_word_len = sum(len(w) for w in words) / len(words)
    broken_penalty = 0.3 if avg_word_len < 2.5 else 0.0
    return round(max(0.0, kn_r - broken_penalty), 3)


# ── Keywords ─────────────────────────────────────────────────

_STOPWORDS = {
    "ಮತ್ತು", "ಇದು", "ಅದು", "ಇದರ", "ಅದರ", "ಇವು", "ಅವು",
    "the", "and", "for", "with", "that", "this", "are", "was"
}


def extract_keywords(text: str, k: int = 10) -> List[str]:
    words = re.findall(r'[\p{Kannada}A-Za-z]{3,}', text.lower())
    words = [w for w in words if w not in _STOPWORDS]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:k]]


# ── Topic ────────────────────────────────────────────────────

def extract_topic(text: str, keywords: List[str]) -> str:
    first_line = text.split("\n")[0].strip()
    if len(first_line) > 5:
        return first_line[:80]
    return " ".join(keywords[:2]) if keywords else "ಇದು"


# ── Synthetic questions ──────────────────────────────────────

_TEMPLATES = {
    "definition":  ["{t} ಎಂದರೇನು?", "What is {t}?"],
    "procedure":   ["{t} ಹೇಗೆ?", "How to {t}?"],
    "benefits":    ["{t} ಪ್ರಯೋಜನಗಳು ಏನು?", "What are the benefits of {t}?"],
    "comparison":  ["{t} ವ್ಯತ್ಯಾಸ ಏನು?", "What is the difference of {t}?"],
    "list":        ["{t} ಪ್ರಕಾರಗಳು ಯಾವವು?", "What are the types of {t}?"],
    "example":     ["{t} ಉದಾಹರಣೆ ನೀಡಿ", "Give an example of {t}"],
    "formula":     ["{t} ಸೂತ್ರ ಏನು?", "What is the formula for {t}?"],
    "cause":       ["{t} ಏಕೆ?", "Why does {t} happen?"],
    "story":       ["{t} ಏನಾಯಿತು?", "What happened in {t}?"],
    "explanation": ["{t} ವಿವರಿಸಿ", "Explain {t}"],
    "concept":     ["{t} ಏನು?", "What is {t}?"],
}


def generate_synthetic_questions(topic: str, ctype: str) -> List[str]:
    return [q.replace("{t}", topic) for q in _TEMPLATES.get(ctype, _TEMPLATES["concept"])]


# ── Embed text builder ───────────────────────────────────────

def build_chunk_embed_text(text: str, ctype: str, qs: List[str], kw: List[str]) -> str:
    return " | ".join([
        f"intent: {ctype}",
        "questions: " + " ".join(qs[:5]),
        "keywords: " + ", ".join(kw[:10]),
        text
    ])


# ── MAIN CHUNKING FUNCTION ───────────────────────────────────

def chunk_text(text: str) -> List[Dict[str, Any]]:
    text = clean_text(text)
    if not text:
        return []

    # Collect all sentences across all paragraphs
    all_sentences: List[str] = []
    for para in text.split("\n\n"):
        all_sentences.extend(split_sentences_kn(para))
    all_sentences = [s for s in all_sentences if s.strip()]

    if not all_sentences:
        return []

    # Sliding window chunker with real overlap
    raw_chunks: List[str] = []
    current: List[str] = []

    for sent in all_sentences:
        current.append(sent)
        if sum(len(s) for s in current) >= CHUNK_SIZE:
            chunk_text_joined = " ".join(current).strip()
            if is_valid_chunk(chunk_text_joined):
                raw_chunks.append(chunk_text_joined)
            # Keep last CHUNK_OVERLAP sentences for next chunk
            current = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP > 0 else []

    # Last chunk
    if current:
        chunk_text_joined = " ".join(current).strip()
        if is_valid_chunk(chunk_text_joined):
            raw_chunks.append(chunk_text_joined)

    # Deduplicate + enrich
    final: List[Dict[str, Any]] = []
    seen = set()

    for ch in raw_chunks:
        norm = " ".join(ch.split())
        if norm in seen:
            continue
        seen.add(norm)

        ctype   = detect_chunk_type(ch)
        kw      = extract_keywords(ch)
        topic   = extract_topic(ch, kw)
        qs      = generate_synthetic_questions(topic, ctype)
        domain  = detect_domain(ch)
        diff    = detect_difficulty(ch)
        lq      = lang_quality_score(ch)

        final.append({
            "text":                ch,
            "chunk_type":          ctype,
            "keywords":            kw,
            "synthetic_questions": qs,
            "topic":               topic,
            "domain":              domain,
            "difficulty":          diff,
            "lang_quality":        lq,
            "embed_text":          build_chunk_embed_text(ch, ctype, qs, kw),
        })

    return final