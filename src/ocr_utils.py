"""
src/ocr_utils.py — HIGH-QUALITY KANNADA OCR
"""

import os
import math
import concurrent.futures
from typing import List, Dict

import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image, ImageFile

from src.config import OCR_DPI, OCR_MAX_WORKERS

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Edit these paths for your system ────────────────────────
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler\poppler-25.12.0\Library\bin"

_KN_START, _KN_END = 0x0C80, 0x0CFF


def _kannada_char_count(text: str) -> int:
    return sum(1 for c in text if _KN_START <= ord(c) <= _KN_END)


# ── Validation ───────────────────────────────────────────────

def validate_kannada_ocr(text: str, min_chars: int = 30) -> bool:
    if not text:
        return False
    if _kannada_char_count(text) < min_chars:
        return False
    words = text.split()
    if len(words) < 5:
        return False
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < 2:
        return False
    return True


# ── Image preprocessing ──────────────────────────────────────

def _deskew(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return gray
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if abs(angle) < 20:
                angles.append(angle)
    if not angles:
        return gray
    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def preprocess_image(img: Image.Image) -> np.ndarray:
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np.copy()
    h, w = gray.shape
    if w < 1800:
        scale = 1800 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = _deskew(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.adaptiveThreshold(gray, 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 11)
    kernel = np.ones((1, 1), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    return gray


# ── OCR core ────────────────────────────────────────────────

_TESS_CONFIG = "--oem 3 --psm 6 -c preserve_interword_spaces=1"


def _ocr_image(img_np: np.ndarray) -> str:
    pil = Image.fromarray(img_np)
    return pytesseract.image_to_string(pil, lang="kan", config=_TESS_CONFIG).strip()


def ocr_page(pdf_path: str, page_num: int) -> str:
    print(f"🔍 OCR page {page_num}")
    images = convert_from_path(
        pdf_path, dpi=OCR_DPI,
        first_page=page_num, last_page=page_num,
        poppler_path=POPPLER_PATH,
        fmt="png", thread_count=2, use_cropbox=True
    )
    if not images:
        return ""
    return _ocr_image(preprocess_image(images[0]))


# ── Parallel full-PDF OCR (optional) ────────────────────────

def _ocr_worker(args) -> Dict:
    pdf_path, page = args
    try:
        return {"page": page, "text": ocr_page(pdf_path, page)}
    except Exception as e:
        print(f"⚠️ OCR fail page {page}: {e}")
        return {"page": page, "text": ""}


def ocr_pdf(pdf_path: str) -> List[Dict]:
    import fitz
    doc = fitz.open(pdf_path)
    total = len(doc)
    doc.close()
    tasks = [(pdf_path, p) for p in range(1, total + 1)]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=OCR_MAX_WORKERS) as ex:
        futures = [ex.submit(_ocr_worker, t) for t in tasks]
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
    results.sort(key=lambda x: x["page"])
    print(f"✅ OCR done: {len(results)} pages")
    return results