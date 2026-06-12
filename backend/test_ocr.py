import sys
sys.path.insert(0, r'C:\Users\SHARANYA\Downloads\kannada-rag\kannada-rag\backend')

import fitz
import cv2
import numpy as np
import pytesseract

_KAN_LO = '\u0C80'
_KAN_HI = '\u0CFF'

def ocr_page_raw(page, dpi=300):
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
    return pytesseract.image_to_string(binary, lang="kan+eng", config="--psm 6 --oem 3")

# Check these skipped pages
skipped = [3, 4, 6, 7, 8, 9, 11, 13, 14, 15, 17, 18, 19, 20]

d = fitz.open(r'C:\Users\SHARANYA\Downloads\kannada-rag\kannada-rag\data\raw\Kannada Stories.pdf')
for pg in skipped[:6]:
    text = ocr_page_raw(d[pg-1], dpi=300)
    kan_count = sum(1 for c in text if _KAN_LO <= c <= _KAN_HI)
    print(f'=== Page {pg} | kannada_chars={kan_count} ===')
    print(repr(text[:600]))
    print()
d.close()