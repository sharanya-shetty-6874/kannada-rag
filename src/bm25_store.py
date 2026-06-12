"""
src/bm25_store.py — BM25 SPARSE INDEX FOR HYBRID RETRIEVAL

pip install rank-bm25
"""

import os
import pickle
import regex as re
from typing import List, Tuple


def _tokenize_kannada(text: str) -> List[str]:
    """
    Tokenize Kannada + Latin text for BM25.
    Splits on Kannada word boundaries and spaces.
    """
    return re.findall(r'[\p{Kannada}]+|[a-zA-Z]{2,}', text.lower())


class BM25Store:

    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.path      = os.path.join(index_dir, "bm25.pkl")
        self.bm25      = None
        self.corpus:   List[str] = []

    def build(self, texts: List[str]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("Install rank-bm25: pip install rank-bm25")

        self.corpus    = texts
        tokenized      = [_tokenize_kannada(t) for t in texts]
        self.bm25      = BM25Okapi(tokenized)

        os.makedirs(self.index_dir, exist_ok=True)
        with open(self.path, "wb") as f:
            pickle.dump((self.bm25, self.corpus), f)
        print(f"✅ BM25 index built: {len(texts)} docs → {self.path}")

    def load(self):
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"BM25 index not found: {self.path}")
        with open(self.path, "rb") as f:
            self.bm25, self.corpus = pickle.load(f)

    def search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if self.bm25 is None:
            return []
        tokens = _tokenize_kannada(query)
        if not tokens:
            return []
        scores  = self.bm25.get_scores(tokens)
        ranked  = sorted(enumerate(scores), key=lambda x: -x[1])
        return [(idx, float(sc)) for idx, sc in ranked[:top_k] if sc > 0]