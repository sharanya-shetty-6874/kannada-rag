import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from config import (
    META_DIR, FAISS_INDEX_PATH, FAISS_IDMAP_PATH,
    CHUNKS_PARQUET, EMBED_MODEL, E5_QUERY_PREFIX
)

TOP_K = 3  # how many results to show

def load_index():
    print("🔹 Loading FAISS index and metadata...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    id_map = np.load(FAISS_IDMAP_PATH)
    df = pd.read_parquet(CHUNKS_PARQUET)
    return index, id_map, df

def search(query_text):
    # 1️⃣ Load model + index
    index, id_map, df = load_index()
    model = SentenceTransformer(EMBED_MODEL)

    # 2️⃣ Encode query with E5 query prefix
    q_emb = model.encode(
        [E5_QUERY_PREFIX + query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 3️⃣ Search FAISS index
    scores, ids = index.search(q_emb, TOP_K)
    top_ids = id_map[ids[0]]

    print("\n🔍 Query:", query_text)
    print("📊 Top matches:\n")
    for rank, (i, s) in enumerate(zip(top_ids, scores[0]), 1):
        row = df.iloc[i]
        print(f"#{rank} (Score={s:.4f}) [Page {row['page']}]")
        print(row['text'][:500].replace("\n", " ") + "...\n")

if __name__ == "__main__":
    query = input("📝 Enter your Kannada query: ")
    search(query)
