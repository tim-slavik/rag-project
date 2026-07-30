import sys
import os

# -------------------------------------------------
# Fix Python import path so project modules resolve
# -------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import json
import glob
import pickle

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from multiprocessing import Pool
from chunking.chunker import Chunker
from chunking.strategies import FixedSizeChunking
from vectorstore.faiss_store import FaissStore


# -------------------------------------------------
# 1. Load AMAQA dataset (Telegram JSONL + Hotel CSV)
# -------------------------------------------------
def load_amaqa():
    print("Loading AMAQA dataset from local GitHub JSONL + CSV files...")

    # -----------------------------
    # Load Telegram JSONL manually
    # -----------------------------
    telegram_records = []
    telegram_files = glob.glob("data/amaqa_raw/data/telegram/*.jsonl")

    for file in tqdm(telegram_files, desc="Telegram files"):
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                telegram_records.append(json.loads(line))

    print(f"Loaded {len(telegram_records):,} Telegram messages")

    # -----------------------------
    # Load Hotel Reviews CSV
    # -----------------------------
    reviews_df = pd.read_csv("data/amaqa_raw/data/hotel_reviews/hotel_reviews.csv")
    reviews_records = reviews_df.to_dict(orient="records")
    print(f"Loaded {len(reviews_records):,} hotel reviews")
    return telegram_records, reviews_records


# -------------------------------------------------
# 2. Extract text + metadata (clean + deterministic)
# -------------------------------------------------
def extract_records(records, source):
    extracted = []

    print(f"Extracting {len(records):,} {source} records...")
    for row in tqdm(records, desc=f"Extract {source}"):

        if source == "telegram":
            # Telegram JSONL always has "text"
            text = row["text"]
            metadata = {k: v for k, v in row.items() if k != "text"}

        else:
            # Hotel reviews CSV: detect the correct text column
            # Common AMAQA hotel review text column is "review"
            if "review" in row:
                text = row["review"]
                metadata = {k: v for k, v in row.items() if k != "review"}

            elif "text" in row:
                text = row["text"]
                metadata = {k: v for k, v in row.items() if k != "text"}

            else:
                # Fallback: treat entire row as text
                text = str(row)
                metadata = {}

        extracted.append((text, metadata))

    return extracted


# -------------------------------------------------
# 3. Chunk documents using your real Chunker
# -------------------------------------------------
def chunk_records(records, chunk_size=300, overlap=50):
    strategy = FixedSizeChunking(size=chunk_size, overlap=overlap)
    chunker = Chunker(strategy=strategy)

    chunks = []
    print(f"Chunking {len(records):,} records ...")
    for text, amaqa_metadata in tqdm(records, desc="Chunking"):
        raw_chunks = chunker.run(text)

        for ch in raw_chunks:
            merged_metadata = {
                **ch["metadata"],        # chunk-level metadata
                **amaqa_metadata         # AMAQA metadata
            }

            chunks.append({
                "text": ch["text"],
                "metadata": merged_metadata
            })

    # Assign global chunk IDs
    for i, chunk in enumerate(chunks):
        chunk["metadata"]["chunk_id"] = i

    print(f"Generated {len(chunks):,} chunks")

    return chunks


# -------------------------------------------------
# 4. Multiprocessing embedding + streaming FAISS
# -------------------------------------------------
def embed_batch(batch):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model.encode(batch, convert_to_numpy=True).astype("float32")


def embed_and_build_faiss_mp(chunks, batch_size=512, workers=4):
    print(f"Embedding {len(chunks):,} chunks using {workers} workers...")

    texts = [c["text"] for c in chunks]
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    print(f"Created {len(batches)} batches of size {batch_size}")

    # Initialize FAISS using a single model instance to get dim
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    dim = model.get_embedding_dimension()
    store = FaissStore(dim=dim)

    with Pool(processes=workers) as pool:
        for emb in tqdm(pool.imap(embed_batch, batches), total=len(batches), desc="Embedding + FAISS"):
            store.add(emb)

    print("FAISS index built.")
    return store


# # -------------------------------------------------
# # 5a. Build FAISS index
# # -------------------------------------------------
# def build_faiss(embeddings):
#     dim = embeddings.shape[1]
#     store = FaissStore(dim=dim)
#     store.add(embeddings.astype("float32"))
#     return store


# -------------------------------------------------
# 5b. Build BM25 index
# -------------------------------------------------
def build_bm25(chunks):
    print("Building BM25 index...")
    tokenized = [c["text"].split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print("BM25 index built.")
    return bm25


# -------------------------------------------------
# 6. Save artifacts
# -------------------------------------------------
def save_all(chunks, faiss_store, bm25):
    print("Saving AMAQA artifacts...")

    # FAISS
    faiss_store.save("data/amaqa.index")

    # Text
    texts = [c["text"] for c in chunks]
    with open("data/amaqa_text.json", "w") as f:
        json.dump(texts, f)

    # Metadata
    metadata = [c["metadata"] for c in chunks]
    with open("data/amaqa_metadata.json", "w") as f:
        json.dump(metadata, f)

    # BM25
    with open("data/amaqa_bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    print("Saved:")
    print(" - data/amaqa.index")
    print(" - data/amaqa_embeddings.npy")
    print(" - data/amaqa_text.json")
    print(" - data/amaqa_metadata.json")
    print(" - data/amaqa_bm25.pkl")


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------
def run_ingestion():
    telegram, reviews = load_amaqa()

    print("Extracting records...")
    records_telegram = extract_records(telegram, source="telegram")
    records_reviews = extract_records(reviews, source="reviews")
    records = records_telegram + records_reviews

    print("Chunking records...")
    chunks = chunk_records(records)

    faiss_store = embed_and_build_faiss_mp(chunks, batch_size=512, workers=4)
    bm25 = build_bm25(chunks)

    save_all(chunks, faiss_store, bm25)

    print("\nAMAQA ingestion complete!")


if __name__ == "__main__":
    run_ingestion()
