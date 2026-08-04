import json
import numpy as np
import pickle

from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from openai import OpenAI

from vectorstore.faiss_store import FaissStore
from rag.synthesis import Synthesizer
client = OpenAI()

# -------------------------------------------------
# LLM backend (OpenAI GPT-4o-mini)
# -------------------------------------------------

def default_llm(prompt):
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages = [{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content


class RAGEngine:
    def __init__(self):
        print("Loading RAG engine...")

        # -----------------------------
        # Load FAISS index
        # -----------------------------
        print("Loading FAISS index...")
        self.faiss = FaissStore.load("data/amaqa.index")

        # -----------------------------
        # Load BM25
        # -----------------------------
        print("Loading BM25 index...")
        with open("data/amaqa_bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)

        # -----------------------------
        # Load text + metadata
        # -----------------------------
        print("Loading text + metadata...")
        with open("data/amaqa_text.json", "r") as f:
            self.texts = json.load(f)

        with open("data/amaqa_metadata.json", "r") as f:
            self.metadata = json.load(f)

        # -----------------------------
        # Load embedding model
        # -----------------------------
        print("Loading MiniLM model...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # -----------------------------
        # Adding llm synthesizer
        # -----------------------------
        self.synthesizer = Synthesizer(default_llm)
        
        
        print("RAG engine ready.")

    # -------------------------------------------------
    # Encode query
    # -------------------------------------------------
    def embed_query(self, query: str):
        return self.model.encode([query], convert_to_numpy=True).astype("float32")

    # -------------------------------------------------
    # FAISS search
    # -------------------------------------------------
    def search_faiss(self, query_emb, k=10):
        scores, idx = self.faiss.search(query_emb, k)
        return idx[0], scores[0]

    # -------------------------------------------------
    # BM25 search
    # -------------------------------------------------
    def search_bm25(self, query: str, k=10):
        scores = self.bm25.get_scores(query.split())
        idx = np.argsort(scores)[::-1][:k]
        return idx, scores[idx]

    # -------------------------------------------------
    # Hybrid search (RRF)
    # -------------------------------------------------
    def hybrid_search(self, query: str, k=10):
        query_emb = self.embed_query(query)

        faiss_idx, _ = self.search_faiss(query_emb, k)
        bm25_idx, _ = self.search_bm25(query, k)

        # Reciprocal Rank Fusion
        rrf_scores = {}
        for rank, idx in enumerate(faiss_idx):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank)

        for rank, idx in enumerate(bm25_idx):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (60 + rank)

        # Sort by fused score
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        top_idx = [i for i, _ in ranked[:k]]

        return top_idx

    # -------------------------------------------------
    # Retrieve chunks
    # -------------------------------------------------
    def retrieve(self, query: str, k=10):
        idx = self.hybrid_search(query,k)
        results = []

        for i in idx:
            results.append({
                "text": self.texts[i],
                "metadata": self.metadata[i]
            })
        return results

    # -------------------------------------------------
    # Simple answer synthesis
    # -------------------------------------------------
    def answer(self, query: str, k=5):
        # Retireve top-k chunks using hybrid FAISS + BM25
        docs = self.retrieve(query, k)

        # Use syntesizer (LLM-backed) to generate a grounded answer
        llm_answer = self.synthesizer.synthesize(query, docs)

        # Return full RAG response payload
        return {
            "query": query,
            "answer": llm_answer,
            "chunks_used": docs
        }