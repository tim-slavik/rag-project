import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel

from rag.engine import RAGEngine
from llm.prompt_builder import PromptBuilder
from llm.llm_wrapper import LLMWrapper

from hybrid_search.hybrid_retriever import HybridRetriever
from vectorstore.faiss_store import FaissStore
from reranking.cross_encoder_reranker import CrossEncoderReranker

# ------------------------
# Request\Response Models
# ------------------------

class RAGRequest(BaseModel):
    query: str

class RAGResponse(BaseModel):
    query: str
    answer: str
    context: list
    prompt: str



# ------------------------
# Demo Embeddings and Docs
# ------------------------

def fake_embed_fn(query: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(query)) % (2**32))
    return rng.random(8)

def load_demo_documents():
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Neural networks are used for machine learning.",
        "A fast brown fox leaps over sleeping dogs.",
        "Foxes are wild animals found in forests."
    ]

def build_retriever(documents):
    embeddings = np.array([np.random.rand(8) for _ in documents])

    vector_store = FaissStore(dim=8)
    vector_store.add(embeddings)

    return HybridRetriever(
        documents=documents,
        vector_store=vector_store,
        fusion_method="rrf",
        rrf_k=60,
    )


# ------------------------
# Build RAG Engine
# ------------------------

def build_rag_engine():
    import json
    import pickle
    import numpy as np
    from sentence_transformers import SentenceTransformer

    # ------------------------
    # Load AMAQA artifacts
    # ------------------------

    faiss_store = FaissStore.load("data/amaqa.index")
    embeddings = np.load("data/amaqa_embeddings.npy")

    with open("data/amaqa_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    with open("data/amaqa_text.json") as f:
        documents = json.load(f)

    with open("data/amaqa_metadata.json") as f:
        metadata = json.load(f)

    # ------------------------
    # Build retriever
    # -----------------------

    retriever = HybridRetriever(
        documents=documents,
        vector_store=faiss_store,
        bm25=bm25,
        fusion_method="rrf",
        rrf_k=60,
        metadata=metadata,
    )

    # ------------------------
    # Real emdedding model
    # ------------------------
    embed_model =SentenceTransformer("sentence-transformer/all-MiniLM-L6-v2")
    embed_fn = lambda q: embed_model.encode([q], convert_to_numpy=True)

    # ------------------------
    # Reranker
    # ------------------------
    reranker = CrossEncoderReranker()

    orchestrator = PipelineOrchestrator(
        embed_fn=embed_fn,
        retriever=retriever,
        reranker=reranker,
        use_reranker=True,
        final_k=5,
    )

    # ------------------------
    # Prompt builder
    # ------------------------
    prompt_builder = PromptBuilder()
    llm = LLMWrapper(lambda prompt: "This is a placeholder LLM answer")

    return RAGEngine(
        orchestrator=orchestrator,
        prompt_builder=prompt_builder,
        llm=llm
    )


# ------------------------  
# FastAPI api
# ------------------------

app = FastAPI(title="Mini-RAG API", version="1.0.0")
engine=build_rag_engine()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/rag", response_model=RAGResponse)
def rag_endpoint(payload: dict):
    query = payload.get("query", "")
    if not query:
        return {"error": "Missing 'query' field in request body."}

    engine = build_rag_engine()
    result = engine.answer(query)

    return result

@app.get("/debug/context")
def debug_context(query: str):
    engine = build_rag_engine()

    # Run retrieval only (no LLM)
    results = engine.orchestrator.run(query)

    # Return raw retrieval output
    return {
        "query":query,
        "chunks":results
    }

