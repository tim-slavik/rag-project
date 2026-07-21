import argparse
import numpy as np

from rag.engine import RAGEngine
from pipeline.orchestrator import PipelineOrchestrator
from llm.prompt_builder import PromptBuilder
from llm.llm_wrapper import LLMWrapper

from hybrid_search.hybrid_retriever import HybridRetriever
from vectorstore.faiss_store import FaissStore
from reranking.cross_encoder_reranker import CrossEncoderReranker


def fake_embed_fn(query:str) -> np.ndarray:
    """
    Simple deterministic embedding function for CLI demo.
    Replace with real embedding model later.
    """
    rng = np.random.default_rng(abs(hash(query)) % (2**32))
    return rng.random(8)

def load_demo_documents():
    """
    Demo documents for CLI usage.
    Replace with real ingestion pipeline later.
    """
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Neural networks are used for machine learning.",
        "A fast brown fox leaps over sleeping dogs.",
        "Foxes are wild animals found in forests."
    ]

def build_retriever(documents):
    """
    Build a HybridRetriever with FAISS and BM25
    """

    # Fake embeddings for demo
    embeddings = np.array([np.random.rand(8) for _ in documents])

    vector_store = FaissStore(dim=8)
    vector_store.add(embeddings)

    return HybridRetriever(
        documents=documents,
        vector_store=vector_store,
        fusion_method="rrf",
        rrf_k=60,
    )

def build_rag_engine():
    """
    Build the full RAG pipeline for CLI usage.
    """

    documents = load_demo_documents()
    retriever = build_retriever(documents)

    reranker = CrossEncoderReranker(seed=595)

    orchestrator = PipelineOrchestrator(
        embed_fn=fake_embed_fn,
        retriever=retriever,
        use_reranker=True,
        final_k=3,
    )

    prompt_builder= PromptBuilder()

    llm = LLMWrapper(lambda prompt: "This is a placeholder LLM answer.")

    return RAGEngine(
        orchestrator=orchestrator,
        prompt_builder=prompt_builder,
        llm=llm,
    )

def main():
    parser = argparse.ArgumentParser(description="Mini-RAG CLI")
    parser.add_argument("query", type=str, help="Your Question")

    args = parser.parse_args()

    engine = build_rag_engine()
    result = engine.answer(args.query)

    print("\n=== RAG Answer ===\n")
    print(result["answer"])
    print("\n=== Context Used ===\n")
    for chunk in result["context"]:
        print(f'- {chunk["text"]} (score={chunk["score"]:.4f})')

if __name__ == "__main__":
    main()