import numpy as np

from rag.engine import RAGEngine
from pipeline.orchestrator import PipelineOrchestrator
from llm.prompt_builder import PromptBuilder
from llm.llm_wrapper import LLMWrapper
from hybrid_search.hybrid_retriever import HybridRetriever
from vectorstore.faiss_store import FaissStore
from reranking.cross_encoder_reranker import CrossEncoderReranker

def fake_embed_fn(query: str) -> np.ndarray:
    """
    Deterministic fake embedding function for testing.
    """

    rng = np.random.default_rng(abs(hash(query) % 2**32))
    return rng.random(8)

def fake_llm_fn(prompt: str) -> str:
    """
    Fake LLM function for testing.
    Returns a predictable string so tests are stable.
    """

    return "FAKE_ANSWER"

def build_test_retriever():
    """
    Helper to build a small HybridRetriever for orchestrator tests.
    """

    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over sleeping dogs",
        "Neural networks are used for machine learning",
        "Foxes are wild animals found in forests"
    ]

    embeddings = np,array([
        np.random.rand(8),
        np.random.rand(8),
        np.random.rand(8),
        np.random.rand(8),
    ])

    vector_store = FaissStore(dim=8)
    vector_store.add(embeddings)

    retriever = HybridRetriever(
        documents=documents,
        vector_store=vector_store,
        fusion_method="rrf",
        rrf_k=60,
    )

    return retriever, documents

def test_rag_engine_end_to_end():
    """
    Full RAG pipeline test:
    - orchestrator retrieval
    - prompt building
    - LLM generation
    - final structured output
    """

    retriever, documents = build_test_retriever()

    reranker = CrossEncoderReranker(seed=264)

    orchestrator = PipelineOrchestrator(
        embed_fn=fake_embed_fn,
        retriever=retriever,
        reranker=reranker,
        use_reranker=True,
        final_k=2,
    )

    prompt_builder = PromptBuilder()
    llm = LLMWrapper(fake_llm_fn)

    engine = RAGEngine(
        orchestrator=orchestrator,
        prompt_builder=prompt_builder,
        llm=llm
    )

    result = engine.answer("brown fox")

    # validate structure
    assert isinstance(result, dict)
    assert "query" in result
    assert "answer" in result
    assert "context" in result

    # validate answer
    assert result["answer"] == "FAKE_ANSWER"

    # validate context chunks
    assert len(result["context"]) == 2
    for chunk in result["context"]:
        assert "doc_id" in chunk
        assert "score" in chunk
        assert "text" in chunk
        assert chunk["text"] in documents

    # validate propmpt contains query and context
    prompt = result["prompt"]
    assert "brown fox" in prompt.lower()
    assert "context" in prompt.lower()
    assert "answer" in prompt.lower()
    assert any("fox" in c["text"].lower() for c in result["context"])