import pytest

from rag.engine import RAGEngine


# -------------------------------------------------
# Fixtures
# -------------------------------------------------

@pytest.fixture
def rag_engine():
    """
    Build a small, test-friendly RAG engine using the new
    RAGEngine.from_documents() constructor.
    This avoids loading AMAQA and keeps tests fast.
    """
    docs = [
        "Foxes are wild animals found in forests.",
        "Dogs are common domestic animals.",
        "Neural networks are used for machine learning."
    ]
    return RAGEngine.from_documents(docs)


# -------------------------------------------------
# Grounding Tests
# -------------------------------------------------

def test_llm_uses_context(rag_engine):
    """
    The LLM should reference information from retrieved context.
    """
    result = rag_engine.answer("What does the document say about foxes?")
    answer = result["answer"]
    chunks = result["chunks_used"]

    # Ensure context contains fox-related text
    assert any("fox" in c["text"].lower() for c in chunks)

    # Ensure answer references foxes
    assert "fox" in answer.lower() or "foxes" in answer.lower()


def test_llm_does_not_hallucinate(rag_engine):
    """
    If the context does not contain relevant information,
    the LLM should NOT invent facts.
    """
    result = rag_engine.answer("What does the document say about unicorns?")
    answer = result["answer"]

    hallucination_markers = [
        "not mentioned",
        "no information",
        "not in the context",
        "the documents do not say",
        "not found"
    ]

    assert any(marker in answer.lower() for marker in hallucination_markers)


def test_llm_respects_context_limits(rag_engine):
    """
    The LLM should not reference information outside the retrieved chunks.
    """
    result = rag_engine.answer("Tell me about machine learning.")
    answer = result["answer"]
    chunks = result["chunks_used"]

    # Context should contain ML doc
    assert any("machine learning" in c["text"].lower() for c in chunks)

    # Answer should reference ML
    assert "machine learning" in answer.lower()

    # Answer should NOT reference foxes or dogs
    assert "fox" not in answer.lower()
    assert "dog" not in answer.lower()


def test_rag_engine_returns_correct_structure(rag_engine):
    """
    Ensure the RAG engine returns the correct response structure.
    """
    result = rag_engine.answer("foxes")

    assert isinstance(result, dict)
    assert "query" in result
    assert "answer" in result
    assert "chunks_used" in result

    assert isinstance(result["answer"], str)
    assert isinstance(result["chunks_used"], list)
    assert all("text" in c for c in result["chunks_used"])
    assert all("metadata" in c for c in result["chunks_used"])
