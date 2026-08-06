import pytest
from rag.engine import RAGEngine


# -------------------------------------------------
# Fixtures
# -------------------------------------------------

@pytest.fixture
def rag_engine():
    """
    Build a small, test-friendly RAG engine using from_documents().
    This tests the full RAG pipeline without loading AMAQA.
    """
    docs = [
        "Foxes are wild animals found in forests.",
        "Dogs are common domestic animals.",
        "Neural networks are used for machine learning.",
        "Forests contain many wild animals including foxes."
    ]
    return RAGEngine.from_documents(docs)


# -------------------------------------------------
# Pipeline Tests
# -------------------------------------------------

def test_pipeline_retrieves_chunks(rag_engine):
    """
    Ensure the pipeline retrieves relevant chunks before synthesis.
    """
    result = rag_engine.answer("foxes")
    chunks = result["chunks_used"]

    assert len(chunks) > 0
    assert any("fox" in c["text"].lower() for c in chunks)


def test_pipeline_synthesizes_answer(rag_engine):
    """
    Ensure the synthesizer produces a non-empty answer.
    """
    result = rag_engine.answer("What does the document say about foxes?")
    answer = result["answer"]

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0


def test_pipeline_respects_grounding(rag_engine):
    """
    Ensure the final answer is grounded in retrieved context.
    """
    result = rag_engine.answer("Tell me about machine learning.")
    answer = result["answer"]
    chunks = result["chunks_used"]

    # Context must contain ML doc
    assert any("machine learning" in c["text"].lower() for c in chunks)

    # Answer must reference ML
    assert "machine learning" in answer.lower()

    # Answer must NOT reference foxes or dogs
    assert "fox" not in answer.lower()
    assert "dog" not in answer.lower()


def test_pipeline_handles_missing_info(rag_engine):
    """
    Ensure the pipeline handles queries with no relevant context.
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


def test_pipeline_output_structure(rag_engine):
    """
    Ensure the full RAG pipeline returns the correct structure.
    """
    result = rag_engine.answer("foxes")

    assert isinstance(result, dict)
    assert "query" in result
    assert "answer" in result
    assert "chunks_used" in result

    assert isinstance(result["query"], str)
    assert isinstance(result["answer"], str)
    assert isinstance(result["chunks_used"], list)

    for chunk in result["chunks_used"]:
        assert "text" in chunk
        assert "metadata" in chunk
