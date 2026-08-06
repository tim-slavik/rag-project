import json
from pathlib import Path
from rag.engine import RAGEngine


# -------------------------------------------------
# Load Golden Questions
# -------------------------------------------------

def load_golden_questions(path="evaluation/golden_questions.json"):
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------------
# Retrieval Metrics
# -------------------------------------------------

def compute_retrieval_metrics(chunks, expected_keywords):
    """
    Measures how well retrieval matches expected content.
    """

    retrieved_texts = " ".join(c["text"].lower() for c in chunks)

    hits = sum(1 for kw in expected_keywords if kw.lower() in retrieved_texts)
    total = len(expected_keywords)

    precision = hits / len(chunks) if chunks else 0
    recall = hits / total if total > 0 else 1  # if no keywords expected, recall = 1

    return {
        "precision": precision,
        "recall": recall,
        "hits": hits,
        "expected": total
    }


# -------------------------------------------------
# Grounding Metrics
# -------------------------------------------------

def compute_grounding(answer, expected_keywords):
    """
    Measures whether the answer uses expected context.
    """

    answer_lower = answer.lower()

    grounded_hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    grounded = grounded_hits > 0 if expected_keywords else True

    return {
        "grounded": grounded,
        "grounded_hits": grounded_hits,
        "expected_keywords": expected_keywords
    }


# -------------------------------------------------
# Hallucination Metrics
# -------------------------------------------------

HALLUCINATION_MARKERS = [
    "not mentioned",
    "no information",
    "not in the context",
    "the documents do not say",
    "not found"
]

def compute_hallucination(answer, expected_keywords):
    """
    Detects hallucinations by checking if the answer invents facts
    when expected_keywords is empty.
    """

    answer_lower = answer.lower()

    if expected_keywords:
        # If keywords exist, hallucination is harder to detect automatically
        return {"hallucination": False}

    # If no keywords expected, answer should contain a "not mentioned" style phrase
    hallucination = not any(marker in answer_lower for marker in HALLUCINATION_MARKERS)

    return {"hallucination": hallucination}


# -------------------------------------------------
# Evaluate a Single Question
# -------------------------------------------------

def evaluate_question(engine, q):
    question = q["question"]
    expected_keywords = q.get("expected_keywords", [])

    result = engine.answer(question)
    answer = result["answer"]
    chunks = result["chunks_used"]

    retrieval = compute_retrieval_metrics(chunks, expected_keywords)
    grounding = compute_grounding(answer, expected_keywords)
    hallucination = compute_hallucination(answer, expected_keywords)

    return {
        "id": q["id"],
        "question": question,
        "answer": answer,
        "retrieval": retrieval,
        "grounding": grounding,
        "hallucination": hallucination
    }


# -------------------------------------------------
# Main Evaluation Runner
# -------------------------------------------------

def run_evaluation():
    print("Loading golden questions...")
    questions = load_golden_questions()

    print("Building test-mode RAG engine...")
    # You can swap this for AMAQA if you want full-scale evaluation
    engine = RAGEngine.from_documents([
        "Foxes are wild animals found in forests.",
        "Dogs are common domestic animals.",
        "Neural networks are used for machine learning.",
        "Forests contain many wild animals including foxes.",
        "Machine learning models can classify images."
    ])

    print("Running evaluation...")
    results = [evaluate_question(engine, q) for q in questions]

    output_path = Path("evaluation/report.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Evaluation complete. Report saved to {output_path}")


if __name__ == "__main__":
    run_evaluation()
