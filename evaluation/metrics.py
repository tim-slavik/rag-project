import json
from pathlib import Path
from statistics import mean


# -------------------------------------------------
# Load Evaluation Results
# -------------------------------------------------

def load_results(path="evaluation/report.json"):
    with open(path, "r") as f:
        return json.load(f)


# -------------------------------------------------
# Retrieval Metrics (Aggregate)
# -------------------------------------------------

def compute_retrieval_scores(results):
    precisions = []
    recalls = []

    for r in results:
        precisions.append(r["retrieval"]["precision"])
        recalls.append(r["retrieval"]["recall"])

    return {
        "precision_avg": mean(precisions) if precisions else 0,
        "recall_avg": mean(recalls) if recalls else 0
    }


# -------------------------------------------------
# Grounding Metrics (Aggregate)
# -------------------------------------------------

def compute_grounding_scores(results):
    grounded_flags = []

    for r in results:
        grounded_flags.append(1 if r["grounding"]["grounded"] else 0)

    return {
        "grounding_rate": mean(grounded_flags) if grounded_flags else 0
    }


# -------------------------------------------------
# Hallucination Metrics (Aggregate)
# -------------------------------------------------

def compute_hallucination_scores(results):
    hallucinations = []

    for r in results:
        hallucinations.append(1 if r["hallucination"]["hallucination"] else 0)

    return {
        "hallucination_rate": mean(hallucinations) if hallucinations else 0
    }


# -------------------------------------------------
# Answer Quality (LLM-as-Judge Placeholder)
# -------------------------------------------------

def compute_answer_quality(results):
    """
    Placeholder for LLM-as-judge scoring.
    For now, we return a neutral score.
    """

    return {
        "answer_quality_score": 0.75  # placeholder
    }


# -------------------------------------------------
# Overall Quality Score
# -------------------------------------------------

def compute_overall_score(retrieval, grounding, hallucination, answer_quality):
    """
    Weighted score combining all metrics.
    You can tune weights based on your priorities.
    """

    precision = retrieval["precision_avg"]
    recall = retrieval["recall_avg"]
    grounding_rate = grounding["grounding_rate"]
    hallucination_rate = hallucination["hallucination_rate"]
    answer_quality_score = answer_quality["answer_quality_score"]

    # Example weighting scheme
    score = (
        0.30 * precision +
        0.20 * recall +
        0.25 * grounding_rate +
        0.15 * answer_quality_score +
        0.10 * (1 - hallucination_rate)
    )

    return {
        "overall_quality_score": round(score, 4)
    }


# -------------------------------------------------
# Main Metrics Runner
# -------------------------------------------------

def run_metrics():
    print("Loading evaluation results...")
    results = load_results()

    print("Computing retrieval metrics...")
    retrieval = compute_retrieval_scores(results)

    print("Computing grounding metrics...")
    grounding = compute_grounding_scores(results)

    print("Computing hallucination metrics...")
    hallucination = compute_hallucination_scores(results)

    print("Computing answer quality metrics...")
    answer_quality = compute_answer_quality(results)

    print("Computing overall quality score...")
    overall = compute_overall_score(retrieval, grounding, hallucination, answer_quality)

    final_report = {
        "retrieval": retrieval,
        "grounding": grounding,
        "hallucination": hallucination,
        "answer_quality": answer_quality,
        "overall": overall
    }

    output_path = Path("evaluation/metrics_report.json")
    with open(output_path, "w") as f:
        json.dump(final_report, f, indent=2)

    print(f"Metrics report saved to {output_path}")


if __name__ == "__main__":
    run_metrics()
