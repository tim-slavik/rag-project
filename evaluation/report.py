import json
from pathlib import Path
from datetime import datetime


# -------------------------------------------------
# Load all evaluation artifacts
# -------------------------------------------------

def load_json(path):
    return json.load(open(path, "r"))


def load_all():
    evaluation = load_json("evaluation/report.json")
    metrics = load_json("evaluation/metrics_report.json")
    judge = load_json("evaluation/judge_report.json")
    return evaluation, metrics, judge


# -------------------------------------------------
# Build Markdown Report
# -------------------------------------------------

def build_markdown(evaluation, metrics, judge):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md = []
    md.append(f"# RAG Quality Evaluation Report")
    md.append(f"Generated on **{timestamp}**")
    md.append("\n---\n")

    # -----------------------------
    # System-Level Metrics
    # -----------------------------
    md.append("## 📊 System-Level Metrics\n")

    md.append("### Retrieval")
    md.append(f"- **Precision@k:** {metrics['retrieval']['precision_avg']:.3f}")
    md.append(f"- **Recall@k:** {metrics['retrieval']['recall_avg']:.3f}\n")

    md.append("### Grounding")
    md.append(f"- **Grounding Rate:** {metrics['grounding']['grounding_rate']:.3f}\n")

    md.append("### Hallucination")
    md.append(f"- **Hallucination Rate:** {metrics['hallucination']['hallucination_rate']:.3f}\n")

    md.append("### Answer Quality (LLM-as-Judge)")
    md.append(f"- **Score:** {metrics['answer_quality']['answer_quality_score']:.3f}\n")

    md.append("### Overall Quality Score")
    md.append(f"- **Overall Score:** {metrics['overall']['overall_quality_score']:.3f}\n")

    md.append("\n---\n")

    # -----------------------------
    # Per-Question Breakdown
    # -----------------------------
    md.append("## 🧪 Per-Question Evaluation\n")

    judge_map = {j["id"]: j for j in judge}

    for q in evaluation:
        qid = q["id"]
        j = judge_map.get(qid, {})

        md.append(f"### Question {qid}: {q['question']}\n")
        md.append(f"**Answer:** {q['answer']}\n")

        # Retrieval
        md.append("**Retrieval Metrics:**")
        md.append(f"- Precision: {q['retrieval']['precision']:.3f}")
        md.append(f"- Recall: {q['retrieval']['recall']:.3f}")
        md.append(f"- Hits: {q['retrieval']['hits']} / {q['retrieval']['expected']}\n")

        # Grounding
        md.append("**Grounding:**")
        md.append(f"- Grounded: {q['grounding']['grounded']}")
        md.append(f"- Grounded Hits: {q['grounding']['grounded_hits']}\n")

        # Hallucination
        md.append("**Hallucination:**")
        md.append(f"- Hallucinated: {q['hallucination']['hallucination']}\n")

        # Judge Scores
        if j:
            js = j["judge_scores"]
            md.append("**LLM-as-Judge Scores:**")
            md.append(f"- Correctness: {js['correctness']:.3f}")
            md.append(f"- Grounding: {js['grounding']:.3f}")
            md.append(f"- Completeness: {js['completeness']:.3f}")
            md.append(f"- Clarity: {js['clarity']:.3f}")
            md.append(f"- Hallucination: {js['hallucination']:.3f}\n")

        md.append("\n---\n")

    return "\n".join(md)


# -------------------------------------------------
# Write Markdown Report
# -------------------------------------------------

def write_report(md):
    output_path = Path("evaluation/final_report.md")
    with open(output_path, "w") as f:
        f.write(md)
    print(f"Final report written to {output_path}")


# -------------------------------------------------
# Main Runner
# -------------------------------------------------

def run():
    print("Loading evaluation artifacts...")
    evaluation, metrics, judge = load_all()

    print("Building Markdown report...")
    md = build_markdown(evaluation, metrics, judge)

    print("Writing report...")
    write_report(md)


if __name__ == "__main__":
    run()
