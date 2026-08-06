import json
from pathlib import Path
from openai import OpenAI

client = OpenAI()


# -------------------------------------------------
# LLM-as-Judge Prompt
# -------------------------------------------------

JUDGE_PROMPT = """
You are an evaluation model. Your job is to score the quality of an answer
produced by a RAG system.

You will receive:
- The question asked
- The answer produced by the RAG system
- The retrieved context chunks
- The expected keywords (if any)

Evaluate the answer on the following criteria:

1. Correctness (0–1):
   Does the answer correctly address the question?

2. Grounding (0–1):
   Does the answer rely on the retrieved context rather than inventing facts?

3. Completeness (0–1):
   Does the answer cover the important information?

4. Clarity (0–1):
   Is the answer easy to understand?

5. Hallucination Risk (0–1):
   Does the answer contain unsupported claims?
   (1 = no hallucination, 0 = hallucination)

Return ONLY a JSON object with the following fields:
{
  "correctness": float,
  "grounding": float,
  "completeness": float,
  "clarity": float,
  "hallucination": float
}
"""


# -------------------------------------------------
# Judge a Single Answer
# -------------------------------------------------

def judge_answer(question, answer, chunks, expected_keywords):
    """
    Uses GPT-4o-mini (or any model you choose) to evaluate the answer.
    """

    context_text = "\n".join([c["text"] for c in chunks])

    payload = {
        "question": question,
        "answer": answer,
        "context": context_text,
        "expected_keywords": expected_keywords
    }

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": json.dumps(payload)}
        ]
    )

    raw = response.choices[0].message.content

    try:
        scores = json.loads(raw)
    except Exception:
        # fallback if model returns non-JSON
        scores = {
            "correctness": 0.5,
            "grounding": 0.5,
            "completeness": 0.5,
            "clarity": 0.5,
            "hallucination": 0.5
        }

    return scores


# -------------------------------------------------
# Run Judge on Full Evaluation Report
# -------------------------------------------------

def run_judge():
    print("Loading evaluation results...")
    results_path = Path("evaluation/report.json")
    results = json.load(open(results_path))

    judged = []

    print("Running LLM-as-judge scoring...")
    for r in results:
        scores = judge_answer(
            question=r["question"],
            answer=r["answer"],
            chunks=r["retrieval"]["chunks"] if "chunks" in r["retrieval"] else r["chunks_used"],
            expected_keywords=r["grounding"]["expected_keywords"]
        )

        judged.append({
            "id": r["id"],
            "question": r["question"],
            "answer": r["answer"],
            "judge_scores": scores
        })

    output_path = Path("evaluation/judge_report.json")
    json.dump(judged, open(output_path, "w"), indent=2)

    print(f"Judge report saved to {output_path}")


if __name__ == "__main__":
    run_judge()
