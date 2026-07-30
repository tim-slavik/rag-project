import requests
import json

API_URL = "http://localhost:8000/rag"

def ask(query: str):
    payload = {"query": query}
    response = requests.post(API_URL, json=payload)

    try:
        data = response.json()
    except Exception:
        print("Server returned non-JSON response:")
        print(response.text)
        return

    print("\n===Answer===")
    print(data.get("answer", ""))

    print("\n===CONTEXT CHUNKS ===")
    for chunk in data.get("context", []):
        print(f"- doc_id={chunk['doc_id']} score={chunk['score']:.4f}")
        print(f"  text: {chunk['text'][:200]}...")
        print(f"  metadata: {json.dumps(chunk['metadata'], indent=2)}")
        print()

    print("\n=== PROMPT SENT TO LLM ===")
    print(data.get("prompt", ""))


if __name__ == "__main__":
    print("RAG CLI Client")
    print("Type a query, or 'exit' to quit.\n")

    while True:
        query = input("Query> ").strip()
        if query.lower() in ("exit", "quit"):
            break

        ask(query)