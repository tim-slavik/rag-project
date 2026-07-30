from typing import List, Dict

class Synthesizer:
    def __init__(self, llm):
        self.llm = llm

    def build_prompt(self, query: str, docs: List[Dict]):
        context = "\n".join([f"Chunk {i+1}:\n{d['text']}" for i,d in enumerate(docs)])

        return f"""
You are a helpful assistant.  Answer the user's question using ONLY the information in the provided context.
If the context does not contain the answer, say "The context does not contain enough information."

Question:
{query}

Context:
{context}

Answer:
"""

    def synthesize(self, query: str, docs= List[Dict]):
        prompt = self.build_prompt(query, docs)
        response = self.llm(prompt)
        return response