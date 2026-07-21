from typing import List, Dict

class PromptBuilder:
    """
    Builds a grounded prompt for the LLM using:
    - the user query
    - the top-k retrieved chunks

    This ensures consistent, deterministic prompt formatting.
    """

    def __init__(self, system_instruction: str = None):
        """
        system_instruction:
            Optional system-level guidance fot the llm.
            Example:
                "You are a helpful assistant.  Use ONLY the provided context."
        """
        self.system_instruction = system_instruction or (
            "You are a helpful assistant.  Use ONLY the provided context to answer."
        )

    def build(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Construct the final prompt.

        context_chunks is a list of dicts:
            {
                "doc_id": int,
                "score": float,
                "text": str
            }
        """
        # Build context section
        context_lines = []
        for i,chunk in enumerate(context_chunks, start=1):
            context_lines.append(f'{i}. {chunk["text"]}')

        context_block = "\n".join(context_lines)


        # Final prompt
        prompt = (
            f'{self.system_instruction}\n\n'
            f'Query:\n{query}\n\n'
            f'Context:\n{context_block}\n\n'
            f'Abswer:'
        )

        return prompt