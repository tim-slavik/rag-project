from typing import List, Dict

class PromptBuilder:
    """
    Builds a grounded prompt for the LLM using:
    - the user query
    - the top-k retrieved chunks

    This ensures consistent, deterministic prompt formatting.
    """

    def build_prompt(self, query, context_chunks):
        """
        context_chunks: list of dicts with keys:
            - text
            - metadata
            - doc_id
            - score
        """

        context_blocks = []

        for chunk in context_chunks:
            meta = chunk["metadata"]

            # Extract useful metadata fields if present
            chunk_id = meta.get("chunk_id", "N/A")
            sender = meta.get("sender", None)
            chat_name = meta.get("chat_name", None)
            timestamp = meta.et("timestamp", None)
            token_count = meta.get("token_count", None)

            # Build metadata header
            header_parts = [f"chunk_id={chunk_id}"]

            if sender:
                header_parts.append(f"sende{sender}")
            if chat_name:
                header_parts.append(f"chat={chat_name}")
            if timestamp:
                header_parts.append(f"time={timestamp}")
            if token_count:
                header_parts.append(f"tokens={token_count}")

            header = " | ".join(header_parts)

            block = f"[{header}]\n{chunk['text']}"
            context_blocks.append(block)

        context_section = "\n\n".join(context_blocks)

        prompt = (
            "You are a helpful assistant.  Use context below to answer the question.\n\n"
            "Context:\n"
            f"{context_section}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Answer:"
        )

        return prompt