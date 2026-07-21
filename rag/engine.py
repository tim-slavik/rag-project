from typing import Dict, Any

from pipeline.orchestrator import PipelineOrchestrator
from llm.prompt_builder import PromptBuilder
from llm.llm_wrapper import LLMWrapper


class RAGEngine:
    """
    High-level RAG engine that:
    1.  Retrieves grounded context via the orchestrator
    2.  Builds a structured prompt
    3.  Generates an answer using the LLM
    """

    def __init__(
            self,
            orchestrator: PipelineOrchestrator,
            prompt_builder: PromptBuilder,
            llm: LLMWrapper,
    ):
        
        self.orchestrator=orchestrator
        self.prompt_builder=prompt_builder
        self.llm=llm

    def answer(self, query: str) ->  Dict[str, Any]:
        """
        Full RAG pipeline:
        - retrieve context
        - build prompt
        - generate answer
        - return structured output
        """

        # Step 1 retrieve grounded context
        context_chunks = self.orchestrator.run(query)

        # Step 2: build prompt
        prompt = self.prompt_builder.build(query, context_chunks)

        # Step 3: generate answer
        answer_text = self.llm.generate(prompt)

        # Step 4: return structured output
        return {
            "query": query,
            "answer": answer_text,
            "context": context_chunks,
            "prompts": prompt,
        } 