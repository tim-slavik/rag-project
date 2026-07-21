from typing import Any, Dict

class LLMWrapper:
    """
    Minimal LLM wrapper.
    This class abstracts awat the underlying model so the RAG engine
    can call 'generate(prompt)' without caring about the backend.

    Replace the implementation of 'genreate()' with your actual LLM call.
    """

    def __init__(self, model_fn):
        """
        model_fn: a callable that takes a prompt and returns generated text.
        Example:
            model_fn(prompt: str) -> str
        """
        self.model_fn = model_fn
    
    def generate(self, prompt: str) -> str:
        """
        Generate text from the LLM using the provided prompt.
        """
        return self.model_fn(prompt)