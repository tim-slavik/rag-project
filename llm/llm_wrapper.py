import openai

class LLMWrapper:
    """
    Simple wrapper arounf LLM provider
    """

    def __init__(self, model = "gpt-4o-mini"):
        self.model = model 

    def __call__(self, prompt: str) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages = [
                    {"role": "user", "content": prompt}
                ],
                temprature=0.2,
            )
            return response["choices"][0]["messages"]["content"]

        except Exception as e:
            return f"LLM error {e}"
