from utils import clean_text


class Chunker:
    """
    Chunker orchestrates the chunking workflow by applying a selected strategy.
    Each strategy must implement a .chunk(text: str) -> List[dict] method.
    """

    def __init__(self, strategy):
        self.strategy = strategy

    def run(self, text: str):
        """
        Clean the text and apply the selected chunking strategy.
        Returns a list of chunk dictionaries.
        """
        cleaned = clean_text(text)
        return self.strategy.chunk(cleaned)