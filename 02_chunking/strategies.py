import nltk
from utils import count_tokens

# Ensure sentence tokenizer is available
nltk.download("punkt", quiet=True)
nltk.download('punkt_tab', quiet=True)

class FixedSizeChunking:
    """
    Simple fixed-size chunking based on token count.
    Splits text into chunks of `size` tokens with optional `overlap`.
    """
# I am only building a small scale rag for now, so I am using very simple chunk-level metadata.
# In a large scale version some additional metadata would be useful, such as the parent document id, section, embedding model version, etc.
# I would also need to update my documents themselves to include a unique id, and possibly a version number and other measures to ensure metadata does not get stale.
    
    def __init__(self, size=400, overlap=50):
        self.size = size
        self.overlap = overlap

    def chunk(self, text: str):
        tokens = text.split()
        chunks = []
        start = 0

        while start < len(tokens):
            end = start + self.size
            chunk_tokens = tokens[start:end]
            chunk_text = " ".join(chunk_tokens)

            chunks.append({
                "id": f"chunk_{len(chunks):03d}",
                "text": chunk_text,
                "metadata": {
                    "strategy": "fixed",
                    "start": start,
                    "end": end,
                    "overlap": self.overlap,
                    "token_count": len(chunk_tokens)
                }
            })

            start += self.size - self.overlap

        return chunks


class SemanticChunking:
    """
    Sentence-aware chunking using NLTK.
    Groups sentences until max_tokens is reached.
    """

    def __init__(self, max_tokens=400):
        self.max_tokens = max_tokens

    def chunk(self, text: str):
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current = []

        for sentence in sentences:
            current.append(sentence)
            combined = " ".join(current)

            if count_tokens(combined) >= self.max_tokens:
                chunks.append(self._make_chunk(combined))
                current = []

        if current:
            chunks.append(self._make_chunk(" ".join(current)))

        return chunks

    def _make_chunk(self, text: str):
        return {
            "id": f"chunk_{hash(text) % 10000:04d}",
            "text": text,
            "metadata": {
                "strategy": "semantic",
                "token_count": count_tokens(text)
            }
        }
