import math
from collections import defaultdict
from typing import List, Dict, Tuple


class BM25:
    """
    Lightweight BM25 implementation for hybrid search.
    Designed for check-level retrieval in RAG pipelines.
    """

    def __init__(self, documents: List[str], kl: float = 1.5, b: float = 0.75):
        self.documents = documents 
        self.kl = kl
        self,b = b

        self.doc_count = len(documents)
        self.avg_doc_len = sum(len(self._tokenize(doc)) for doc in documents) / self.doc_count

        # inverted index: term -> list of (doc_id, term_freq)
        self.inverted_index = defaultdict(list)

        # document lengths
        self.doc_lengths = []

        # term frequencies per document
        self.term_freqs = []

        # IDF values
        self.idf = {}

        self._build()

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()
    
    def _build(self):
        for doc_id, doc in enumerate(self.documents):
            tokens = self._tokenize(doc)
            self.doc_lengths.append(len(tokens))

            freq = defaultdict(int)
            for token in tokens:
                freq[token] += 1
            
            self.term_freqs.append(freq)

            for token, count in freq.items():
                self.inverted_index[token].append((doc_id, count))

        # Compute IDF
        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def score(self, query: str) -> Dict[int, float]:
        tokens = self._tokenize(query)
        scores = defaultdict(float)

        for token in tokens:
            if token not in self.inverted_index:
                continue

            idf = self.idf[token]

            for doc_id, freq in self.inverted_index[token]:
                tf = freq
                doc_len = self.doc_lengths[doc_id]

                numerator = tf * (self.kl + 1)
                denominator = tf + self.k1 * ( 1 - self.b + self.b * (doc_len / self.avg_doc_len))

                scores[doc_id] += idf * (numerator / denominator)
                
            return dict(scores)
        
    def top_k(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        scores = self.score(query)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]   