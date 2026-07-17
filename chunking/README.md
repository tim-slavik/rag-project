# 02_chunking

## Overview
This module implements text chunking strategies used to prepare documents for embedding and retrieval in the RAG pipeline. Effective chunking improves semantic coherence, reduces hallucinations, and increases retrieval accuracy by ensuring each chunk contains meaningful, context‑preserving information.

## Goals
- Split raw text into structured, retrievable units  
- Support multiple chunking strategies (fixed, semantic, overlapping)  
- Provide reusable utilities for token counting, boundary detection, and cleaning  
- Produce consistent chunk metadata for downstream modules  

## Features
- **Fixed‑size chunking:** simple, predictable chunk boundaries  
- **Semantic chunking:** sentence/paragraph‑aware segmentation  
- **Overlapping windows:** preserves context across chunk boundaries  
- **Configurable parameters:** chunk size, overlap, tokenization method  
- **Chunk metadata:** source, length, boundaries, strategy used  

## Folder Structure
```text
2_chunking/
|-- chunker.py
|-- strategies.py
|-- utils.py
|-- examples/
|---- sample_text.txt
|-- tests/
|---- test_chunker.py
|-- README.md
```

## Usage
# Update when files are complete and structure is stable
```python
from chunker import Chunker
from strategies import FixedSizeChunking

chunker = Chunker(strategy=FixedSizeChunking(size=400, overlap=50))
chunks = chunker.run(text)
```
## Output

{
  "id": "chunk_001",
  "text": "...",
  "metadata": {
    "strategy": "fixed",
    "start": 0,
    "end": 400,
    "overlap": 50
}

## Next steps
This module feeds directly into 03_metadata, where chunk‑level metadata is enriched and prepared for vector storage.