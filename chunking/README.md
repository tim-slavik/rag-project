# Chunking Module

The `chunking/` module provides the document‑chunking logic used during ingestion for the RAG system. Chunking determines how raw text is split into smaller, retrievable units that can be embedded, indexed, and searched efficiently.

This module currently includes:
- a simple orchestrator (`chunker.py`)
- two chunking strategies (`FixedSizeChunking`, `SemanticChunking`)
- lightweight utilities for text cleaning and token counting (`utils.py`)

The design is intentionally minimal and dependency‑free, suitable for a small‑scale RAG system while remaining extensible for future enhancements.

## Files

### `chunker.py`
The orchestrator for chunking.

Responsibilities:
- Accept a chunking strategy instance
- Clean the input text using `clean_text`
- Apply the strategy’s `.chunk()` method
- Return a list of chunk dictionaries

This keeps the pipeline flexible: any strategy implementing `.chunk(text)` can be plugged in.

### `strategies.py`
Contains the actual chunking strategies.

#### `FixedSizeChunking`
Characteristics:
- Splits text into fixed‑size token windows
- Supports overlap between chunks
- Produces deterministic chunk IDs (`chunk_000`, `chunk_001`, …)
- Includes basic metadata (start, end, overlap, token_count)

This strategy is simple, predictable, and ideal for small‑scale RAG ingestion.

#### `SemanticChunking`
Characteristics:
- Uses NLTK sentence tokenization
- Groups sentences until a token limit is reached
- Produces chunk IDs based on a hash of the text
- Includes metadata (strategy, token_count)

This strategy produces more coherent chunks for narrative or structured documents.

### `utils.py`
Shared utilities for chunking.

#### `clean_text(text)`
Normalizes whitespace and strips leading/trailing spaces.  
Ensures consistent chunk boundaries across strategies.

#### `count_tokens(text)`
Simple whitespace‑based token counter.  
Keeps chunking logic lightweight and strategy‑agnostic.

## How It Fits in the System

Chunking is part of the ingestion pipeline:

Raw Document  
↓  
clean_text()  
↓  
Chunker(strategy).run()  
↓  
Chunk dictionaries  
↓  
Embedding generation  
↓  
FAISS index  
↓  
Hybrid retrieval


Chunk quality directly affects retrieval quality and grounding.

## Responsibilities

The chunking module is responsible for:
- Splitting documents into meaningful units
- Providing multiple chunking strategies
- Producing deterministic, metadata‑rich chunks
- Remaining lightweight and dependency‑free (aside from NLTK)

## Not Responsible For

The chunking module does NOT:
- Generate embeddings
- Build FAISS indexes
- Perform retrieval
- Fuse results
- Call the LLM
- Handle synthesis or prompting

Those responsibilities belong to other modules.

## Extensibility

This module is designed to support future enhancements such as:
- sliding‑window chunking
- paragraph‑aware chunking
- model‑based semantic segmentation
- metadata‑aware chunking (document ID, section, versioning)
- adaptive chunk sizes based on content density

Adding a new strategy only requires implementing `.chunk(text)`.

## Testing

Chunking behavior is indirectly validated through:
- ingestion tests
- retrieval tests
- pipeline integration tests

Optional future tests could include:
- fixed‑size chunk boundary tests
- semantic chunking sentence grouping tests
- metadata correctness tests
- text cleaning normalization tests

These can be added later if you expand the module.
