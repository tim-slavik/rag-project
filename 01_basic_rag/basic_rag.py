from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os

from rag_utils import text_preprocessing


def build_basic_rag():
    
    # A small configuration dictionary for the scraper and text processing,
    # nothing crazy and practically zero error handling, just a simple hand selected example to get me started.
    
    # config =   {
    #     'base_url' : 'https://www.uspcak9.com/k9-training-articles',
    #     'output_folder' : 'rag',
    #     'output_filename' : 'prepared_docs.json',
    #     'doc_types_to_download' : ['.pdf', '.docx'],
    #     'User-Agent' : 'Mozilla/5.0',        # Comment line for a generic session

    # }

    # I wanted to add a personal touch and I never tried to scrape a website before, so I thought this would be a good opportunity to learn and practice.  I realize there are way better methods but this suffices for this project.
    
    # Run first time and then save output to a file, then comment out the next line and load from the file for subsequent runs.
    # docs = text_preprocessing.get_prepared_docs(config)
    # # Save the prepared docs to a JSON file in the current working directory
    # out_path = os.path.join(os.getcwd(), 'prepared_docs.json')
    # with open(out_path, 'w', encoding='utf-8') as f:
    #     json.dump(docs, f, ensure_ascii=False, indent=2)
    

    with(open(os.path.join(os.getcwd(), 'prepared_docs.json'), 'r', encoding='utf-8')) as f:
        docs = json.load(f)
    
    # Some generic docs for testing.
    # docs = [
    # "Retrieval‑augmented generation improves model reliability by grounding responses in external documents rather than relying solely on parametric memory. A typical RAG pipeline includes preprocessing, chunking, embedding generation, vector storage, retrieval, and answer synthesis. Each stage influences the quality of the final answer, especially chunk size and embedding model selection.",
    # "Chunking determines how text is segmented before embedding. Fixed‑size chunking works well for structured content like manuals, while semantic chunking is better for legal or narrative documents. Overlap between chunks helps preserve context across boundaries, improving retrieval precision for multi‑sentence queries.",
    # "Embedding models vary in speed, dimensionality, and semantic fidelity. Lightweight models such as all‑MiniLM‑L6‑v2 offer fast inference suitable for real‑time applications. Larger models like text‑embedding‑3‑large capture deeper semantic relationships, improving retrieval accuracy for complex queries but requiring more compute.",
    # "Vector search retrieves candidate chunks based on similarity scores, but reranking often improves relevance. Techniques like cross‑encoders, metadata filtering, and domain‑specific heuristics help prioritize the most contextually aligned chunks. High‑quality retrieval reduces hallucinations and strengthens groundedness.",
    # "RAG evaluation focuses on metrics such as retrieval precision, context relevance, and groundedness. Groundedness measures how well generated answers align with retrieved evidence. Production systems often incorporate guardrails, citation requirements, and confidence scoring to ensure factual consistency and reduce unsupported claims."
    # ]



    # load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    # create embeddings
    embeddings = model.encode(docs)

    # build Faiss index
    print(embeddings.shape)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Query
    query = "How to increase stamina & endurance for a working police canine?"
    query_embedding = model.encode([query])

    # Search
    k = 2  # number of nearest neighbors
    distances, indices = index.search(np.array(query_embedding), k)

    # Print results
    print("\nQuery:", query)    
    print("\nTop Results:")
    for idx in indices[0]:
        print("-", docs[idx])
    print(__name__)
if __name__ == "__main__":
    build_basic_rag()
