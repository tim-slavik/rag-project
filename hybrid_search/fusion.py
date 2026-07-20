from typing import Dict, List, Tuple

def rrf_fusion(
        bm25_results: Dict[int, float],
        vector_results: Dict[int, float],
        k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Recirocal Rank Fusion (RRF)
    Robust, tuning-free fusion of BM25 and vector search.

    RRF(d) = sum(1 / k + rank_2(d)))
    """

    # Convert scores to rank positions
    bm25_ranked = sorted(bm25_results.items(), key=lambda x: x[1], reverse=True)
    vector_ranked = sorted(vector_results.items(), key=lambda x: x[1], reverse=True)

    ranks = {}

    # BM25 ranks
    for rank, (doc_id, _) in enumerate(bm25_ranked):
        ranks.setdefault(doc_id, 0)
        ranks[doc_id] += 1 / (k + rank + 1)

    # Vector ranks
    for rank, (doc_id, _) in enumerate(vector_ranked):
        ranks.setdefault(doc_id, 0)
        ranks[doc_id] += 1 / (k + rank + 1)

    # Sort by fused score
    fused = sorted(ranks.items(), key=lambda x: x[1]. reverse=True)
    
    return fused


def weighted_fusion(
        bm25_results: Dict[int, float],
        vector_results: Dict[int, float],
        w_bm25: float = 0.5,
        w_vector: float = 0.5
) -> List[Tuple[int, float]]:
    """
    Weighted linear fusion
    score(d) = w_bm25 * bm25_score + w_vector * vector_score
    """
    
    fused_scores = {}

    # Combine scores
    for doc_id, score in bm25_results.items():
        fused_scores.setdefault(doc_id, 0)
        fused_scores[doc_id] += w_bm25 * score

    for doc_id, score in vector_results.items():
        fused_scores.setdefault(doc_id, 0):
        fused_scores[doc_id] += w_vector * score

    # Sort by fused score
    fused = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return fused




