import torch
from sentence_transformers import CrossEncoder

_RERANKER = None

def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazy loader for CrossEncoder reranker model (cached in memory)."""
    global _RERANKER
    if _RERANKER is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _RERANKER = CrossEncoder(model_name, device=device)
    return _RERANKER


def rerank_documents(query: str, docs: list, top_n: int = 3, return_score: bool = True):
    
    if not docs:
        return ([], -999.0) if return_score else []

    try:
        reranker = get_reranker()
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)
        
        # Sort documents descending by score
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scored_docs[:top_n]]
        top_score = float(scored_docs[0][1]) if scored_docs else -999.0
        
        if return_score:
            return top_docs, top_score
        return top_docs
    except Exception as e:
        print(f"Reranking failed: {e}. Falling back to raw retrieved docs.")
        if return_score:
            return docs[:top_n], 0.0
        return docs[:top_n]