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


def rerank_documents(query: str, docs: list, top_n: int = 5) -> list:
    """Reranks retrieved documents based on cross-encoder similarity scores."""
    if not docs:
        return []

    try:
        reranker = get_reranker()
        pairs = [[query, doc.page_content] for doc in docs]
        scores = reranker.predict(pairs)
        
        # Sort documents descending by score
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_docs[:top_n]]
    except Exception as e:
        print(f"Reranking failed: {e}. Falling back to top 3 raw retrieved docs.")
        return docs[:3]