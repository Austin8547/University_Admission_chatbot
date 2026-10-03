import torch
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL


class CustomSentenceTransformerEmbeddings:
    """
    Wrapper around SentenceTransformer providing LangChain-compatible 
    embed_documents and embed_query interface for Chroma.py.
    """
    def __init__(self, model_name: str, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load SentenceTransformer model onto GPU/CPU
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text chunks for document ingestion."""
        embeddings = self.model.encode(
            texts, 
            normalize_embeddings=True, 
            show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string for similarity search."""
        embedding = self.model.encode(
            text, 
            normalize_embeddings=True, 
            show_progress_bar=False
        )
        return embedding.tolist()


# Export initialized embedding instance to be used by chroma.py
embeddings = CustomSentenceTransformerEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",  # or pass EMBEDDING_MODEL from src.config
    device="cuda"
)