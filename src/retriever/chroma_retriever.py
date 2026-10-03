import os
from langchain_chroma import Chroma
from src.config import CHROMA_PATH
from src.embeddings.embedding import embeddings  # Updated import


def get_chroma_retriever(k=3):
    """
    Load Chroma vectorstore and return a retriever using local BGE embeddings.
    """
    if not os.path.exists(CHROMA_PATH):
        raise FileNotFoundError(f"Chroma DB not found at {CHROMA_PATH}. Run ingestion first!")

    # Load existing Chroma database
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # Return retriever
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )