from langchain_chroma import Chroma
from src.config import CHROMA_PATH
from src.embeddings.embedding import embeddings  # Updated import


def store_in_chroma(chunks):
    """
    Store document chunks in ChromaDB using local SentenceTransformer embeddings.
    """
    print("Storing embeddings in ChromaDB...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )

    print(f"ChromaDB persisted at: {CHROMA_PATH}")
    print(f"Total documents in Chroma: {vectorstore._collection.count()}")

    return vectorstore