from src.ingestion.load_docs import load_all_documents
from src.ingestion.split_doc import split_documents
from src.ingestion.store_chroma import store_in_chroma

def main():
    # 1 Load all documents
    print("Loading documents...")
    all_docs = load_all_documents()
    print(f"Total documents loaded: {len(all_docs)}")

    # 2️ Split documents into chunks
    print("Splitting documents into chunks...")
    chunks = split_documents(all_docs)
    print(f"Total chunks created: {len(chunks)}")

    # ️3 Store in Chroma
    print("Creating Chroma vector database...")
    vectorstore = store_in_chroma(chunks)
    print("Vector database creation complete!")

if __name__ == "__main__":
    main()