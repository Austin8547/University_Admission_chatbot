import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP

CHUNK_PATH = "/home/austin/agentic/University_Admission_chatbot/data"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)


def split_documents(documents):
    """Split documents into clean text chunks."""
    chunks = text_splitter.split_documents(documents)
    return chunks


def save_chunks_to_json(chunks, filename="chunks.json", output_dir=CHUNK_PATH):
    """Convert Document chunks to dictionary format and save as JSON."""
    # Ensure target directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct full target file path
    file_path = os.path.join(output_dir, filename)
    
    chunks_data = [
        {
            "page_content": chunk.page_content,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=4)
        
    print(f"Successfully saved {len(chunks)} chunks to {file_path}")