import os

def format_docs_with_sources(docs: list, max_chars_per_doc: int = 400) -> str:
    """Format documents with clear source labels and page numbers for the LLM with token bounding."""
    formatted_blocks = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "")
        page_info = f" (Page {page})" if page else ""
        content = doc.page_content.replace("\n", " ").strip()
        if len(content) > max_chars_per_doc:
            content = content[:max_chars_per_doc].rstrip() + "..."
        
        formatted_blocks.append(f"[Source {i}]: {source}{page_info}\n{content}")
        
    return "\n\n".join(formatted_blocks)