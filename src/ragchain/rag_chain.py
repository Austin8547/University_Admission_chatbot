from langchain_groq import ChatGroq
from src.retriever.chroma_retriever import get_chroma_retriever
from src.ragchain.reranker import rerank_documents
from src.ragchain.metadata import format_docs_with_sources
from src.config import LLM_MODEL, GROQ_API_KEY


def run_chain(query: str) -> str:
    """Executes the full RAG pipeline: retrieval -> reranking -> prompt generation -> Groq LLM response."""
    # 1. Fetch top candidate documents from Chroma
    base_retriever = get_chroma_retriever(k=10)
    docs = base_retriever.invoke(query)

    if not docs:
        return "I couldn't find any relevant documents to answer your question."

    # 2. Rerank docs using CrossEncoder
    top_docs = rerank_documents(query, docs, top_n=5)

    # 3. Format context with source metadata
    context_text = format_docs_with_sources(top_docs)

    # 4. Instantiate ChatGroq LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.3
    )

    system_prompt = f"""
You are an expert Admissions Counselor for Kerala University. Your goal is to provide accurate, helpful, and professional assistance to students.

Instructions:
1. Analyze the Context carefully.
2. Answer based ONLY on the provided Context Documents. Do not use outside knowledge. If the answer is not in the context, say "I don't have that information in my current documents."
3. Cite Sources: When stating facts, reference the source ID (e.g., [Source 1]).
4. Be clear, structured, and use bullet points for steps or lists where applicable.

Context Documents:
{context_text}

User Question: {query}

Answer:
"""

    response = llm.invoke(system_prompt)
    return response.content