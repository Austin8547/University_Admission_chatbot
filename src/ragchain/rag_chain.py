import re
from langchain_groq import ChatGroq

from src.retriever.chroma_retriever import get_chroma_retriever
from src.ragchain.reranker import rerank_documents
from src.retriever.metadata import format_docs_with_sources
from src.ragchain.tools import tavily_search
from src.ragchain.memory_context import get_memory_context, clear_memory_context
from src.config import LLM_MODEL, GROQ_API_KEY


# Cached LLM instance to avoid repeated connection handshakes
_LLM_CLIENT = None

def get_llm():
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = ChatGroq(
            model=LLM_MODEL,
            groq_api_key=GROQ_API_KEY,
            temperature=0.3,
            max_tokens=600  # Bounds output token count to prevent TPM spikes
        )
    return _LLM_CLIENT


GREETING_WORDS = {"hi", "hello", "hey", "good morning", "good evening", "greetings", "thanks", "thank you", "bye", "who are you"}

def is_greeting(query: str) -> bool:
    clean = re.sub(r"[^\w\s]", "", query.strip().lower())
    return clean in GREETING_WORDS or (len(clean.split()) <= 2 and any(w in clean.split() for w in ["hi", "hello", "hey", "thanks"]))


def _search_web_and_answer(query: str, history_section: str, llm: ChatGroq) -> str:
    """Executes Tavily web search and generates a concise answer using retrieved web snippets."""
    try:
        print(f"[RAG -> Tavily] Searching web for: {query}")
        web_results = tavily_search.invoke({"query": query})
        results_list = web_results.get("results", [])

        if not results_list:
            return "I couldn't find sufficient information in the university documents or online to answer your question."

        # Keep only top 2 web results with 250-character snippets to conserve prompt tokens
        web_context = ""
        for i, result in enumerate(results_list[:2], start=1):
            title = result.get("title", "")
            url = result.get("url", "")
            snippet = result.get("content", "")[:250].strip()
            web_context += f"[Web Source {i}] Title: {title}\nURL: {url}\nSnippet: {snippet}\n\n"

        prompt = f"""You are a Kerala University Admissions Counselor.
Answer accurately using ONLY the Web Search Results below. Cite [Web Source X].
If the answer is not in the web results, reply: "I don't have enough information to answer that question."
Be clear, structured, and concise.

{history_section}Web Search Results:
{web_context}

Question: {query}
Answer:"""

        response = llm.invoke(prompt)
        return response.content

    except Exception as e:
        print(f"[RAG Tavily Error] Web search failed: {e}")
        return "I couldn't find enough information in the local university documents, and web search is currently unavailable."


def run_chain(query: str, session_id: str = "default") -> str:
    """
    Executes the token-optimized RAG pipeline:
    1. Fast-path greetings (zero API search waste, ~50 tokens).
    2. Local Chroma retrieval + CrossEncoder rerank (top 3 chunks).
    3. Conditional Tavily web search only when local docs are missing/insufficient.
    4. Compact prompt templates with bounded history and document snippets (~500-600 tokens).
    """
    memory = get_memory_context(session_id)
    # Bound history to last 4 messages (2 turns) and truncate past AI answers to 200 chars
    history_context = memory.get_context_string(max_recent_messages=4, max_ai_chars=200)
    history_section = (
        f"Conversation History:\n{history_context}\n\n"
        if history_context
        else ""
    )

    llm = get_llm()

    # 1. Fast-path: Greetings / Chit-chat (0 Tavily calls, ~50 tokens)
    if is_greeting(query):
        reply = "Hello! I am the Kerala University Admissions Assistant. How can I help you with courses, eligibility, or admissions today?"
        memory.add_interaction(user_query=query, ai_response=reply)
        return reply

    # 2. Fetch top candidates from Chroma & rerank (top 3 chunks)
    base_retriever = get_chroma_retriever(k=8)
    docs = base_retriever.invoke(query)
    top_docs, top_score = rerank_documents(query, docs, top_n=3, return_score=True)

    # 3. Check Condition 1: No document matches the query
    if not top_docs or top_score < 0.0:
        print(f"[RAG] No relevant local document matched (top_score={top_score:.2f}). Using Tavily fallback.")
        final_answer = _search_web_and_answer(query, history_section, llm)
        memory.add_interaction(user_query=query, ai_response=final_answer)
        return final_answer

    # 4. Local documents matched! Format with token bounding (max 400 chars per doc)
    context_text = format_docs_with_sources(top_docs, max_chars_per_doc=400)

    local_prompt = f"""You are an Admissions Counselor for Kerala University.
Answer accurately using ONLY the provided Local University Documents. Cite [Source X].
If information is missing to answer the question, reply with ONLY: INSUFFICIENT_LOCAL_CONTEXT

{history_section}Local University Documents:
{context_text}

Question: {query}
Answer:"""

    local_response = llm.invoke(local_prompt)
    local_answer = local_response.content.strip()

    # 5. Check Condition 2: LLM couldn't answer from local documents
    if "INSUFFICIENT_LOCAL_CONTEXT" in local_answer or "not have enough information" in local_answer.lower():
        print("[RAG] Local documents insufficient for query. Using Tavily fallback.")
        final_answer = _search_web_and_answer(query, history_section, llm)
    else:
        # Local documents answered successfully! Tavily was NEVER called.
        final_answer = local_answer

    memory.add_interaction(user_query=query, ai_response=final_answer)
    return final_answer