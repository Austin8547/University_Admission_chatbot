
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, get_buffer_string
from langchain_groq import ChatGroq
from src.config import LLM_MODEL, GROQ_API_KEY


class SummarizationMiddleware:
    """
    Middleware that monitors conversation length and automatically summarizes
    older chat messages using LangChain and Groq LLM when the message count
    reaches a predefined threshold (default: 10 chats/messages).

    This preserves conversation context while dramatically reducing prompt tokens
    sent to the API, preventing token limit exhaustion and reducing cost.
    """

    def __init__(
        self,
        threshold: int = 10,
        keep_recent: int = 4,
        model_name: str = LLM_MODEL,
        api_key: str = GROQ_API_KEY,
    ):
        self.threshold = threshold
        self.keep_recent = keep_recent
        self.llm = ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=0.2,
        )

    def process(self, memory: "MemoryContext") -> None:
        """
        Executes middleware logic:
        If unsummarized message count >= threshold, summarize older turns
        and keep only the most recent turns in the active message buffer.
        """
        if len(memory.messages) < self.threshold:
            return

        try:
            # Older messages to condense into summary
            messages_to_summarize = (
                memory.messages[:-self.keep_recent]
                if self.keep_recent > 0
                else memory.messages
            )
            recent_messages = (
                memory.messages[-self.keep_recent:]
                if self.keep_recent > 0
                else []
            )

            if not messages_to_summarize:
                return

            conversation_str = get_buffer_string(
                messages_to_summarize,
                human_prefix="Student",
                ai_prefix="Admissions Counselor",
            )

            previous_summary = memory.summary or "None"

            summary_prompt = f"""You are a specialized memory summarizer for a Kerala University Admissions Assistant.
Your task is to produce a concise, information-dense summary of the conversation history.

Existing Summary:
{previous_summary}

New Dialogue to Summarize:
{conversation_str}

Instructions:
1. Merge the existing summary with the new dialogue into a single coherent summary.
2. Focus on key student details, questions asked, degree programs/courses of interest, eligibility criteria discussed, deadlines, and answers provided.
3. Be concise and factual (under 150 words). Do not omit crucial details.
4. Output only the summarized text without preamble.

Summary:"""

            response = self.llm.invoke(summary_prompt)
            new_summary = response.content.strip()

            # Update memory state
            memory.summary = new_summary
            memory.messages = list(recent_messages)
            print(
                f"[MemoryMiddleware] Summarized {len(messages_to_summarize)} older messages to save API tokens."
            )

        except Exception as e:
            print(f"[MemoryMiddleware Warning] Failed to summarize conversation: {e}")


class MemoryContext:
    """
    Manages conversational memory for a chat session.
    Maintains a running summary and recent message history, executing
    registered middlewares on updates.
    """

    def __init__(
        self,
        session_id: str = "default",
        middlewares: Optional[List[SummarizationMiddleware]] = None,
    ):
        self.session_id = session_id
        self.messages: List[BaseMessage] = []
        self.summary: str = ""
        self.middlewares = (
            middlewares
            if middlewares is not None
            else [SummarizationMiddleware(threshold=10, keep_recent=4)]
        )

    def add_user_message(self, content: str) -> None:
        """Add a student message to memory."""
        self.messages.append(HumanMessage(content=content))

    def add_ai_message(self, content: str) -> None:
        """Add counselor/AI response to memory and execute middlewares."""
        self.messages.append(AIMessage(content=content))
        self._run_middlewares()

    def add_interaction(self, user_query: str, ai_response: str) -> None:
        """Convenience method to record a full turn (user + AI) and apply middlewares."""
        self.messages.append(HumanMessage(content=user_query))
        self.messages.append(AIMessage(content=ai_response))
        self._run_middlewares()

    def _run_middlewares(self) -> None:
        """Pass memory through all registered middlewares."""
        for middleware in self.middlewares:
            middleware.process(self)

    def get_context_string(self, max_recent_messages: int = 4, max_ai_chars: int = 250) -> str:
        """
        Formats memory for inclusion into the RAG prompt with strict token boundaries.
        Combines summary with only the most recent conversation turns,
        truncating previous AI responses so they don't cause prompt token bloat.
        """
        blocks = []
        if self.summary:
            blocks.append(f"[Previous Summary]: {self.summary}")

        if self.messages:
            # Keep only the most recent messages (e.g., last 2 student-counselor turns)
            recent = self.messages[-max_recent_messages:]
            condensed = []
            for msg in recent:
                if isinstance(msg, HumanMessage):
                    condensed.append(f"Student: {msg.content}")
                elif isinstance(msg, AIMessage):
                    content = msg.content.strip()
                    if len(content) > max_ai_chars:
                        content = content[:max_ai_chars].rstrip() + "..."
                    condensed.append(f"Counselor: {content}")
                else:
                    condensed.append(f"{msg.content}")

            blocks.append("[Recent Chat]:\n" + "\n".join(condensed))

        return "\n\n".join(blocks)

    def has_history(self) -> bool:
        """Checks if there is any stored history or summary."""
        return bool(self.summary or self.messages)

    def clear(self) -> None:
        """Reset conversation memory and summary."""
        self.messages = []
        self.summary = ""


# Global session registry to manage per-session memory instances
_SESSION_STORE: Dict[str, MemoryContext] = {}


def get_memory_context(session_id: str = "default") -> MemoryContext:
    """Retrieve or create the MemoryContext for a given session."""
    if session_id not in _SESSION_STORE:
        _SESSION_STORE[session_id] = MemoryContext(session_id=session_id)
    return _SESSION_STORE[session_id]


def clear_memory_context(session_id: Optional[str] = None) -> None:
    """Clear memory for a specific session, or all sessions if session_id is None."""
    global _SESSION_STORE
    if session_id is None:
        _SESSION_STORE.clear()
    elif session_id in _SESSION_STORE:
        _SESSION_STORE[session_id].clear()
