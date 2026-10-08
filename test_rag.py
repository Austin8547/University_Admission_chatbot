import sys
from pathlib import Path

# Ensure project root is in Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.ragchain.rag_chain import run_chain


def main():
    print("=" * 60)
    print("      Testing University Admission Chatbot RAG Chain      ")
    print("=" * 60)

    # Sample query relevant to Kerala University admission prospectus
    test_query = "What are the eligibility criteria for M.Sc. Data Science admission?"

    print(f"\n[Query]: {test_query}\n")
    print("Running retrieval, reranking, and Groq LLM inference...\n")

    try:
        response = run_chain(test_query)
        print("=" * 60)
        print("Response Generated Successfully:")
        print("=" * 60)
        print(response)
        print("=" * 60)
    except Exception as e:
        print(f"\n[ERROR] Pipeline test failed: {e}")


if __name__ == "__main__":
    main()