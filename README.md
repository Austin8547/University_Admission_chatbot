# Kerala University Admission Assistant

An intelligent, token-optimized RAG (Retrieval-Augmented Generation) chatbot designed to assist students with admission queries, course eligibility, syllabi, and schedules for the University of Kerala.

---

## What Is It?

A multi-turn conversational AI system that answers university admission questions using official institutional documents (UG/PG prospectuses, syllabi, timetables, and FAQs). If local documents do not contain the answer, it dynamically falls back to live web search to deliver accurate and cited answers.

---

## Key Features

- **Hybrid Retrieval & Reranking**: Vector retrieval via ChromaDB (`BAAI/bge-base-en-v1.5`) paired with Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`) for high precision.
- **Dynamic Web Search Fallback**: Seamless integration with Tavily Search when university documents lack context, returning up-to-date web citations.
- **Smart Conversational Memory**: Preserves multi-turn session context with automatic summarization middleware to prevent token overflow.
- **Fast-Path Greeting Handling**: Responds to greetings instantly without unnecessary LLM or search tool calls.
- **Dual Interfaces**:
  - **Streamlit App**: Interactive chat UI featuring session history, memory compression status, and reset controls.
  - **FastAPI Backend**: REST API endpoint (`/api/chat`) for seamless external integration.
- **Source Citations**: Clearly references local document excerpts and web sources.

---

## What's Inside?

```text
University_Admission_chatbot/
├── data/                  # Knowledge base documents
│   ├── info/              # General FAQs, contact details, and credentials
│   ├── pg/                # PG prospectuses & timetables (PDFs)
│   └── ug/                # UG prospectuses & syllabus documents
├── chroma_db_data/        # Persistent Chroma vector store
├── src/
│   ├── ingestion/         # Document loading, text chunking & vector storage
│   ├── embeddings/        # HuggingFace / BAAI embedding model setup
│   ├── retriever/         # Chroma retrieval & document source citation formatting
│   ├── ragchain/          # RAG pipeline, reranker, memory middleware & Tavily tool
│   └── config.py          # Centralized configuration & environment settings
├── streamlit_app.py       # Streamlit web interface
├── app.py                 # FastAPI REST API server
├── main.py                # Data ingestion script to build the vector database
├── test_rag.py            # Quick test script for query verification
└── requirements.txt       # Project dependencies
```

---

## Tech Stack

- **LLM**: Groq (`qwen/qwen3.8-27b`)
- **Embeddings**: HuggingFace BAAI (`BAAI/bge-base-en-v1.5`)
- **Reranker**: Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
- **Vector Database**: ChromaDB
- **Web Search**: Tavily Search API
- **Frameworks**: LangChain, Streamlit, FastAPI, Uvicorn

---

## Setup & Usage

### 1. Installation

```bash
git clone https://github.com/Austin8547/RAG.git
cd University_Admission_chatbot
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### 3. Ingest Documents (Build Vector Database)

Load and index documents from `data/` into ChromaDB:

```bash
python main.py
```

### 4. Run Applications

- **Streamlit Web UI**:
  ```bash
  streamlit run streamlit_app.py
  ```
  Accessible at `http://localhost:8501`.

- **FastAPI Server**:
  ```bash
  python app.py
  ```
  Or using Uvicorn:
  ```bash
  uvicorn app:app --reload
  ```
  Accessible at `http://localhost:8000` (API documentation at `http://localhost:8000/docs`).

- **Test the RAG Chain**:
  ```bash
  python test_rag.py
  ```
