import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY not found! Add it inside your .env file.")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

if TAVILY_API_KEY is None:
    raise ValueError("TAVILY_API_KEY not found! Add it inside your .env file.")

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

LLM_MODEL = "qwen/qwen3.8-27b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 60

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db_data")

# Data Paths
INFO_DIR = os.path.join(DATA_DIR, "info")
PG_DIR = os.path.join(DATA_DIR, "pg")
UG_DIR = os.path.join(DATA_DIR, "ug")