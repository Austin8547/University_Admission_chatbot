import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError(" GOOGLE_API_KEY not found! Add it inside your .env file.")


EMBEDDING_MODEL = "models/text-embedding-004"
LLM_MODEL = "gemini-2.5-flash"

CHUNK_SIZE= 600
CHUNK_OVERLAP=80

# paths

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db_data")


# DATA PATHS

INFO_DIR = os.path.join(DATA_DIR, "info")
PG_DIR = os.path.join(DATA_DIR, "pg")
UG_DIR = os.path.join(DATA_DIR, "ug")
