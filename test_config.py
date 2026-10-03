# test_config.py
from src.config import GROQ_API_KEY, EMBEDDING_MODEL, LLM_MODEL, CHROMA_PATH, DATA_DIR

print("=== CONFIGURATION CHECK ===")
print(f"GROQ_API_KEY: {'[LOADED SUCCESSFULLY]' if GROQ_API_KEY else '[MISSING]'}")
print(f"Embedding Model: {EMBEDDING_MODEL}")
print(f"LLM Model:       {LLM_MODEL}")
print(f"Data Directory:  {DATA_DIR}")
print(f"Chroma Path:     {CHROMA_PATH}")