from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

tavily_search = TavilySearch(
    max_results=2,
    topic="general",
    search_depth="basic",
    include_answer=False,
)