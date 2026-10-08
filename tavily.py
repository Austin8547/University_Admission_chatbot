from src.ragchain.tools import tavily_search

result = tavily_search.invoke({
    "query": "University of Kerala MSc Data Science admission"
})

print(result)