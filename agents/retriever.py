from crewai import Agent, Task, Crew
from crewai.tools import tool
from tools.llm_config import get_llm
from tools.vector_store import VectorStore

llm = get_llm()
store = VectorStore()

CHUNK_LOG: list[str] = []   # raw chunk text from every search, used by crew_config.py
MIN_RELEVANCE = 0.1         # drop weak matches


@tool("Document Search Tool")
def search_documents(query: str) -> str:
    """Searches the vector store for document chunks relevant to the given query.
    Returns the top matching text chunks with their relevance."""
    results = store.query(query, n_results=3)
    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    kept = [(doc, 1 - dist) for doc, dist in zip(docs, distances) if 1 - dist >= MIN_RELEVANCE]

    if not kept:
        return "No relevant documents found."

    CHUNK_LOG.extend(doc for doc, _ in kept)
    return "\n\n".join(f"[relevance: {rel:.2f}] {doc}" for doc, rel in kept)


retriever_agent = Agent(
    role="Document Retriever",
    goal="Find the most relevant document chunks for a given sub-query using the document search tool.",
    backstory=(
        "You are a precise, efficient information retriever. Given a specific "
        "research sub-query, you search the document store ONCE with a clear, "
        "well-formed query and return the most relevant excerpts. You do not "
        "re-search with reworded variations of the same query - one well-formed "
        "search call is sufficient. You never fabricate content."
    ),
    tools=[search_documents],
    llm=llm,
    max_iter=3,
    verbose=True
)

if __name__ == "__main__":
    test_task = Task(
        description="Search for information relevant to: 'What is CrewAI used for?'",
        expected_output="The most relevant document excerpts found, with brief context on why each is relevant.",
        agent=retriever_agent
    )

    crew = Crew(agents=[retriever_agent], tasks=[test_task], verbose=True)
    result = crew.kickoff()
    print("\n--- RESULT ---")
    print(result)