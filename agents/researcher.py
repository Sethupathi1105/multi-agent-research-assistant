from crewai import Agent, Task, Crew
from tools.llm_config import get_llm

llm = get_llm()

researcher_agent = Agent(
    role="Researcher",
    goal="Synthesize retrieved document excerpts into clear, well-organized draft claims that answer the research sub-query.",
    backstory=(
        "You are a careful research synthesizer. Given raw excerpts retrieved from "
        "documents, you organize them into coherent draft claims — clearly stating "
        "what the evidence supports. You never add information beyond what the "
        "excerpts actually say, and you flag when the excerpts are insufficient "
        "to fully answer the question."
    ),
    llm=llm,
    max_iter=3,
    verbose=True
)

if __name__ == "__main__":
    test_excerpts = (
        "[relevance: 0.89] CrewAI is a multi-agent framework.\n"
        "[relevance: 0.30] Docker packages apps into containers."
    )

    test_task = Task(
        description=(
            f"Based on the following retrieved excerpts, write draft claims that answer "
            f"the question 'What is CrewAI used for?'\n\nExcerpts:\n{test_excerpts}"
        ),
        expected_output="A short list of draft claims, each grounded in the given excerpts, with no fabricated information.",
        agent=researcher_agent
    )

    crew = Crew(agents=[researcher_agent], tasks=[test_task], verbose=True)
    result = crew.kickoff()
    print("\n--- RESULT ---")
    print(result)