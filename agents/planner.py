from tools.llm_config import get_llm
from crewai import Agent, Task, Crew

llm = get_llm()

planner_agent = Agent(
    role="Query Planner",
    goal="Break down a broad research question into 3-5 focused, specific sub-queries that together cover the topic thoroughly.",
    backstory=(
        "You are an expert research strategist. You take vague or broad questions "
        "and decompose them into precise, answerable sub-questions that a retrieval "
        "system can effectively search for."
    ),
    llm=llm,
    verbose=True
)

if __name__ == "__main__":
    test_task = Task(
        description="Break down this research question into focused sub-queries: 'How does CrewAI compare to LangGraph for building multi-agent systems?'",
        expected_output="A numbered list of 3-5 specific sub-queries.",
        agent=planner_agent
    )

    crew = Crew(agents=[planner_agent], tasks=[test_task], verbose=True)
    result = crew.kickoff()
    print("\n--- RESULT ---")
    print(result)