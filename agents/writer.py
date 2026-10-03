from crewai import Agent, Task, Crew
from tools.llm_config import get_llm

llm = get_llm()

writer_agent = Agent(
    role="Report Writer",
    goal="Compile verified claims into a clear, structured final report, clearly distinguishing verified findings from unsupported or partially supported ones.",
    backstory=(
        "You are a precise technical writer. Given a set of claims with their "
        "verification verdicts, you compile them into a clean, readable report. "
        "You present VERIFIED claims as established findings, clearly flag "
        "PARTIALLY SUPPORTED or UNSUPPORTED claims as such (never presenting them "
        "as solid fact), and organize the report with a brief summary followed by "
        "the detailed findings. You do not add new information beyond what was "
        "given to you."
    ),
    llm=llm,
    max_iter=3,
    verbose=True
)

if __name__ == "__main__":
    verified_claims = (
        "- CrewAI is used as a multi-agent framework. VERIFIED - the source excerpt "
        "explicitly states \"CrewAI is a multi-agent framework.\"\n"
        "- CrewAI is the most popular multi-agent framework and is used by thousands "
        "of companies. PARTIALLY SUPPORTED - the source confirms CrewAI is a "
        "multi-agent framework, but offers no evidence of popularity or company adoption."
    )

    test_task = Task(
        description=(
            f"Compile the following verified claims into a final structured report "
            f"answering the question 'What is CrewAI used for?'\n\n"
            f"Verified Claims:\n{verified_claims}"
        ),
        expected_output="A structured report with a brief summary and detailed findings, clearly distinguishing verified vs. unverified information.",
        agent=writer_agent
    )

    crew = Crew(agents=[writer_agent], tasks=[test_task], verbose=True)
    result = crew.kickoff()
    print("\n--- RESULT ---")
    print(result)