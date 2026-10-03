import json
from crewai import Agent, Task, Crew
from tools.llm_config import get_llm

llm = get_llm()

verifier_agent = Agent(
    role="Fact Verifier",
    goal="Judge whether each claim is supported by the single quote given with it, and output only JSON.",
    backstory=(
        "You are a rigorous fact-checker. Each claim comes with one quote from the source. "
        "You judge ONLY whether that quote, by itself, supports the claim. "
        "VERIFIED: the quote literally says it. "
        "PARTIALLY SUPPORTED: the claim adds detail, interpretation or a conclusion beyond the quote, "
        "turns an example or recommendation into a capability, or states promotional wording "
        "('leading', 'enterprise-ready', 'production-ready') as fact instead of as what the "
        "documentation says. "
        "UNSUPPORTED: the quote does not support the claim. "
        "Every quote comes from the CrewAI documentation, so the subject of a quote (CrewAI, its agents, "
        "its crews) is implied. Do not mark a claim UNSUPPORTED only because the quote does not name CrewAI. "
        "You never use outside knowledge. When unsure, choose the weaker verdict. "
        "You output only the JSON list you are asked for, with no other text."
        "Every fact in the claim must appear in the quote. If the claim adds a fact, name or scope that the "
        "quote does not contain, even if it is true elsewhere, choose PARTIALLY SUPPORTED. "
        "Leaving words out, or phrasing the claim differently, is not a reason to downgrade it. "
    ),
    llm=llm,
    max_iter=3,
    verbose=True
)

if __name__ == "__main__":
    # Expected verdicts: 1 VERIFIED, 2 PARTIALLY, 3 PARTIALLY, 4 UNSUPPORTED
    test_claims = json.dumps([
        {"id": 1, "claim": "CrewAI agents can delegate tasks when allowed.",
         "quote": "Delegate tasks when allowed"},
        {"id": 2, "claim": "CrewAI is the standard for enterprise-ready AI automation.",
         "quote": "CrewAI is the standard for enterprise-ready AI automation"},
        {"id": 3, "claim": "CrewAI integrates with SerperDevTool for internet search.",
         "quote": "search_tool = SerperDevTool()"},
        {"id": 4, "claim": "CrewAI supports Anthropic models.",
         "quote": "manager_llm - The language model used by the manager agent"},
    ], indent=1)

    test_task = Task(
        description=(
            "Check each claim against the single quote given with it. Judge ONLY whether "
            "that quote, by itself, supports the claim.\n\n"
            "CLAIMS:\n" + test_claims + "\n\n"
            'Output ONLY a JSON list: [{"id": 1, "verdict": "VERIFIED", "reason": "one short line"}]'
        ),
        expected_output="A JSON list of {id, verdict, reason} objects and nothing else.",
        agent=verifier_agent
    )

    crew = Crew(agents=[verifier_agent], tasks=[test_task], verbose=True)
    print("\n--- RESULT ---")
    print(crew.kickoff())