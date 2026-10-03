from crewai import Task
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.verifier import verifier_agent
from agents.writer import writer_agent


def plan_task(question: str) -> Task:
    return Task(
        description=(
            f"Break down this research question into 3-4 focused, specific "
            f"sub-queries: '{question}'. Use only terms that appear in the question. "
            "Do not introduce other products, frameworks or competitors. "
            "Each sub-query must cover a DIFFERENT aspect (for example: what it is, how it is "
            "structured, what it is used for). Do not rephrase the same question. "
            "Output ONLY a numbered list, one sub-query per line."
        ),
        expected_output="A numbered list of 3-4 specific sub-queries.",
        agent=planner_agent,
    )


def research_task(question: str, evidence: str, sub_queries: list[str]) -> Task:
    plan = "\n".join(f"{i}. {q}" for i, q in enumerate(sub_queries, 1))
    return Task(
        description=(
            f"Research question: '{question}'\n\n"
            f"SUB-QUERIES:\n{plan}\n\n"
            f"EVIDENCE (each chunk has an ID):\n{evidence}\n\n"
            "Write draft claims answering the question using ONLY this evidence.\n"
            'Output ONLY a JSON list. Each item: {"claim": "...", "sub_query": 1, "chunk_id": "C1", "quote": "..."}\n'
            "Rules:\n"
            "- sub_query: the number of the sub-query the claim answers. If the evidence does not "
            "answer a sub-query, write no claim for it.\n"
            "- quote: copied exactly, character for character, from the chunk named by chunk_id (max 30 words). "
            "Prefer a full sentence that includes its subject.\n"
            "- claim: say no more than the quote says. No added adjectives, comparisons or conclusions.\n"
            "- Never complete a sentence that is cut off in a chunk.\n"
            "- Marketing wording (leading, enterprise-ready, production-ready, standard) must be phrased "
            "as 'The documentation describes CrewAI as ...'.\n"
            "- Only for actual code snippets (for example SerperDevTool()), phrase the claim as "
            "'A code example shows ...'. Rows in the 'Use Case - Architecture' list are use cases the "
            "documentation lists, not code examples.\n"
            "- No text outside the JSON list."
        ),
        expected_output="A JSON list of {claim, sub_query, chunk_id, quote} objects and nothing else.",
        agent=researcher_agent,
    )


def verify_task(claims_json: str) -> Task:
    return Task(
        description=(
            "Check each claim against the single quote given with it. Judge ONLY whether "
            "that quote, by itself, supports the claim.\n\n"
            f"CLAIMS:\n{claims_json}\n\n"
            "Verdicts:\n"
            "- VERIFIED: everything the claim states is in the quote. A claim that is narrower than the "
            "quote, leaves out words such as 'leading', or starts with 'The documentation describes' "
            "is VERIFIED.\n"
            "- PARTIALLY SUPPORTED: the claim contains a fact, name or scope that is not in the quote "
            "(for example it says 'CrewAI' where the quote says 'CrewAI AMP'), or it states an example "
            "or recommendation as a general capability.\n"
            "- UNSUPPORTED: the quote does not support the claim.\n"
            "Do NOT downgrade a claim for leaving words out or for how it is phrased.\n"
            'Output ONLY a JSON list: [{"id": 1, "verdict": "VERIFIED", "reason": "one short line"}]'
        ),
        expected_output="A JSON list of {id, verdict, reason} objects and nothing else.",
        agent=verifier_agent,
    )


def write_task(question: str, claims_json: str) -> Task:
    return Task(
        description=(
            f"Write a short structured report answering: '{question}'.\n\n"
            f"CHECKED CLAIMS:\n{claims_json}\n\n"
            "Rules:\n"
            "- Use ONLY these claims. Add no facts, adjectives or comparisons of your own.\n"
            "- Do not use words like 'uniquely', 'verified' or 'proven'.\n"
            "- Keep the words 'The documentation describes' exactly as they appear in a claim.\n"
            "- Never state promotional wording (standard, enterprise-ready, leading, production-ready) "
            "as fact. In the Summary, leave it out or begin the sentence with 'The documentation describes'.\n"
            "- Do not merge two claims into one sentence.\n"
            "- Mark every PARTIALLY SUPPORTED claim with '(partially supported)'.\n"
            "- Structure: a 2-3 sentence Summary, then Findings grouped under short headings.\n"
            "- Do NOT write a Gaps or Limitations section; it is added automatically."
        ),
        expected_output="A markdown report with a Summary and Findings.",
        agent=writer_agent,
    )