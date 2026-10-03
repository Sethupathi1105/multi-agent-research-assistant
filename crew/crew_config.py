import os
from crewai import LLM

PROVIDER = os.getenv("LLM_PROVIDER", "openai")

if PROVIDER == "openai":
    llm = LLM(model="gpt-4o-mini", temperature=0.2)
else:
    llm = LLM(model="groq/<your-current-groq-model>", temperature=0.2)