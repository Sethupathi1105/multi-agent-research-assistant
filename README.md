# Multi-Agent Research Assistant

A 5-agent research pipeline (planner, retriever, researcher, verifier, writer) built with CrewAI. It answers questions from ingested documentation, checks every claim's quote against the source chunk, and returns a report with a per-claim evidence table.

**Stack:** CrewAI, ChromaDB, FastAPI (async job API), Streamlit, Groq / OpenAI via LiteLLM

## Run locally

1. `pip install -r requirements.txt`
2. Create `.env` with `OPENAI_API_KEY` or `GROQ_API_KEY`, and `LLM_PROVIDER=openai` or `groq`
3. API: `uvicorn api.main:app --port 8000`
4. UI: `streamlit run frontend/app.py`

## Known limitations

- A run takes several minutes on free-tier rate limits
- The knowledge base is two CrewAI docs PDFs
- Not deployed yet