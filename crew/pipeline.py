import json
import re
import time
from datetime import datetime
from pathlib import Path

from crewai import Crew, Process
from tools.vector_store import VectorStore
from crew.tasks import plan_task, research_task, verify_task, write_task

store = VectorStore()
MIN_RELEVANCE = 0.1
N_RESULTS = 5
SOURCE_NOTE = "All evidence comes from the ingested CrewAI documentation; no independent sources were used."
MARKETING = re.compile(r"\b(leading|enterprise|production[- ]ready|standard for|scalab\w*)", re.I)
DOC_SAYS = re.compile(r"documentation (states|describes|says)", re.I)
CAPABILITY = re.compile(r"\b(allow|allows|support|supports|enable|enables|empower|empowers|provide|provides)\b", re.I)


def _norm(s: str) -> str:
    # letters and digits only: ignores punctuation, bullets and hidden PDF glyphs
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _run(task) -> str:
    crew = Crew(agents=[task.agent], tasks=[task], process=Process.sequential, verbose=False)
    return crew.kickoff().raw


def _json_list(text: str) -> list:
    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        raise ValueError("No JSON list in LLM output:\n" + text[:500])
    return json.loads(m.group(0))


def _parse_sub_queries(text: str) -> list[str]:
    qs = re.findall(r"^\s*\d+[.)]\s*(.+)$", text, flags=re.M)
    return [q.strip().strip("*") for q in qs][:5]


def _unhedged_marketing(body: str) -> list[str]:
    hits = []
    for s in re.split(r"(?<=[.!?])\s+|\n", body):
        if MARKETING.search(s) and not DOC_SAYS.search(s):
            hits.append(s.strip(" *-#"))
    return hits


def retrieve(queries):
    ids_by_text, per_query, scores = {}, {}, {}
    for q in queries:
        res = store.query(q, n_results=N_RESULTS)
        docs = res.get("documents", [[]])[0]
        dists = res.get("distances", [[]])[0]
        hits, sc = [], []
        for doc, dist in zip(docs, dists):
            rel = 1 - dist
            sc.append(round(rel, 2))
            if rel < MIN_RELEVANCE:
                continue
            if doc not in ids_by_text:
                ids_by_text[doc] = f"C{len(ids_by_text) + 1}"
            hits.append(ids_by_text[doc])
        per_query[q] = hits
        scores[q] = sc
    return {cid: text for text, cid in ids_by_text.items()}, per_query, scores


def run_pipeline(question: str) -> dict:
    t0 = time.perf_counter()
    run_dir = Path("runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    def save(name, obj):
        text = obj if isinstance(obj, str) else json.dumps(obj, indent=2, ensure_ascii=False)
        (run_dir / name).write_text(text, encoding="utf-8")

    # 1. plan
    plan_raw = _run(plan_task(question))
    save("1_plan.md", plan_raw)
    sub_queries = _parse_sub_queries(plan_raw)

    # 2. retrieve (plain Python, no LLM). The original question is searched too.
    by_id, per_query, scores = retrieve([question] + sub_queries)
    save("2_chunks.json", {"chunks": by_id, "per_query": per_query, "scores": scores})
    no_hits = [q for q, ids in per_query.items() if not ids]

    kept, dropped, usable = [], [], []
    body = "No relevant evidence was retrieved, so no report could be written."

    if by_id:
        evidence = "\n\n".join(f"[{cid}]\n{txt}" for cid, txt in by_id.items())

        # 3. research
        research_raw = _run(research_task(question, evidence, sub_queries))
        save("3_research_raw.md", research_raw)
        claims = _json_list(research_raw)

        # 4. mechanical quote check
        for c in claims:
            chunk = by_id.get(str(c.get("chunk_id")), "")
            q = _norm(str(c.get("quote", "")))
            (kept if q and q in _norm(chunk) else dropped).append(c)
        for i, c in enumerate(kept, 1):
            c["id"] = i
        save("4_quote_check.json", {"kept": kept, "dropped": dropped})

        if kept:
            # 5. verify (LLM judges claim vs quote only)
            payload = [{"id": c["id"], "claim": c["claim"], "quote": c["quote"]} for c in kept]
            verify_raw = _run(verify_task(json.dumps(payload, indent=1, ensure_ascii=False)))
            save("5_verify_raw.md", verify_raw)
            verdicts = {int(v["id"]): v for v in _json_list(verify_raw)}

            for c in kept:
                v = verdicts.get(c["id"], {})
                verdict = str(v.get("verdict", "UNSUPPORTED")).upper()
                if verdict not in ("VERIFIED", "PARTIALLY SUPPORTED", "UNSUPPORTED"):
                    verdict = "UNSUPPORTED"
                if verdict == "VERIFIED" and MARKETING.search(c["claim"]) and not DOC_SAYS.search(c["claim"]):
                    verdict = "PARTIALLY SUPPORTED"
                if verdict == "VERIFIED" and CAPABILITY.search(c["claim"]) and not CAPABILITY.search(c["quote"]):
                    verdict = "PARTIALLY SUPPORTED"
                c["verdict"], c["reason"] = verdict, v.get("reason", "")
            save("6_claims_checked.json", kept)

            usable = [c for c in kept if c["verdict"] != "UNSUPPORTED"]
            if usable:
                # 6. write
                claims_json = json.dumps(
                    [{"claim": c["claim"], "verdict": c["verdict"]} for c in usable],
                    indent=1, ensure_ascii=False)
                body = _run(write_task(question, claims_json))

    # gaps + evidence table are added by code, not by the writer
    gaps = [SOURCE_NOTE]
    gaps += [f"Promotional wording in the report is not hedged as a documentation claim: \"{s[:120]}\""
             for s in _unhedged_marketing(body)]
    gaps += [f"No relevant chunks found for: {q}" for q in no_hits]

    covered = {int(c["sub_query"]) for c in usable if str(c.get("sub_query", "")).isdigit()}
    gaps += [f"No supported claims for sub-query {i}: {q}"
             for i, q in enumerate(sub_queries, 1) if i not in covered]

    if dropped:
        gaps.append(f"{len(dropped)} drafted claim(s) removed because their quote was not found in the cited chunk.")
    n_unsup = sum(1 for c in kept if c.get("verdict") == "UNSUPPORTED")
    if n_unsup:
        gaps.append(f"{n_unsup} claim(s) removed as UNSUPPORTED by the verifier.")
    n_part = sum(1 for c in usable if c["verdict"] == "PARTIALLY SUPPORTED")
    if n_part:
        gaps.append(f"{n_part} claim(s) are only partially supported by the evidence.")

    table = ["| # | Verdict | Chunk | Claim |", "|---|---|---|---|"]
    table += [f"| {c['id']} | {c['verdict']} | {c['chunk_id']} | {c['claim']} |" for c in usable]

    report = (body.strip()
              + "\n\n## Gaps and Limitations\n" + "\n".join(f"- {g}" for g in gaps)
              + "\n\n## Evidence\n" + "\n".join(table))
    save("final_report.md", report)

    elapsed = time.perf_counter() - t0
    print(f"\nSaved run to {run_dir.resolve()}  |  {elapsed:.0f}s")
    return {"report": report, "claims": usable, "run_dir": str(run_dir), "seconds": elapsed}