"""
agents/fit_scorer.py — Agent 1: Fit Scorer

Responsibilities:
- Compares the resume against the JD
- Extracts matched/missing keywords using the extract_keywords tool
- Retrieves industry context from RAG
- Produces a structured FitScoreOutput (score 0-100, strengths, gaps, recommendation)

Routing output:
- score >= 60 → "proceed"
- score 40-59 → "proceed" (moderate fit, flagged in output)
- score < 40  → "low_fit_warning" (HITL asked whether to continue)
"""

import json
from state import GraphState, FitScoreOutput, KeywordAnalysis
from tools.tools import extract_keywords
from rag.rag_setup import retrieve_context
from agents.json_utils import clean_json_response
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """You are an expert recruitment consultant and career coach.
Your job is to objectively score how well a candidate's resume matches a job description.

Be honest and precise. A score of 70+ means a genuinely strong match.
A score below 40 means significant gaps that would likely result in rejection at screen.

Always base your analysis on evidence in the resume, not assumptions.
Output ONLY valid JSON matching the FitScoreOutput schema — no explanation, no markdown."""


def fit_scorer_node(state: GraphState) -> dict:
    """Score the resume vs JD and output structured FitScoreOutput."""
    log = list(state.get("agent_log", []))
    log.append("FIT_SCORER: Starting analysis.")

    jd = state["job_description"]
    resume = state["raw_resume"]
    role_title = state.get("role_title", "the role")

    # ── Tool call: extract keywords from JD and resume ────────────────────────
    jd_keywords = extract_keywords.invoke({"text": jd})
    resume_keywords = extract_keywords.invoke({"text": resume})

    log.append(f"FIT_SCORER: JD keywords extracted — {len(jd_keywords.get('technical_skills', []))} tech skills found.")
    log.append(f"FIT_SCORER: Resume keywords extracted — {len(resume_keywords.get('technical_skills', []))} tech skills found.")

    # ── RAG: retrieve industry context ────────────────────────────────────────
    rag_context = retrieve_context(f"{role_title} job requirements skills")
    log.append("FIT_SCORER: RAG context retrieved.")

    # ── LLM call ──────────────────────────────────────────────────────────────
    schema = FitScoreOutput.model_json_schema()

    prompt = f"""
Job Description:
{jd}

Candidate Resume:
{resume}

Industry Context (from knowledge base):
{rag_context}

Keyword Analysis Results:
JD technical skills: {jd_keywords.get('technical_skills', [])}
JD soft skills: {jd_keywords.get('soft_skills', [])}
Resume technical skills: {resume_keywords.get('technical_skills', [])}
Resume soft skills: {resume_keywords.get('soft_skills', [])}

Score this candidate against the JD. Output ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}

For matched_keywords: list skills/keywords present in BOTH the JD and resume.
For missing_keywords: list JD requirements NOT found in the resume.
For bonus_keywords: strong resume skills not in JD but valuable for the role.
"""

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_retries=3)
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        raw = clean_json_response(response.content)

        parsed = FitScoreOutput.model_validate_json(raw)
        log.append(f"FIT_SCORER: Score = {parsed.score}, Recommendation = {parsed.recommendation}.")

        # ── Routing decision ──────────────────────────────────────────────────
        if parsed.score < 40:
            routing = "low_fit_warning"
            log.append("FIT_SCORER: Low fit detected — routing to warning node.")
        else:
            routing = "proceed"
            log.append("FIT_SCORER: Fit acceptable — routing to CV tailor.")

        return {
            "fit_score": parsed.model_dump(),
            "routing_decision": routing,
            "agent_log": log,
        }

    except Exception as e:
        log.append(f"FIT_SCORER: ERROR — {e}")
        # Fallback: create a minimal output so the graph doesn't crash
        fallback = FitScoreOutput(
            score=50,
            keyword_analysis=KeywordAnalysis(
                matched_keywords=[], missing_keywords=[], bonus_keywords=[]
            ),
            strengths=["Could not fully analyse — using fallback."],
            gaps=["Analysis failed — review manually."],
            recommendation="moderate_fit",
            reasoning=f"Analysis encountered an error: {e}",
        )
        return {
            "fit_score": fallback.model_dump(),
            "routing_decision": "proceed",
            "agent_log": log,
            "error": str(e),
        }
