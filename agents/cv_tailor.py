"""
agents/cv_tailor.py — Agent 2: CV Tailor

Responsibilities:
- Takes the FitScoreOutput (matched/missing keywords) from Agent 1
- Uses rewrite_bullets tool to get rewrite suggestions
- Produces a tailored professional summary, rewritten bullets, and skills section
- Outputs structured TailoredCVOutput

Key principle: never fabricate experience. Rewrite to surface relevant
existing experience more clearly, incorporating JD keywords truthfully.
"""

import json
from state import GraphState, TailoredCVOutput
from tools.tools import rewrite_bullets
from agents.json_utils import clean_json_response
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """You are a professional CV writer and career consultant.

Your task is to tailor a resume to a specific job description.
CRITICAL RULES:
1. Never fabricate or invent experience, skills, or achievements.
2. Only rewrite existing content to highlight relevance to the JD.
3. Incorporate missing keywords from the JD where they can be genuinely inferred
   from the candidate's existing experience.
4. Quantify achievements wherever the original text allows (if numbers exist, use them).
5. Keep bullet points concise: under 20 words each, starting with a strong action verb.
6. Output ONLY valid JSON — no markdown, no explanation."""


def cv_tailor_node(state: GraphState) -> dict:
    """Rewrite resume sections to target the JD."""
    log = list(state.get("agent_log", []))
    log.append("CV_TAILOR: Starting CV tailoring.")

    jd = state["job_description"]
    resume = state["raw_resume"]
    role_title = state.get("role_title", "the role")
    company_name = state.get("company_name", "the company")

    fit_data = state.get("fit_score", {})
    keyword_analysis = fit_data.get("keyword_analysis", {})
    missing_keywords = keyword_analysis.get("missing_keywords", [])
    matched_keywords = keyword_analysis.get("matched_keywords", [])

    log.append(f"CV_TAILOR: Working with {len(missing_keywords)} missing keywords to incorporate.")

    # ── Tool call: get bullet rewrite suggestions ─────────────────────────────
    # Extract rough bullets from resume (simple split on newlines + dash/bullet chars)
    raw_bullets = []
    for line in resume.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        has_bullet_marker = any(c in stripped for c in ["•", "-", "*", "→", "·"])
        looks_like_sentence = len(stripped) > 40 and stripped[0].isupper()
        if (len(stripped) > 30 and has_bullet_marker) or looks_like_sentence:
            raw_bullets.append(stripped.lstrip("•-*→").strip())
    raw_bullets = raw_bullets[:8]

    if not raw_bullets:
        # Fallback: take non-empty lines over 40 chars
        raw_bullets = [
            line.strip() for line in resume.split("\n")
            if len(line.strip()) > 40
        ][:6]

    log.append(f"CV_TAILOR: Extracted {len(raw_bullets)} candidate bullets from resume.")

    rewrite_suggestions = []
    if raw_bullets and missing_keywords:
        try:
            rewrite_suggestions = rewrite_bullets.invoke({
                "original_bullets": raw_bullets[:6],
                "target_keywords": missing_keywords[:8],
                "role_title": role_title,
            })
            log.append(f"CV_TAILOR: Got {len(rewrite_suggestions)} rewrite suggestions from tool.")
        except Exception as e:
            log.append(f"CV_TAILOR: rewrite_bullets tool error (non-fatal): {e}")

    # ── LLM call ──────────────────────────────────────────────────────────────
    schema = TailoredCVOutput.model_json_schema()

    prompt = f"""
Target Role: {role_title} at {company_name}

Job Description:
{jd}

Original Resume:
{resume}

Keyword Analysis:
- Keywords already in resume: {matched_keywords}
- Keywords missing from resume (to incorporate where truthful): {missing_keywords}

Rewrite Suggestions from tool:
{json.dumps(rewrite_suggestions, indent=2) if rewrite_suggestions else "None generated."}

Produce a tailored version of this resume. Output ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}

For tailored_bullets: write 4-6 strong bullets starting with action verbs, weaving in
missing keywords where they can be truthfully inferred. Do NOT invent experience.
For changes_made: briefly explain what you changed and why (2-4 items).
"""

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_retries=3)
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        raw = clean_json_response(response.content)

        parsed = TailoredCVOutput.model_validate_json(raw)
        log.append(f"CV_TAILOR: Produced {len(parsed.tailored_bullets)} tailored bullets.")

        return {
            "tailored_cv": parsed.model_dump(),
            "agent_log": log,
        }

    except Exception as e:
        log.append(f"CV_TAILOR: ERROR — {e}")
        fallback = TailoredCVOutput(
            summary=f"Experienced professional applying for {role_title}.",
            tailored_bullets=raw_bullets[:4] if raw_bullets else ["See original resume."],
            skills_section=matched_keywords[:8] if matched_keywords else [],
            changes_made=[f"Error during tailoring: {e}. Original bullets preserved."],
        )
        return {
            "tailored_cv": fallback.model_dump(),
            "agent_log": log,
            "error": str(e),
        }
