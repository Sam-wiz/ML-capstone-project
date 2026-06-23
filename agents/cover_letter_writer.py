"""
agents/cover_letter_writer.py — Agent 3: Cover Letter Writer

Responsibilities:
- Retrieves company/industry context from RAG
- Uses fit score data and tailored CV to write a targeted cover letter
- Runs validate_cover_letter tool as a quality gate
- If validation fails, attempts one self-correction pass
- Outputs structured CoverLetterOutput
"""

import json
from state import GraphState, CoverLetterOutput
from tools.tools import validate_cover_letter
from rag.rag_setup import retrieve_context
from agents.json_utils import clean_json_response
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


SYSTEM_PROMPT = """You are an expert cover letter writer with 15 years of recruitment experience.

Write cover letters that are:
- Specific to the role and company (no generic phrasing)
- Concise: 3-4 paragraphs, 250-350 words total
- Evidence-based: reference actual achievements from the resume
- Compelling: open with a hook, not "I am writing to apply for..."
- Professional but human in tone

NEVER use: "I am a team player", "I am passionate about", "I am a quick learner",
"I would be a great fit", "To whom it may concern".

Output ONLY valid JSON — no markdown fences, no preamble."""


def cover_letter_writer_node(state: GraphState) -> dict:
    """Write a personalised cover letter."""
    log = list(state.get("agent_log", []))
    log.append("COVER_LETTER: Starting cover letter generation.")

    jd = state["job_description"]
    resume = state["raw_resume"]
    role_title = state.get("role_title", "the role")
    company_name = state.get("company_name", "the company")

    fit_data = state.get("fit_score", {})
    strengths = fit_data.get("strengths", [])
    score = fit_data.get("score", 50)

    tailored_cv_data = state.get("tailored_cv", {})
    summary = tailored_cv_data.get("summary", "")
    top_bullets = tailored_cv_data.get("tailored_bullets", [])[:3]

    # ── RAG: retrieve cover letter best practices + industry context ──────────
    cl_context = retrieve_context("cover letter best practices writing tips")
    industry_context = retrieve_context(f"{role_title} key skills requirements")
    log.append("COVER_LETTER: RAG context retrieved (best practices + industry).")

    # ── Build prompt ──────────────────────────────────────────────────────────
    schema = CoverLetterOutput.model_json_schema()

    prompt = f"""
Role: {role_title}
Company: {company_name}
Fit Score: {score}/100
Candidate Strengths: {strengths}

Job Description:
{jd[:1200]}

Candidate's Tailored Summary:
{summary}

Top Achievements from Resume:
{chr(10).join(f'- {b}' for b in top_bullets)}

Full Resume (for reference):
{resume[:1000]}

Cover Letter Best Practices (from knowledge base):
{cl_context}

Industry Context:
{industry_context}

Write a personalised, compelling cover letter for this application.
Output ONLY valid JSON matching this schema:
{json.dumps(schema, indent=2)}

The full_text field should be the complete assembled letter ready to copy-paste,
including salutation and sign-off.
"""

    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_retries=3)

        def generate_letter(messages) -> CoverLetterOutput:
            response = llm.invoke(messages)
            raw = clean_json_response(response.content)
            return CoverLetterOutput.model_validate_json(raw)

        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
        parsed = generate_letter(messages)
        log.append("COVER_LETTER: Initial draft generated.")

        # ── Tool call: validate quality ───────────────────────────────────────
        validation = validate_cover_letter.invoke({"cover_letter_text": parsed.full_text})
        log.append(f"COVER_LETTER: Validation — passed={validation['passed']}, words={validation['word_count']}.")

        # ── Self-correction if validation fails (up to 2 retries) ────────────
        for attempt in range(2):
            if validation["passed"]:
                break
            log.append(f"COVER_LETTER: Issues (attempt {attempt + 1}): {validation['issues']}. Self-correcting.")
            correction_prompt = f"""
The cover letter you wrote has these issues:
{chr(10).join(f'- {i}' for i in validation['issues'])}

Please fix these issues and regenerate the full cover letter.
The original draft was:
{parsed.full_text}

Output ONLY valid JSON matching the same schema.
"""
            messages.append(HumanMessage(content=correction_prompt))
            try:
                parsed = generate_letter(messages)
                validation = validate_cover_letter.invoke({"cover_letter_text": parsed.full_text})
                log.append(f"COVER_LETTER: Self-correction attempt {attempt + 1} — passed={validation['passed']}.")
            except Exception as e:
                log.append(f"COVER_LETTER: Self-correction attempt {attempt + 1} failed ({e}), keeping current draft.")
                break

        log.append("COVER_LETTER: Cover letter finalised.")
        return {
            "cover_letter": parsed.model_dump(),
            "agent_log": log,
        }

    except Exception as e:
        log.append(f"COVER_LETTER: ERROR — {e}")
        fallback_text = f"""Dear Hiring Manager,

I am interested in the {role_title} role at {company_name} because the position aligns with the experience highlighted in my resume and the requirements described in the job posting.

My background includes relevant project work, collaboration, and delivery responsibilities that can be reviewed in the tailored CV section above. The fit analysis should be used to confirm which strengths are strongest and which gaps still need human review before this application is submitted.

I would welcome the opportunity to discuss how my experience could support the team. Thank you for your time and consideration.

Sincerely,
[Your Name]"""
        fallback = CoverLetterOutput(
            subject_line=f"Application for {role_title} at {company_name}",
            opening_paragraph=f"I am interested in the {role_title} role at {company_name} because the position aligns with the experience highlighted in my resume and the requirements described in the job posting.",
            body_paragraph_1="My background includes relevant project work, collaboration, and delivery responsibilities that can be reviewed in the tailored CV section above.",
            body_paragraph_2="The fit analysis should be used to confirm which strengths are strongest and which gaps still need human review before this application is submitted.",
            closing_paragraph="I would welcome the opportunity to discuss how my experience could support the team. Thank you for your time and consideration.",
            full_text=fallback_text,
        )
        return {
            "cover_letter": fallback.model_dump(),
            "agent_log": log,
            "error": str(e),
        }
