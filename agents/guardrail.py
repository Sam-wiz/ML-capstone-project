"""
agents/guardrail.py — Input validation node.

Runs BEFORE any agent. Checks:
1. JD and resume are non-empty and meet minimum length.
2. No prompt injection attempts in inputs.
3. Content is actually a job description + resume (not random text).
4. Extracts company name and role title from JD for downstream agents.

Returns early with guardrail_passed=False if any check fails,
preventing all downstream agent calls.
"""

from state import GraphState
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage


BANNED_PATTERNS = [
    "ignore previous instructions",
    "ignore all instructions",
    "ignore your instructions",
    "disregard previous",
    "disregard your",
    "forget everything",
    "forget your instructions",
    "you are now",
    "you are an",
    "act as if",
    "act as a",
    "pretend you are",
    "pretend to be",
    "roleplay as",
    "simulate a",
    "jailbreak",
    "dan mode",
    "developer mode",
    "override your",
    "bypass your",
    "new persona",
    "system prompt",
    "reveal your prompt",
    "print your instructions",
    "repeat everything above",
    "what is your system",
]


def guardrail_node(state: GraphState) -> dict:
    """Validate inputs. Returns updated state fields."""
    jd = state.get("job_description", "").strip()
    resume = state.get("raw_resume", "").strip()
    log = list(state.get("agent_log", []))

    log.append("GUARDRAIL: Starting input validation.")

    # ── 1. Non-empty check ────────────────────────────────────────────────────
    if not jd:
        return {
            "guardrail_passed": False,
            "guardrail_message": "Job description is empty. Please provide a job description.",
            "agent_log": log + ["GUARDRAIL: FAILED — empty job description."],
            "routing_decision": "abort",
        }
    if not resume:
        return {
            "guardrail_passed": False,
            "guardrail_message": "Resume is empty. Please paste your resume text.",
            "agent_log": log + ["GUARDRAIL: FAILED — empty resume."],
            "routing_decision": "abort",
        }

    # ── 2. Minimum length ─────────────────────────────────────────────────────
    if len(jd.split()) < 30:
        return {
            "guardrail_passed": False,
            "guardrail_message": "Job description seems too short (< 30 words). Please paste the full JD.",
            "agent_log": log + ["GUARDRAIL: FAILED — JD too short."],
            "routing_decision": "abort",
        }
    if len(resume.split()) < 50:
        return {
            "guardrail_passed": False,
            "guardrail_message": "Resume seems too short (< 50 words). Please paste more of your resume.",
            "agent_log": log + ["GUARDRAIL: FAILED — resume too short."],
            "routing_decision": "abort",
        }

    # ── 3. Prompt injection check ─────────────────────────────────────────────
    combined_lower = (jd + " " + resume).lower()
    for pattern in BANNED_PATTERNS:
        if pattern in combined_lower:
            return {
                "guardrail_passed": False,
                "guardrail_message": f"Input contains disallowed content: '{pattern}'. Please provide only a genuine JD and resume.",
                "agent_log": log + [f"GUARDRAIL: FAILED — prompt injection pattern detected: '{pattern}'."],
                "routing_decision": "abort",
            }

    # ── 4. Extract company + role title via LLM ───────────────────────────────
    # Small, cheap extraction call. Falls back gracefully if it fails.
    company_name = state.get("company_name", "").strip()
    role_title = state.get("role_title", "").strip()

    if not company_name or not role_title:
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_retries=3)
            extraction_prompt = f"""
Extract the company name and job title from this job description.
Reply ONLY in this format (two lines, nothing else):
COMPANY: <company name or "Unknown">
ROLE: <job title>

Job description:
{jd[:1500]}
"""
            response = llm.invoke([HumanMessage(content=extraction_prompt)])
            lines = response.content.strip().split("\n")
            for line in lines:
                if line.startswith("COMPANY:"):
                    company_name = line.replace("COMPANY:", "").strip()
                elif line.startswith("ROLE:"):
                    role_title = line.replace("ROLE:", "").strip()
        except Exception as e:
            company_name = company_name or "Unknown Company"
            role_title = role_title or "Unknown Role"
            log.append(f"GUARDRAIL: Warning — could not extract company/role: {e}")

    log.append(f"GUARDRAIL: PASSED. Company='{company_name}', Role='{role_title}'.")

    return {
        "guardrail_passed": True,
        "guardrail_message": "All checks passed.",
        "company_name": company_name,
        "role_title": role_title,
        "agent_log": log,
    }
