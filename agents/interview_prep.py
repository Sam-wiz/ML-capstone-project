"""
agents/interview_prep.py — Interview Prep Agent.

Generates 12 tailored interview questions from the JD + resume:
  3 behavioural   (STAR format)
  3 technical     (stack-specific)
  2 situational   (hypothetical scenarios)
  2 role-specific (culture fit, growth mindset)
  1 company       (why this company)
  1 opener        (tell me about yourself)

Each question includes talking points pulled from the resume,
a difficulty rating, and a concise coaching tip.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from agents.json_utils import clean_json_response
import json


CATEGORIES = {
    "behavioural":  {"label": "Behavioural",  "color": "#eff6ff", "accent": "#2563eb", "icon": "🧠"},
    "technical":    {"label": "Technical",    "color": "#faf5ff", "accent": "#7c3aed", "icon": "⚙️"},
    "situational":  {"label": "Situational",  "color": "#fff7ed", "accent": "#ea580c", "icon": "💡"},
    "role-specific":{"label": "Role-Specific","color": "#f0fdf4", "accent": "#16a34a", "icon": "🎯"},
    "company":      {"label": "Why Company",  "color": "#fef9c3", "accent": "#ca8a04", "icon": "🏢"},
    "opener":       {"label": "Opener",       "color": "#fdf4ff", "accent": "#9333ea", "icon": "👋"},
}


class InterviewQuestion(BaseModel):
    category: str = Field(description="behavioural | technical | situational | role-specific | company | opener")
    question: str
    difficulty: str = Field(description="easy | medium | hard")
    talking_points: list[str] = Field(description="2-3 specific talking points from the candidate's resume")
    sample_answer_structure: str = Field(description="One sentence on how to structure the answer (e.g. STAR format)")
    tip: str = Field(description="One short coaching tip for this specific question")


class InterviewPrepOutput(BaseModel):
    questions: list[InterviewQuestion]
    overall_tip: str
    red_flags_to_avoid: list[str] = Field(description="2-3 common mistakes candidates make for this specific role")


_SYSTEM = """You are a senior technical recruiter and interview coach at a top-tier tech company.
You have conducted 500+ interviews and know exactly what interviewers look for.

Given a job description and candidate resume, generate exactly 12 interview questions the candidate
is VERY LIKELY to face. The breakdown must be:
- 3 behavioural (STAR format, tied to JD requirements)
- 3 technical (specific to the exact tech stack in the JD)
- 2 situational (hypothetical "what would you do if..." tied to the role)
- 2 role-specific (culture, growth mindset, ownership)
- 1 "why this company / why this role"
- 1 "tell me about yourself" opener

IMPORTANT rules:
- Pull SPECIFIC details from the resume (project names, metrics, tools, companies)
- Technical questions must reference the exact technologies in the JD, not generic ones
- talking_points must name actual things from their resume (e.g. "mention your work on X project at Y company")
- difficulty: easy for openers/why-company, medium for behavioural/situational, hard for technical deep-dives
- Return ONLY valid JSON, no markdown fences."""


def run_interview_prep(job_description: str, resume: str, role_title: str = "", company: str = "") -> dict:
    llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, max_retries=3)
    schema = InterviewPrepOutput.model_json_schema()

    prompt = f"""Role: {role_title or 'Not specified'}
Company: {company or 'Not specified'}

JOB DESCRIPTION:
{job_description[:3000]}

CANDIDATE RESUME:
{resume[:3000]}

Generate exactly 12 interview questions. Be specific to THIS role and THIS resume.
Schema: {json.dumps(schema, indent=2)}"""

    resp = llm.invoke([SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)])
    raw  = clean_json_response(resp.content)
    out  = InterviewPrepOutput.model_validate_json(raw)
    return out.model_dump()
