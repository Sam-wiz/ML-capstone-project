"""
state.py — Shared graph state + all Pydantic output models.

Every agent reads from and writes to GraphState.
Pydantic models enforce structured handoffs between agents.
"""

from typing import Optional, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Pydantic output models (structured handoffs)
# ─────────────────────────────────────────────

class KeywordAnalysis(BaseModel):
    matched_keywords: list[str] = Field(description="Keywords present in both JD and resume")
    missing_keywords: list[str] = Field(description="Keywords in JD but absent from resume")
    bonus_keywords: list[str] = Field(description="Strong resume keywords not in JD but relevant")


class FitScoreOutput(BaseModel):
    score: int = Field(ge=0, le=100, description="Overall fit score 0-100")
    keyword_analysis: KeywordAnalysis
    strengths: list[str] = Field(description="Top 3 reasons this candidate fits")
    gaps: list[str] = Field(description="Top 3 gaps or concerns")
    recommendation: Literal["strong_fit", "moderate_fit", "low_fit"]
    reasoning: str = Field(description="1-2 sentence summary of the scoring rationale")


class TailoredCVOutput(BaseModel):
    summary: str = Field(description="Rewritten professional summary targeting this role")
    tailored_bullets: list[str] = Field(description="Rewritten experience bullets (max 6) with JD keywords woven in")
    skills_section: list[str] = Field(description="Prioritised skills list matching JD requirements")
    changes_made: list[str] = Field(description="Brief list of what was changed and why")


class CoverLetterOutput(BaseModel):
    subject_line: str = Field(description="Email subject line for the application")
    opening_paragraph: str
    body_paragraph_1: str = Field(description="Relevant experience paragraph")
    body_paragraph_2: str = Field(description="Why this company / role paragraph")
    closing_paragraph: str
    full_text: str = Field(description="Complete assembled cover letter")


class HumanFeedback(BaseModel):
    decision: Literal["approve", "edit", "reject"]
    edited_cv: Optional[str] = None
    edited_cover_letter: Optional[str] = None
    feedback_notes: Optional[str] = None


# ─────────────────────────────────────────────
# Main graph state
# ─────────────────────────────────────────────

class GraphState(TypedDict):
    # ── Inputs ──
    job_description: str
    raw_resume: str
    company_name: str          # extracted or provided
    role_title: str            # extracted or provided

    # ── Guardrail ──
    guardrail_passed: bool
    guardrail_message: str

    # ── Agent outputs (Pydantic models serialised to dicts) ──
    fit_score: Optional[dict]           # FitScoreOutput.model_dump()
    tailored_cv: Optional[dict]         # TailoredCVOutput.model_dump()
    cover_letter: Optional[dict]        # CoverLetterOutput.model_dump()

    # ── HITL ──
    human_feedback: Optional[dict]      # HumanFeedback.model_dump()
    awaiting_human: bool

    # ── Routing ──
    routing_decision: Optional[str]     # "proceed" | "low_fit_warning" | "abort"

    # ── Final output ──
    final_package: Optional[str]        # assembled markdown

    # ── Debug / trace ──
    agent_log: list[str]
    error: Optional[str]
