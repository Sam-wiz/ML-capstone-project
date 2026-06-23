"""
tools/tools.py — LangChain tools used by the agents.

Tools are plain functions wrapped with @tool.
Each agent only gets the tools relevant to its job.
"""

from langchain_core.tools import tool
import re


@tool
def extract_keywords(text: str) -> dict:
    """
    Extract skills, technologies, and role-relevant keywords from a text.
    Returns categorised lists: technical_skills, soft_skills, qualifications, role_keywords.
    Use this on both the job description and the resume to find overlaps.
    """
    # In production: use an LLM call or spaCy NER.
    # For demo/eval: rule-based extractor that works without an extra API call.
    text_lower = text.lower()

    tech_patterns = [
        "python", "java", "javascript", "typescript", "react", "node",
        "sql", "postgres", "mysql", "mongodb", "aws", "gcp", "azure",
        "docker", "kubernetes", "git", "ci/cd", "rest", "api", "graphql",
        "machine learning", "deep learning", "nlp", "data analysis",
        "excel", "tableau", "power bi", "salesforce", "jira", "figma",
        "agile", "scrum", "product management", "ux", "ui",
    ]
    soft_patterns = [
        "communication", "leadership", "teamwork", "problem solving",
        "analytical", "collaborative", "initiative", "adaptable",
        "organised", "detail-oriented", "strategic", "creative",
    ]
    qual_patterns = [
        "bachelor", "master", "phd", "mba", "degree", "certification",
        "certified", "years of experience", "years experience",
    ]

    found_tech = [k for k in tech_patterns if k in text_lower]
    found_soft = [k for k in soft_patterns if k in text_lower]
    found_qual = [k for k in qual_patterns if k in text_lower]

    # Also grab capitalised acronyms (AWS, SQL, etc.)
    acronyms = re.findall(r'\b[A-Z]{2,6}\b', text)
    unique_acronyms = list(set(acronyms))[:10]

    return {
        "technical_skills": found_tech,
        "soft_skills": found_soft,
        "qualifications": found_qual,
        "acronyms": unique_acronyms,
        "word_count": len(text.split()),
    }


@tool
def rewrite_bullets(
    original_bullets: list,
    target_keywords: list,
    role_title: str,
) -> list:
    """
    Given a list of resume bullet points, a list of target keywords from the JD,
    and the role title, return rewrite suggestions that incorporate the keywords
    while preserving factual accuracy.

    This tool returns SUGGESTIONS — the CV Tailor agent decides which to use.
    The LLM agent should call this to get structured suggestions, then synthesise
    them into the final tailored bullets.
    """
    suggestions = []
    keywords_lower = [k.lower() for k in target_keywords]

    for bullet in original_bullets:
        bullet_lower = bullet.lower()
        missing = [k for k in keywords_lower if k not in bullet_lower]

        if not missing:
            suggestions.append({
                "original": bullet,
                "suggestion": bullet,
                "keywords_added": [],
                "note": "Already contains target keywords — keep as is.",
            })
        else:
            top_missing = missing[:2]
            suggestions.append({
                "original": bullet,
                "suggestion": bullet,  # LLM agent rewrites using this hint
                "keywords_added": top_missing,
                "note": f"Consider weaving in: {', '.join(top_missing)}",
            })

    return suggestions


@tool
def validate_cover_letter(cover_letter_text: str) -> dict:
    """
    Basic quality checks on a cover letter.
    Returns a dict with pass/fail flags and suggestions.
    """
    issues = []
    word_count = len(cover_letter_text.split())

    if word_count < 150:
        issues.append("Too short — aim for 250-400 words.")
    if word_count > 500:
        issues.append("Too long — trim to under 400 words.")
    if "dear hiring manager" not in cover_letter_text.lower() and "dear " not in cover_letter_text.lower():
        issues.append("Missing salutation (Dear [Name]/Hiring Manager).")
    if not any(p in cover_letter_text.lower() for p in ["sincerely", "regards", "thank you"]):
        issues.append("Missing closing (Sincerely / Best regards).")

    return {
        "passed": len(issues) == 0,
        "word_count": word_count,
        "issues": issues,
        "suggestion": "Looks good!" if not issues else "Address the issues above before finalising.",
    }
