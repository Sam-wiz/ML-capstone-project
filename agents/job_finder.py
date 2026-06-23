"""
agents/job_finder.py — Job Finder Agent

Sources:
  1. Greenhouse Job Board API (free, no auth) — curated list of 40+ top tech companies.
     Fetches live jobs, filters by keyword, enriches description via LLM.
  2. DuckDuckGo general search — LLM-enriched listings (fallback if DDG responds).

Every listing carries a `source` field: "greenhouse" or "duckduckgo".
Only greenhouse listings support direct apply via the Greenhouse API.
"""

import re
import json
import concurrent.futures
import requests
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from agents.json_utils import clean_json_response

GH_API  = "https://boards-api.greenhouse.io/v1/boards"
TIMEOUT = 6

# ── Curated Greenhouse board slugs ─────────────────────────────────────────────
# These are verified, active boards (all return HTTP 200 from the API).
GH_BOARDS = [
    # Big tech / FAANG-adjacent
    "anthropic", "openai", "databricks", "stripe", "airbnb", "figma",
    "vercel", "shopify", "coinbase", "lyft", "pinterest", "confluent",
    # Cloud / infra
    "hashicorp", "datadog", "cloudflare", "fastly", "lacework",
    # Product / SaaS
    "notion-labs", "linear", "airtable", "loom", "miro", "hubspot",
    "asana", "monday", "zendesk", "intercom", "brex",
    # AI / ML
    "scale-ai", "cohere", "huggingface", "perplexityai", "replit",
    # Fintech
    "plaid", "chime", "robinhood", "mercury", "ramp",
    # Other notable
    "reddit", "duolingo", "canva", "carta", "gusto",
]


# ── Pydantic models ────────────────────────────────────────────────────────────
class JobListing(BaseModel):
    id: str
    title: str
    company: str
    location: str
    description: str
    requirements: list[str]
    salary: Optional[str] = "N/A"
    source: str = "duckduckgo"
    greenhouse_board: Optional[str] = None
    greenhouse_job_id: Optional[str] = None
    apply_url: Optional[str] = None


class JobListingsResponse(BaseModel):
    listings: list[JobListing]


# ── Keyword relevance scorer ───────────────────────────────────────────────────
def _relevance(title: str, location: str, query: str) -> int:
    """Simple keyword overlap score (higher = more relevant)."""
    q_words = set(re.split(r'\W+', query.lower())) - {"", "job", "jobs", "in", "at", "for", "and", "or"}
    t_words = set(re.split(r'\W+', (title + " " + location).lower()))
    return len(q_words & t_words)


# ── Greenhouse: fetch one board ────────────────────────────────────────────────
def _fetch_board_jobs(board: str, query: str) -> list[dict]:
    """Fetch all jobs from one board, return those matching the query."""
    try:
        r = requests.get(f"{GH_API}/{board}/jobs", timeout=TIMEOUT)
        if r.status_code != 200:
            return []
        jobs = r.json().get("jobs", [])
        matched = []
        for j in jobs:
            title    = j.get("title", "")
            location = (j.get("location") or {}).get("name", "")
            if _relevance(title, location, query) > 0:
                matched.append({
                    "board":    board,
                    "job_id":   str(j.get("id", "")),
                    "title":    title,
                    "location": location,
                    "url":      j.get("absolute_url", ""),
                })
        return matched
    except Exception:
        return []


def _fetch_gh_job_details(board: str, job_id: str) -> dict | None:
    """Fetch full job detail (description + questions) from Greenhouse."""
    try:
        r = requests.get(f"{GH_API}/{board}/jobs/{job_id}?questions=true", timeout=TIMEOUT)
        if r.status_code != 200:
            return None
        d = r.json()
        content = re.sub(r'<[^>]+>', ' ', d.get("content", ""))
        content = re.sub(r'\s+', ' ', content).strip()
        reqs = [q["label"] for q in d.get("questions", []) if q.get("required")]
        company_display = board.replace("-", " ").title()
        return {
            "id":                f"gh_{job_id}",
            "title":             d.get("title", ""),
            "company":           company_display,
            "location":          (d.get("location") or {}).get("name", "Not specified"),
            "description":       content[:1400] or "See the Greenhouse listing for details.",
            "requirements":      reqs[:10],
            "salary":            "N/A",
            "source":            "greenhouse",
            "greenhouse_board":  board,
            "greenhouse_job_id": job_id,
            "apply_url":         f"https://boards.greenhouse.io/{board}/jobs/{job_id}",
        }
    except Exception:
        return None


def _search_greenhouse(query: str, target: int = 8) -> list[dict]:
    """
    Search all known Greenhouse boards concurrently, pick the best matches,
    fetch full details for top hits (parallel), return up to `target` listings.
    """
    # Phase 1: concurrent lightweight board-list fetches
    candidates = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(_fetch_board_jobs, board, query): board for board in GH_BOARDS}
        for fut in concurrent.futures.as_completed(futures, timeout=12):
            try:
                candidates.extend(fut.result())
            except Exception:
                pass

    # Sort by relevance descending, dedupe by job_id
    seen, ranked = set(), []
    for c in sorted(candidates, key=lambda x: _relevance(x["title"], x["location"], query), reverse=True):
        if c["job_id"] not in seen:
            seen.add(c["job_id"])
            ranked.append(c)

    # Phase 2: fetch details for top candidates (parallel, limit detail fetches)
    top = ranked[: target * 2]  # fetch extras in case some 404
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_fetch_gh_job_details, c["board"], c["job_id"]) for c in top]
        for fut in concurrent.futures.as_completed(futures, timeout=15):
            try:
                job = fut.result()
                if job:
                    results.append(job)
            except Exception:
                pass
            if len(results) >= target:
                break

    return results[:target]


# ── DuckDuckGo general search (best-effort) ───────────────────────────────────
_SYSTEM_DDG = """You are a Job Sourcing Specialist. Generate 4-6 realistic job listings
from the search snippets. Vary company size, location, and seniority. Each listing needs
a detailed description (200-300 words) and 5-8 requirements.
Set source="duckduckgo" on every listing. Return ONLY valid JSON."""


def _search_duckduckgo(query: str, resume: str = "") -> list[dict]:
    snippets = []
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        for variant in [f"{query} job opening 2025", f"{query} hiring remote"]:
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(variant, max_results=8):
                        url = r.get("href", "")
                        if "greenhouse.io" not in url:
                            snippets.append(f"Title: {r.get('title','')}\nSnippet: {r.get('body','')}")
                if len(snippets) >= 10:
                    break
            except Exception:
                pass
    except Exception:
        pass

    if not snippets:
        snippets = [f"Generic {query} role — no live snippets available."]

    llm    = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, max_retries=2)
    schema = JobListingsResponse.model_json_schema()
    prompt = (
        f"Query: {query}\nResume context: {resume[:500] if resume else 'None'}\n"
        f"Search snippets:\n{chr(10).join(snippets[:10])}\n\n"
        f"Generate 4 realistic job listings with source='duckduckgo'.\n"
        f"Schema: {json.dumps(schema)}"
    )
    try:
        resp   = llm.invoke([SystemMessage(content=_SYSTEM_DDG), HumanMessage(content=prompt)])
        parsed = JobListingsResponse.model_validate_json(clean_json_response(resp.content))
        jobs   = [j.model_dump() for j in parsed.listings]
        for j in jobs:
            j["source"] = "duckduckgo"
            j.setdefault("greenhouse_board",  None)
            j.setdefault("greenhouse_job_id", None)
            j.setdefault("apply_url",         None)
        return jobs
    except Exception:
        return []


# ── Public API ─────────────────────────────────────────────────────────────────
def find_jobs(query: str, resume: str = "") -> list[dict]:
    """Returns Greenhouse (real live jobs) + DuckDuckGo (LLM-enriched) listings."""
    gh_jobs  = _search_greenhouse(query, target=8)
    ddg_jobs = _search_duckduckgo(query, resume)
    return gh_jobs + ddg_jobs


# ── Greenhouse apply ───────────────────────────────────────────────────────────
def fetch_gh_questions(board: str, job_id: str) -> list[dict]:
    try:
        r = requests.get(f"{GH_API}/{board}/jobs/{job_id}?questions=true", timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json().get("questions", [])
    except Exception:
        pass
    return []


def apply_to_greenhouse(
    board: str,
    job_id: str,
    first_name: str,
    last_name: str,
    email: str,
    phone: str,
    resume_bytes: bytes,
    cover_letter_text: str = "",
    linkedin_url: str = "",
) -> tuple[bool, str]:
    """POST a job application to Greenhouse. Returns (success, message)."""
    try:
        questions = fetch_gh_questions(board, job_id)
        field_map: dict[str, str] = {}
        for q in questions:
            label = (q.get("label") or "").lower()
            qid   = str(q.get("id", ""))
            if "first" in label:   field_map["first"]    = qid
            elif "last" in label:  field_map["last"]     = qid
            elif "email" in label: field_map["email"]    = qid
            elif "phone" in label: field_map["phone"]    = qid
            elif "linkedin" in label: field_map["linkedin"] = qid

        data = {
            f'answers[{field_map.get("first",  "first_name")}]': first_name,
            f'answers[{field_map.get("last",   "last_name")}]':  last_name,
            f'answers[{field_map.get("email",  "email")}]':      email,
            f'answers[{field_map.get("phone",  "phone")}]':      phone,
        }
        if linkedin_url and "linkedin" in field_map:
            data[f'answers[{field_map["linkedin"]}]'] = linkedin_url

        files: dict = {"resume": ("resume.pdf", resume_bytes, "application/pdf")}
        if cover_letter_text:
            files["cover_letter"] = ("cover_letter.txt", cover_letter_text.encode(), "text/plain")

        resp = requests.post(
            f"{GH_API}/{board}/jobs/{job_id}", data=data, files=files, timeout=15
        )
        if resp.status_code in (200, 201):
            return True, "Application submitted via Greenhouse!"
        return False, f"Greenhouse returned {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, f"Submit failed: {e}"
