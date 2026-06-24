# Individual Contributions Documentation
## Job Application Assistant — Multi-Agent Orchestration Capstone

**Repository:** `https://github.com/Sam-wiz/ML-capstone-project` \
**Team size:** 5 members

---

## Samrudh — State Schema, Graph Architecture & Persistence

**Files owned:** `state.py`, `graph.py`

### What I built

I designed the LangGraph graph that every other agent plugs into, the shared state object that flows through it, and the persistent checkpoint system.

**`state.py`**

I defined `GraphState`, the single TypedDict that acts as the shared memory of the entire pipeline. Every node reads from it and writes back to it. I also designed all four Pydantic v2 output models that enforce strict structured handoffs between agents:

- `KeywordAnalysis` — matched, missing, and bonus keywords
- `FitScoreOutput` — score (int, 0–100 enforced), keyword analysis, strengths, gaps, recommendation enum (`strong_fit` / `moderate_fit` / `low_fit`), reasoning string
- `TailoredCVOutput` — rewritten summary, tailored bullets (max 6), skills section, change log
- `CoverLetterOutput` — paragraph-by-paragraph structure plus full assembled text
- `HumanFeedback` — decision enum (`approve` / `edit` / `reject`), optional edited CV and cover letter, notes

Using Pydantic v2 means agents cannot pass malformed data downstream. If an agent's LLM response doesn't match the schema, `model_validate_json()` raises `ValidationError` immediately and the fallback path in that agent activates — rather than corrupt state silently propagating through the graph.

**`graph.py`**

I implemented the full `StateGraph`:

- Registered all 7 nodes: `guardrail`, `fit_scorer`, `low_fit_warning`, `cv_tailor`, `cover_letter_writer`, `hitl_review`, `assembler`
- Wrote 4 routing functions with explicit conditional logic:
  - `route_after_guardrail` — abort if `guardrail_passed = False`, else proceed to fit scorer
  - `route_after_fit_score` — `low_fit_warning` if score < 40, else `cv_tailor`
  - `route_after_low_fit_warning` — `cv_tailor` if human approves, `END` if rejects
  - `route_after_hitl_review` — `assembler` if approve/edit, `END` if reject
- Migrated from `MemorySaver` to `SqliteSaver` (`langgraph-checkpoint-sqlite`) for durable persistence — checkpoints now survive server restarts, so in-progress HITL reviews are never lost
- Set `interrupt_before=["hitl_review", "low_fit_warning"]` — the graph persists state to the checkpointer before pausing at these nodes

**Design decisions I own:**
- I interrupt *before* the HITL nodes rather than inside them. State is checkpointed before any human-facing logic runs. Resuming with `graph.stream(None, config)` after `update_state()` re-enters cleanly from the saved checkpoint.
- SqliteSaver is opened with a persistent `sqlite3.connect(..., check_same_thread=False)` connection at module load time, so all Streamlit reruns share the same connection without re-opening the database.


---

## Pranav — Guardrail Node, Fit Scorer & Interview Prep Agent

**Files owned:** `agents/guardrail.py`, `agents/fit_scorer.py`, `agents/interview_prep.py`

### What I built

I built the first two nodes in the pipeline and the post-pipeline Interview Prep agent.

**`agents/guardrail.py`**

The guardrail runs synchronously and makes zero LLM calls for its core checks:

1. **Empty check** — blocks if JD or resume is empty after stripping whitespace
2. **Minimum length** — JD must be ≥ 30 words; resume must be ≥ 50 words. Catches copy-paste errors and test inputs.
3. **Prompt injection scan** — checks against 8 known injection patterns: `"ignore previous instructions"`, `"ignore all instructions"`, `"you are now"`, `"disregard your"`, `"forget everything"`, `"jailbreak"`, `"act as"`, `"pretend you are"`. String-match is fast and catches the most common patterns before any token is spent.
4. **Metadata extraction** — makes one cheap `gpt-4o-mini` call to extract `company_name` and `role_title` from the JD. These are used by all three agents for personalisation. Falls back gracefully if the extraction fails.

If any check fails: `guardrail_passed = False`, `routing_decision = "abort"`, `route_after_guardrail` sends the graph to `END`. No agents run, no API costs incurred.

**`agents/fit_scorer.py`**

The most analytically complex agent:

1. Calls `extract_keywords` tool on the JD → gets structured keyword lists
2. Calls `extract_keywords` tool on the resume → gets structured keyword lists
3. Calls `retrieve_context(f"{role_title} job requirements")` for RAG grounding
4. Constructs a detailed scoring prompt with the keyword diff and industry context
5. Calls `gpt-4o-mini` at temperature 0.1 (low temperature → consistent scoring)
6. Parses response into `FitScoreOutput` via `model_validate_json()`
7. Sets `routing_decision`: `"low_fit_warning"` if score < 40, `"proceed"` otherwise

The fallback I wrote: if the LLM call or JSON parse fails, the agent creates a default `FitScoreOutput` with score=50, `"moderate_fit"`, and logs the error to `agent_log`. The graph continues — it never crashes silently or hangs.

**`agents/interview_prep.py`** *(new)*

A post-pipeline agent that generates 12 tailored interview questions from the JD and resume after the user approves their application package. Question breakdown:

- 3 behavioural (STAR format, tied to specific JD requirements)
- 3 technical (exact tech stack from the JD — not generic questions)
- 2 situational (hypothetical role-specific scenarios)
- 2 role-specific (culture fit, ownership mindset)
- 1 "why this company / role"
- 1 opener ("tell me about yourself")

Each question carries: `difficulty` (easy/medium/hard), `talking_points` naming specific projects and metrics from the candidate's own resume, `sample_answer_structure`, and a concise coaching `tip`. The output also includes `red_flags_to_avoid` — 3 common mistakes specific to this role type.

**Design decisions I own:**
- Two separate keyword tool calls (JD and resume) give the scoring prompt a clean structured diff rather than asking the LLM to compare in natural language.
- Interview questions reference actual resume artifacts (project names, companies, tools used) rather than being generic — this is enforced in the system prompt and validated at runtime.


---

## Rushabh — CV Tailor Agent, Tools Module & Resume PDF Generation

**Files owned:** `agents/cv_tailor.py`, `tools/tools.py`, `tools/pdf_parser.py`, `tools/resume_pdf_writer.py`

### What I built

I built Agent 2, the entire tools module that all agents depend on, and the two-step resume PDF pipeline.

**`tools/tools.py`**

Three `@tool`-decorated functions:

`extract_keywords(text: str) → dict` — Rule-based extractor using pattern matching against lists of known technical skills, soft skills, qualification keywords, and regex for uppercase acronyms. Returns four categorised lists. Entirely deterministic — no LLM call, no variability, no cost. Keeping this rule-based prevents hallucination on keyword extraction from contaminating the scoring prompt.

`rewrite_bullets(original_bullets, target_keywords, role_title) → list` — Takes resume bullets and the missing keywords from `FitScoreOutput`. Returns per-bullet dicts with the original bullet, keywords to weave in, and a hint note. These are *suggestions* to the LLM, not finished rewrites — the LLM synthesises from them, which is more reliable than freeform generation.

`validate_cover_letter(cover_letter_text: str) → dict` — Quality gate used by Agent 3. Checks: word count (150–500 range), salutation, closing. Returns `passed: bool` and a list of issues.

**`agents/cv_tailor.py`**

Agent 2 workflow: extracts bullet candidates from the resume using line heuristics, calls `rewrite_bullets` with the top 6 bullets and missing keywords, constructs a hint-guided LLM prompt, parses into `TailoredCVOutput`. The system prompt enforces that the agent never fabricates experience — only surfaces existing content using JD keywords.

**`tools/pdf_parser.py`** *(new)*

PDF resume parser using `pdfplumber`. Handles multi-column layouts (which PyPDF2 fails on) using `extract_text(x_tolerance=3, y_tolerance=3)`. Accepts raw file bytes (from `st.file_uploader`) and returns clean plain text.

**`tools/resume_pdf_writer.py`** *(new)*

A two-step resume update pipeline:

1. `generate_resume_structured()` — Single `gpt-4o-mini` call: takes the original resume text and the tailored CV output, applies minimal targeted edits (keyword-adding, summary rewrite), and returns a complete structured JSON representation of the updated resume. The LLM prompt uses `<placeholder>` format (not `—` separators) to prevent the model from copying schema markup into values.

2. `render_resume_html()` / `render_resume_pdf_from_data()` — Deterministic renderers that take the structured JSON and produce either HTML (for in-app preview with a "Generated by Job Application Assistant" watermark footer) or a clean PDF via `fpdf2` (watermark omitted for download). Layout constants are tuned to a single page: tight margins (15mm L/R, 12mm top, 10mm bottom), Times Bold for name/headings, line height 4.5mm.

A `_s()` helper handles `fpdf2`'s latin-1 encoding constraint by substituting `•→-`, `–→-`, `—→-`, `→→->`, and smart quotes before encoding.

**Design decisions I own:**
- LLM edits the whole resume as a unit rather than patching individual sections — this preserves formatting continuity and avoids awkward transitions.
- `resume_structured_data` is cached in `st.session_state` so the LLM is called exactly once when entering the Done stage, not on every Streamlit rerun.


---

## Navneet — Cover Letter Writer, RAG Pipeline & Job Finder

**Files owned:** `agents/cover_letter_writer.py`, `rag/rag_setup.py`, `agents/job_finder.py`

### What I built

I built Agent 3, the ChromaDB RAG pipeline that grounds both the Fit Scorer and Cover Letter Writer in factual domain knowledge, and the multi-source job search system.

**`rag/rag_setup.py`**

I set up a local ChromaDB vector store with 7 embedded knowledge documents:

| Document | Domain |
|----------|--------|
| Software engineering job market trends | `software_engineering` |
| Data science role requirements | `data_science` |
| Product management competencies | `product_management` |
| Cover letter best practices and structure | `general` |
| Resume tailoring + ATS optimisation | `general` |
| Startup vs enterprise application strategy | `general` |
| Marketing and growth role requirements | `marketing` |

Technical setup: `text-embedding-3-small` embeddings, `RecursiveCharacterTextSplitter` (400 token chunks, 50 token overlap), `persist_directory="./chroma_db"` for disk persistence (first run embeds and saves; subsequent runs load from disk, `os.path.exists()` check prevents re-embedding).

**`agents/cover_letter_writer.py`**

Agent 3 makes two RAG calls before writing — best practices context and industry-specific content separately, rather than one combined query that mixes relevance. Uses `FitScoreOutput.strengths`, `TailoredCVOutput.summary`, and the top 3 tailored bullets as evidence. `temperature=0.5` (higher than other agents — cover letters need natural prose). Self-correction loop: calls `validate_cover_letter` after generation; if it fails, retries with a correction prompt appending the specific issues.

**`agents/job_finder.py`** *(new)*

A dual-source job search system:

*Greenhouse (primary source):* Rather than relying on search-engine discovery (which is fragile and rate-limited), I maintain a curated list of 40+ verified Greenhouse board slugs (Stripe, Anthropic, Figma, Vercel, Pinterest, Lyft, Coinbase, Databricks, etc.). Search runs in two parallel phases:

1. **Board-list phase** — concurrent `ThreadPoolExecutor(max_workers=12)` fetches all boards' job lists, filters by keyword relevance (token-overlap scoring against title + location), deduplicates by job ID
2. **Detail phase** — concurrent `ThreadPoolExecutor(max_workers=8)` fetches full descriptions + question lists for the top candidates

This returns 8+ real, verified job listings with live descriptions in ~3–5 seconds. Every Greenhouse listing carries `greenhouse_board` and `greenhouse_job_id` for downstream use.

*DuckDuckGo (secondary source):* Falls back to LLM-enriched listings when DDG responds. Uses `gpt-4o-mini` to synthesise 4 realistic listings from search snippets. Graceful: if DDG returns nothing (rate-limited, renamed package), the LLM generates plausible listings anyway.

Every listing has a `source` field (`"greenhouse"` or `"duckduckgo"`) shown as a badge in the UI.

**Design decisions I own:**
- Curated board list vs. dynamic discovery: DDG's `site:boards.greenhouse.io` search returns 0 results consistently (rate-limited + package deprecated). A maintained list of 40+ top tech companies is more reliable and doesn't depend on a third-party search index.
- Two-phase parallel fetch: board-list fetches are cheap (small JSON); detail fetches are expensive (full HTML description). Running them in separate pools avoids wasting detail-fetch capacity on jobs that won't make the relevance cut.


---

## Ayush — HITL Nodes, Assembler, Streamlit UI, Auth & Application Tracker

**Files owned:** `agents/hitl_and_assembler.py`, `app.py`, `main.py`, `evals/run_evals.py`, `auth/auth.py`, `db/database.py`

### What I built

I built the human-in-the-loop layer, the final output assembler, both interfaces, the evaluation suite, the JWT authentication system, and the application tracker database.

**`agents/hitl_and_assembler.py`**

`low_fit_warning_node` — Fires when `fit_score < 40`. Prepares display state for the UI. The graph is already interrupted by `interrupt_before` — this node just writes the warning payload to state.

`hitl_review_node` — The main review gate. Intentionally thin: sets `awaiting_human = True` and logs the pause. All review logic lives in the UI — separation of graph state from UI state is deliberate.

`assembler_node` — Reads `human_feedback`: if the human provided edited CV or cover letter text, those replace the agent-generated versions. Assembles everything into a single markdown document with sections for fit analysis, tailored CV, and cover letter. Resets `awaiting_human = False`.

**`app.py` — Streamlit UI** *(significantly expanded)*

A 5-stage state machine (`input → running → low_fit → review → done → aborted`) across three pages:

- **Job Finder** — Search results from both sources with source badges (Greenhouse vs DuckDuckGo), expand-to-details, one-click "Tailor for this role" that pre-fills the Tailor page
- **Tailor Application** — Step tracker, resume PDF upload (bytes cached in session), pipeline runner, HITL review with editable fields, done stage with three tabs (Application Package, Updated Resume, Interview Prep)
- **My Applications** — Application tracker (see below)

The Done stage runs two LLM calls (resume structuring and interview prep) cached in session state so Streamlit reruns don't re-call the LLM.

**`auth/auth.py`** *(new)*

JWT authentication:
- `bcrypt` for password hashing (salt rounds default, `checkpw` for constant-time comparison)
- `PyJWT` for token encoding/decoding — `sub` stored as string (JWT spec) and cast back to `int` on decode
- 7-day token expiry; `JWT_SECRET` read from env with a hardcoded dev fallback
- `register()` / `login()` return `(bool, message)` / `(token | None, message)` tuples — no exceptions bubble to the UI

**`db/database.py`** *(new)*

SQLite schema with two tables:

`users` — `id`, `email` (unique), `password_hash`, `name`, `created_at`

`applications` — `id`, `user_id` (FK → users, cascade delete), `job_title`, `company`, `location`, `status` (CHECK constraint: `Applied / Screening / Interview / Offer / Rejected`), `source`, `greenhouse_board`, `greenhouse_job_id`, `fit_score`, `job_description`, `notes`, `applied_at`, `updated_at`. An `AFTER UPDATE` trigger keeps `updated_at` in sync automatically.

Helper functions: `init_db()`, `create_user()`, `get_user_by_email()`, `get_user_by_id()`, `add_application()`, `get_applications()`, `update_application_status()`, `delete_application()`.

The tracker page shows per-status counts (Applied / Screening / Interview / Offer / Rejected), inline status updates with notes, delete, and a manual entry form for roles applied to outside the app.

Applications are auto-added to the tracker when the user approves a tailoring pipeline (with fit score) or applies via an external link.

**`main.py` — CLI runner**

Terminal-based HITL loop using `input()`. Reads graph state after each interrupt, prints agent outputs, prompts for a decision, calls `graph.update_state()` and `graph.stream(None, config)` to resume. Saves the final package to `output_package.md`.

**`evals/run_evals.py` — 5 evaluation cases**

| # | Case | Key assertions |
|---|------|---------------|
| 1 | Happy path | `guardrail_passed=True`, `score ≥ 60`, `routing="proceed"`, all outputs populated, final package assembled |
| 2 | Empty resume | `guardrail_passed=False`, `fit_score=None` (no agents ran), message mentions resume |
| 3 | Prompt injection | `guardrail_passed=False`, `routing="abort"`, no LLM calls made |
| 4 | Low fit routing | `score < 50`, `routing_decision="low_fit_warning"`, graph paused at interrupt |
| 5 | Structured output validation | `FitScoreOutput` validates with Pydantic, score in 0–100, `TailoredCVOutput` has ≥ 3 bullets, cover letter word count 150–500 |

Each eval that reaches HITL auto-approves using `HumanFeedback(decision="approve")` injected via `graph.update_state()` — so the full pipeline can run unattended in CI.

**Design decisions I own:**
- Streamlit's `stage` variable is separate from LangGraph's `awaiting_human` field. The LangGraph field records *where the graph paused*. The Streamlit field controls *which screen to render*. A Streamlit rerun doesn't accidentally trigger a graph resume.
- Auth uses a button-toggle pattern (`st.session_state.auth_mode`) instead of `st.tabs()` inside columns — Streamlit's widget state machine behaves unreliably with tabs nested inside column layouts, causing blank-page rendering bugs.


---

## Contribution Summary

| Member | Files | Rubric areas |
|--------|-------|-------------|
| Samrudh | `state.py`, `graph.py` | LangGraph graph · Pydantic structured outputs · routing/branching · SqliteSaver persistence |
| Pranav | `agents/guardrail.py`, `agents/fit_scorer.py`, `agents/interview_prep.py` | Guardrails · Fit Scorer · Interview Prep agent · tool use |
| Rushabh | `agents/cv_tailor.py`, `tools/tools.py`, `tools/pdf_parser.py`, `tools/resume_pdf_writer.py` | CV Tailor · tools module · PDF parsing · LLM-driven resume PDF generation |
| Navneet | `agents/cover_letter_writer.py`, `rag/rag_setup.py`, `agents/job_finder.py` | Cover Letter writer · RAG/ChromaDB · Greenhouse + DuckDuckGo job search |
| Ayush | `agents/hitl_and_assembler.py`, `app.py`, `main.py`, `evals/run_evals.py`, `auth/auth.py`, `db/database.py` | HITL · assembler · Streamlit UI · JWT auth · application tracker · evaluations |
