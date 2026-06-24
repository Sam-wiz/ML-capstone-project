# Individual Contributions Documentation
## Job Application Assistant — Multi-Agent Orchestration Capstone

**Repository:** `https://github.com/Sam-wiz/ML-capstone-project` \
**Team size:** 5 members

## Samrudh — State Schema & Graph Architecture

**Files owned:** `state.py`, `graph.py`

### What I built

I designed the LangGraph graph that every other agent plugs into, and the shared state object that flows through it.

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
- Configured `MemorySaver` as the checkpointer — this is what enables the interrupt/resume HITL pattern
- Set `interrupt_before=["hitl_review", "low_fit_warning"]` — the graph persists state to the checkpointer before pausing at these nodes, so if the UI crashes during review, state is not lost

**Design decision I own:** I interrupt *before* the HITL nodes rather than inside them. This means state is checkpointed before any human-facing logic runs. Resuming with `graph.stream(None, config)` after `update_state()` re-enters cleanly from the saved checkpoint.


## Pranav — Guardrail Node & Fit Scorer Agent

**Files owned:** `agents/guardrail.py`, `agents/fit_scorer.py`

### What I built

I built the first two nodes in the pipeline: the guardrail that protects the system before any LLM call, and the Fit Scorer that produces the numerical assessment everything downstream is built around.

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

**Design decision I own:** I use two separate `extract_keywords` tool calls (one on JD, one on resume) rather than one combined call. This gives the scoring prompt a clean, structured diff — matched keywords and missing keywords as separate lists — rather than asking the LLM to do the comparison itself in natural language. Structured input → more reliable scoring output.


## Rushabh — CV Tailor Agent & Tools Module

**Files owned:** `agents/cv_tailor.py`, `tools/tools.py`

### What I built

I built Agent 2 and the entire tools module that all three agents depend on.

**`tools/tools.py`**

Three `@tool`-decorated functions:

`extract_keywords(text: str) → dict`
Rule-based extractor using pattern matching against lists of known technical skills, soft skills, qualification keywords, and regex for uppercase acronyms. Returns four categorised lists. Entirely deterministic — no LLM call, no variability, no cost. I chose rule-based deliberately: asking the LLM to extract keywords inside the scoring prompt increases hallucination on that sub-task. Separating it into a deterministic tool keeps the main prompt focused on reasoning.

`rewrite_bullets(original_bullets, target_keywords, role_title) → list`
Takes resume bullets and the missing keywords from `FitScoreOutput`. Returns per-bullet dicts with the original bullet, a list of keywords to weave in, and a hint note. This is used by Agent 2 as structured hints to the LLM — not as finished rewrites. The LLM synthesises from these hints, which is more reliable than asking it to freestyle. The tool itself never changes the bullet text.

`validate_cover_letter(cover_letter_text: str) → dict`
Quality gate used by Agent 3. Checks: word count (150–500 range), presence of a salutation (`"dear"`), presence of a closing (`"sincerely"`, `"regards"`, `"thank you"`). Returns `passed: bool` and a list of issues. If `passed = False`, Agent 3 appends a correction prompt and retries.

**`agents/cv_tailor.py`**

Agent 2 workflow:

1. Extracts raw bullet candidates from the resume text using line heuristics (lines with bullet chars or capitalized lines over 40 chars)
2. Calls `rewrite_bullets` with the top 6 bullets and the missing keywords from Agent 1's output
3. Constructs an LLM prompt with the tool's suggestions as structured hints
4. Parses into `TailoredCVOutput`

The most important constraint in the system prompt: **never fabricate experience**. The agent is instructed to rewrite existing content to surface relevance using JD keywords, not to invent skills or achievements. The tool's keyword hints reinforce this — they tell the LLM *which* keywords to weave in, not to make up new bullets.

**Design decision I own:** The `rewrite_bullets` tool returns *suggestions*, not finished bullets. The LLM agent decides which suggestions to apply and how. This keeps tool output deterministic and auditable, while leaving creative synthesis to the LLM — the right division of labour.


## Navneet — Cover Letter Writer Agent & RAG Pipeline

**Files owned:** `agents/cover_letter_writer.py`, `rag/rag_setup.py`

### What I built

I built Agent 3 and the ChromaDB RAG pipeline that grounds both the Fit Scorer and Cover Letter Writer in factual domain knowledge.

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

I embedded all documents inline as `Document` objects — the project runs immediately after `pip install` with no external files required.

Technical setup:
- **Embedding:** `text-embedding-3-small` — cheaper than `ada-002`, comparable quality for retrieval
- **Chunking:** `RecursiveCharacterTextSplitter`, 400 token chunks, 50 token overlap
- **Persistence:** `persist_directory="./chroma_db"` — first run embeds and saves; subsequent runs load from disk. I implemented a `_retriever` module-level singleton with an `os.path.exists()` check to avoid re-embedding on every invocation.
- **Retrieval:** similarity search, k=3 chunks per query

`retrieve_context(query)` is the public function. It initialises the retriever on first call, runs the query, and returns the top-3 chunks joined as a single string.

**`agents/cover_letter_writer.py`**

Agent 3 makes two RAG calls before writing:
1. `retrieve_context("cover letter best practices writing tips")` — structural and stylistic guidance
2. `retrieve_context(f"{role_title} key skills requirements")` — industry-specific content

It then builds a prompt using `FitScoreOutput.strengths`, `TailoredCVOutput.summary`, and the top 3 tailored bullets as evidence — so the letter directly references the candidate's strongest points.

I set `temperature=0.5` — higher than the other agents because cover letters benefit from natural-sounding, varied prose rather than deterministic output.

**Self-correction loop I implemented:** After generation, the agent calls `validate_cover_letter`. If it fails, the agent appends a correction prompt identifying the specific issues and calls the LLM again. If the retry also fails, the original draft is kept — the graph never hangs.

**Design decision I own:** Two separate RAG queries (best practices + industry context) rather than one combined query. A combined query like "cover letter best practices software engineering" returns a mix that competes for relevance. Two queries give the model clean, focused context blocks for different purposes.


## Ayush — HITL Nodes, Assembler, Streamlit UI & Evaluations

**Files owned:** `agents/hitl_and_assembler.py`, `app.py`, `main.py`, `evals/run_evals.py`

### What I built

I built the human-in-the-loop layer, the final output assembler, both interfaces (Streamlit and CLI), and the full evaluation suite.

**`agents/hitl_and_assembler.py`**

Three nodes:

`low_fit_warning_node` — Fires when `fit_score < 40`. Sets `awaiting_human = True` and writes a warning message (score, recommendation, gap list) to `guardrail_message` for the UI to surface. The graph is already interrupted at this point by `interrupt_before` — this node just prepares the display state.

`hitl_review_node` — The main review gate. Sets `awaiting_human = True` and logs the pause. All actual review logic lives in the UI — this node is intentionally thin. The separation between graph state and UI state is deliberate: the graph doesn't care *how* the human reviews, only *what decision* they return.

`assembler_node` — Runs after HITL approval. Reads `human_feedback` from state: if the human provided edited CV or cover letter text, those replace the agent-generated versions. Assembles everything into a single markdown document with sections for fit analysis (score, strengths, gaps, keyword match), tailored CV, and cover letter. Resets `awaiting_human = False`.

**`app.py` — Streamlit UI**

A 5-stage state machine using `st.session_state.stage`:

| Stage | What the user sees | Transition |
|-------|-------------------|-----------|
| `input` | JD + resume text areas | Click "Generate" |
| `low_fit` | Score, gaps, proceed/abort buttons | Button click |
| `review` | Full agent outputs, editable fields, 3 buttons | Button click |
| `done` | Final package + download button | — |
| `aborted` | Abort message + reset button | — |

The HITL resume pattern: after the user clicks a decision button, I call `graph.update_state(config, {"human_feedback": feedback.model_dump(), "awaiting_human": False})`, then `graph.stream(None, config, stream_mode="values")`. The `None` input tells LangGraph to resume from the checkpoint rather than start fresh.

**`main.py` — CLI runner**

Terminal-based HITL loop using `input()`. Reads graph state after each interrupt, prints the agent outputs, prompts for a decision, calls `graph.update_state()` and `graph.stream(None, config)` to resume. Saves the final package to `output_package.md`.

**`evals/run_evals.py` — 5 evaluation cases**

| # | Case | Key assertions |
|---|------|---------------|
| 1 | Happy path | `guardrail_passed=True`, `score ≥ 60`, `routing="proceed"`, all outputs populated, final package assembled |
| 2 | Empty resume | `guardrail_passed=False`, `fit_score=None` (no agents ran), message mentions resume |
| 3 | Prompt injection | `guardrail_passed=False`, `routing="abort"`, no LLM calls made |
| 4 | Low fit routing | `score < 50`, `routing_decision="low_fit_warning"`, graph paused at interrupt |
| 5 | Structured output validation | `FitScoreOutput` validates with Pydantic, score in 0–100, `TailoredCVOutput` has ≥ 3 bullets, cover letter word count 150–500 |

Each eval that reaches HITL auto-approves using `HumanFeedback(decision="approve")` injected via `graph.update_state()` — so the full pipeline can run unattended in CI.

**Design decision I own:** I keep Streamlit's `stage` variable separate from LangGraph's `awaiting_human` field. The LangGraph field records *where the graph paused*. The Streamlit field controls *which screen to render*. They sometimes mirror each other but are independently managed — a Streamlit rerun doesn't accidentally trigger a graph resume.

## Contribution Summary

| Member | Files | Rubric areas |
|--------|-------|-------------|
| Samrudh | `state.py`, `graph.py` | LangGraph implementation · structured outputs · routing/branching · state management |
| Pranav | `agents/guardrail.py`, `agents/fit_scorer.py` | Guardrails · Agent 1 · conditional routing · tool use |
| Rushabh | `agents/cv_tailor.py`, `tools/tools.py` | Agent 2 · tool use · structured outputs |
| Navneet | `agents/cover_letter_writer.py`, `rag/rag_setup.py` | Agent 3 · RAG/knowledge grounding |
| Ayush | `agents/hitl_and_assembler.py` · `app.py` · `main.py` · `evals/run_evals.py` | HITL · demo quality · evaluation · debugging/observability |
