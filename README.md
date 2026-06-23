# Job Application Assistant
### Multi-Agent AI System | LangGraph Capstone

A LangGraph-based multi-agent pipeline that takes a raw job description and resume, and produces a tailored CV, targeted cover letter, and fit score — with human review and approval before any output is finalised.

> Built for the **Multi Agent Orchestration [AI/ML]** course capstone.

---

## What it does

You paste a job description and your resume. Three specialised AI agents work in sequence:

1. **Fit Scorer** — compares your resume against the JD, scores the match 0–100, and identifies keyword gaps
2. **CV Tailor** — rewrites your resume bullets and summary to surface hidden relevance using the gap analysis
3. **Cover Letter Writer** — generates a personalised, RAG-grounded cover letter

Before anything is finalised, **you review everything** and choose to approve, edit, or reject. Only after your approval does the system produce the final output package.

---

## Architecture

```
JD + Resume
     │
[Guardrail]  — validates input, blocks injections
     │
[Fit Scorer Agent]  ← extract_keywords tool + RAG
     │
  score < 40? ──► [Low Fit Warning] ──► HITL: proceed or abort
     │
[CV Tailor Agent]  ← rewrite_bullets tool
     │
[Cover Letter Writer Agent]  ← RAG (2 queries) + self-correction
     │
[HITL Review]  ◄── INTERRUPT: approve / edit / reject
     │
[Assembler]  →  Final markdown package
```

**Conditional routing:** score < 40 triggers a low-fit warning gate before tailoring begins.
**Two HITL interrupts:** one on low fit, one always before final output.

### Agents

| Agent | File | Tools | Output model |
|-------|------|-------|-------------|
| Fit Scorer | `agents/fit_scorer.py` | `extract_keywords`, RAG retriever | `FitScoreOutput` |
| CV Tailor | `agents/cv_tailor.py` | `rewrite_bullets` | `TailoredCVOutput` |
| Cover Letter Writer | `agents/cover_letter_writer.py` | `validate_cover_letter`, RAG retriever | `CoverLetterOutput` |

### Tools

| Tool | Description |
|------|-------------|
| `extract_keywords(text)` | Rule-based keyword extractor — deterministic, no LLM cost |
| `rewrite_bullets(bullets, keywords, role)` | Returns keyword-targeted rewrite suggestions per bullet |
| `validate_cover_letter(text)` | Quality gate: word count, salutation, closing checks |

### RAG Knowledge Base

Seven embedded documents covering software engineering, data science, product management, marketing, cover letter best practices, resume/ATS optimisation, and startup vs enterprise strategy. The ChromaDB vector store is persisted locally; OpenAI embeddings are used when `OPENAI_API_KEY` is configured, and the retriever falls back to inline keyword-ranked context if embedding retrieval is unavailable.

---

## Project Structure

```
job_assistant/
├── PRD.md                     ← full product requirements document
├── README.md                  ← this file
├── CONTRIBUTIONS.md           ← individual member contributions
├── requirements.txt
├── .env.example
├── config.py                  ← runtime config helpers, including API-key preflight
├── state.py                   ← GraphState TypedDict + 4 Pydantic models
├── graph.py                   ← LangGraph StateGraph (7 nodes, 4 routing functions)
├── main.py                    ← CLI runner with HITL loop
├── app.py                     ← Streamlit UI (5-stage state machine)
├── agents/
│   ├── guardrail.py           ← input validation + injection detection
│   ├── json_utils.py          ← resilient JSON cleanup for structured LLM responses
│   ├── fit_scorer.py          ← Agent 1: scores 0–100, keyword analysis
│   ├── cv_tailor.py           ← Agent 2: rewrites CV to match JD
│   ├── cover_letter_writer.py ← Agent 3: RAG-grounded cover letter
│   └── hitl_and_assembler.py  ← HITL nodes + final assembler
├── tools/
│   └── tools.py               ← 3 @tool-decorated helper functions
├── rag/
│   └── rag_setup.py           ← ChromaDB setup + retrieve_context()
└── evals/
    └── run_evals.py           ← 5 evaluation scenarios with assertions
```

---

## Setup

**Requirements:** Python 3.10+, an OpenAI API key.

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd job_assistant

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

**.env file:**
```
OPENAI_API_KEY=sk-...

# Optional — enables LangSmith tracing (free tier)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=job-assistant-capstone
```

---

## Running the project

### Streamlit UI (recommended for demo)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Paste JD and resume, click Generate, review outputs, approve.

### CLI mode (good for testing)

```bash
python main.py
```

Uses the hardcoded sample inputs in `main.py`. HITL prompts appear in the terminal.

### Evaluation suite

```bash
python evals/run_evals.py
```

Runs all 5 eval cases and prints a pass/fail report. Evals 2 and 3 (guardrail cases) run fast with no LLM cost. Evals 1, 4, 5 make LLM calls and are skipped automatically if `OPENAI_API_KEY` is not configured.

---

## Evaluation cases

| # | Case | Tests |
|---|------|-------|
| 1 | Happy path — strong match | Full pipeline, score ≥ 60, all outputs populated |
| 2 | Empty resume | Guardrail blocks before any agent runs |
| 3 | Prompt injection | Injection detected, graph aborts |
| 4 | Low fit routing | Score < 40, `routing_decision = low_fit_warning` |
| 5 | Structured output validation | All Pydantic models validate, ranges correct |

---

## Rubric coverage

| Requirement | How it's met |
|-------------|-------------|
| 3+ agents with distinct roles | Fit Scorer, CV Tailor, Cover Letter Writer |
| LangGraph orchestration | `StateGraph` with nodes, edges, conditional routing, `MemorySaver` |
| State management | `GraphState` TypedDict passed through every node |
| Tool use (2+) | `extract_keywords`, `rewrite_bullets`, `validate_cover_letter` |
| Structured outputs | 4 Pydantic v2 models for all agent handoffs |
| Conditional routing | Score-based routing to `low_fit_warning` or `cv_tailor` |
| RAG | ChromaDB with 7 embedded domain documents |
| Human-in-the-loop | Two interrupt points: low-fit gate + full review gate |
| Guardrails | Empty input, min length, prompt injection detection, no-fabrication constraint |
| Evaluation | 5 test cases with assertion-based pass/fail |
| Observability | LangSmith tracing via env vars + `agent_log` in every state |

---

## Key design decisions

**Why LangGraph over a LangChain chain:** Steps 2 and 3 depend on structured output from step 1. Conditional routing based on a score threshold determines which path the graph takes. Two separate HITL interrupt points need to pause and resume execution. These require a stateful graph, not a linear chain.

**Why Pydantic for agent outputs:** Agents cannot pass malformed data downstream. If an agent produces invalid JSON, Pydantic raises `ValidationError` immediately rather than silently corrupting the state.

**Why rule-based tools:** `extract_keywords` and `rewrite_bullets` are deterministic, fast, and cost zero tokens. Asking the LLM to do keyword extraction inside the main scoring prompt increases hallucination on that sub-task.

**Why two RAG queries in Agent 3:** A single combined query would return mixed results. Two targeted queries (best practices + industry context) give the cover letter writer cleaner, more focused context chunks.

---

## Team

| Member | Contribution |
|--------|-------------|
| Member 1 | State schema + LangGraph graph architecture |
| Member 2 | Guardrail node + Fit Scorer agent |
| Member 3 | CV Tailor agent + tools module |
| Member 4 | Cover Letter Writer agent + RAG pipeline |
| Member 5 | HITL/assembler nodes + Streamlit UI + evaluations |

See `CONTRIBUTIONS.md` for detailed individual contribution descriptions.
