# Product Requirements Document
## Job Application Assistant — Multi-Agent AI System

| Field | Value |
|-------|-------|
| Document version | 1.0 |
| Status | Final (capstone submission) |
| Course | Multi Agent Orchestration [AI/ML] |
| Team size | 5 members |
| Presentation | 25–30 June |
| Tech stack | LangGraph · LangChain · OpenAI GPT-4o-mini · ChromaDB · Pydantic v2 · Streamlit |

---

## 1. Problem Statement

Job seekers submit generic applications. Recruiters discard them within 6 seconds.

Tailoring a resume and cover letter to a specific job description is a well-understood best practice — but it requires reading the JD carefully, cross-referencing it against your own resume, identifying keyword gaps, rewriting bullet points, and writing a targeted cover letter. Done properly this takes 1–3 hours per application. Most candidates either skip it or do it inconsistently.

The deeper problem is that this is not one task — it is a pipeline of dependent tasks requiring different reasoning modes:

1. Analyse the JD and resume independently and compare them (analytical)
2. Rewrite CV content to surface hidden relevance (creative + constrained)
3. Write a cover letter grounded in both the JD and company context (generative + factual)
4. Let the human review everything before any output is used (quality gate)

No single LLM prompt can do all four well. Steps 2 and 3 depend on the output of step 1. Step 4 must block steps 2 and 3 from being applied until a human approves them. This is a coordination problem, not a generation problem — which is why it requires a multi-agent architecture.

---

## 2. Target Users

**Primary:** Job seekers applying to competitive roles who want tailored applications but lack the time to do it manually. Students, recent graduates, career changers.

**Secondary:** Career coaches who want to review and approve AI-generated drafts before sending to clients.

---

## 3. Goals

| Goal | Metric |
|------|--------|
| Reduce application tailoring time | From ~2 hrs manual → under 5 minutes with human review |
| Improve keyword match rate | Resume should match ≥70% of JD keywords on strong-fit roles |
| Maintain human control | Zero applications sent without explicit human approve/edit/reject decision |
| Prevent hallucination | Agent must never fabricate experience not present in original resume |
| Transparent pipeline | Every agent decision logged; LangSmith traces available |

---

## 4. Non-Goals

- This system does **not** submit applications automatically. Output is a markdown package the user downloads and uses themselves.
- This system does **not** scrape job boards. The user provides the JD by pasting it.
- This system does **not** store or persist user data between sessions. Each run is stateless beyond the current LangGraph checkpoint.
- This system is **not** a career advice chatbot. It performs a specific, bounded workflow.

---

## 5. System Architecture

### 5.1 Overview

```
User Input (JD + Resume text)
          │
    ┌─────▼──────────┐
    │  Guardrail Node │  Validates input. Blocks empty, too-short,
    │                 │  or injection-containing inputs. Extracts
    └─────┬──────────┘  company name + role title via LLM.
          │ passes / aborts
    ┌─────▼──────────┐
    │  Fit Scorer     │  Agent 1. Scores resume vs JD (0–100).
    │  Agent          │  Extracts matched/missing keywords.
    └─────┬──────────┘  Outputs FitScoreOutput (Pydantic).
          │
    score < 40?──────────────────────────────────────┐
          │ no                                        │ yes
    ┌─────▼──────────┐                    ┌──────────▼────────┐
    │  CV Tailor      │                    │ Low Fit Warning   │
    │  Agent          │◄── proceed ────────│ Node (HITL #1)    │
    └─────┬──────────┘    human approves   └──────────────────┘
          │                                        │ human rejects → END
    ┌─────▼──────────┐
    │  Cover Letter   │  Agent 3. RAG-grounded letter.
    │  Writer Agent   │  Self-corrects via validate_cover_letter tool.
    └─────┬──────────┘  Outputs CoverLetterOutput (Pydantic).
          │
    ┌─────▼──────────┐
    │  HITL Review    │  ◄── INTERRUPT. Human reads all three outputs.
    │  Node (HITL #2) │  Can approve, edit inline, or reject entirely.
    └─────┬──────────┘
          │ approve / edit
    ┌─────▼──────────┐
    │  Assembler Node │  Combines all outputs (+ any human edits)
    │                 │  into final markdown application package.
    └─────────────────┘
              │
         Final Package (markdown, downloadable)
```

### 5.2 Agent Specifications

#### Agent 1 — Fit Scorer (`agents/fit_scorer.py`)

| Property | Value |
|----------|-------|
| Model | `gpt-4o-mini`, temperature 0.1 |
| Tools | `extract_keywords` (on JD), `extract_keywords` (on resume) |
| RAG | `retrieve_context(f"{role_title} job requirements skills")` |
| Output model | `FitScoreOutput` |
| Output fields | `score` (int 0–100), `keyword_analysis` (matched/missing/bonus), `strengths` (list), `gaps` (list), `recommendation` (enum: strong_fit / moderate_fit / low_fit), `reasoning` (str) |
| Routing trigger | `score < 40` → `low_fit_warning`; `score ≥ 40` → `cv_tailor` |
| Failure mode | Falls back to score=50, moderate_fit — never crashes graph |

#### Agent 2 — CV Tailor (`agents/cv_tailor.py`)

| Property | Value |
|----------|-------|
| Model | `gpt-4o-mini`, temperature 0.3 |
| Tools | `rewrite_bullets(bullets, missing_keywords, role_title)` |
| Input | `FitScoreOutput.keyword_analysis.missing_keywords` from Agent 1 |
| Output model | `TailoredCVOutput` |
| Output fields | `summary` (str), `tailored_bullets` (list, max 6), `skills_section` (list), `changes_made` (list) |
| Hard constraint | System prompt explicitly prohibits fabricating experience |
| Failure mode | Falls back to original bullets — never invents content |

#### Agent 3 — Cover Letter Writer (`agents/cover_letter_writer.py`)

| Property | Value |
|----------|-------|
| Model | `gpt-4o-mini`, temperature 0.5 |
| Tools | `validate_cover_letter(text)` → triggers self-correction if fails |
| RAG | Two queries: cover letter best practices + industry/role context |
| Input | `FitScoreOutput.strengths`, `TailoredCVOutput.summary`, top 3 bullets |
| Output model | `CoverLetterOutput` |
| Output fields | `subject_line`, `opening_paragraph`, `body_paragraph_1`, `body_paragraph_2`, `closing_paragraph`, `full_text` |
| Self-correction | If `validate_cover_letter` fails, appends correction prompt and retries once |

### 5.3 Tools (`tools/tools.py`)

| Tool | Type | Description |
|------|------|-------------|
| `extract_keywords(text)` | Rule-based | Extracts technical skills, soft skills, qualifications, acronyms from text. Deterministic, zero LLM cost. |
| `rewrite_bullets(bullets, keywords, role)` | Rule-based | Returns per-bullet rewrite suggestions with keyword hints. Used by Agent 2 as structured input. |
| `validate_cover_letter(text)` | Rule-based | Checks word count (150–500), salutation, closing. Returns `passed` bool + issue list. |

All tools are `@tool`-decorated LangChain functions. No tool makes LLM calls — they are deterministic helpers.

### 5.4 RAG Knowledge Base (`rag/rag_setup.py`)

| Document | Domain | Used by |
|----------|--------|---------|
| Software engineering job market trends | `software_engineering` | Agent 1, Agent 3 |
| Data science role requirements | `data_science` | Agent 1, Agent 3 |
| Product management competencies | `product_management` | Agent 1, Agent 3 |
| Cover letter best practices | `general` | Agent 3 |
| Resume tailoring + ATS optimisation | `general` | Agent 1, Agent 2 |
| Startup vs enterprise application strategy | `general` | Agent 3 |
| Marketing and growth role requirements | `marketing` | Agent 1, Agent 3 |

- **Embedding model:** `text-embedding-3-small` (OpenAI)
- **Vector store:** ChromaDB (local, persisted to `./chroma_db`)
- **Chunk size:** 400 tokens, 50 token overlap
- **Retrieval:** similarity search, k=3

### 5.5 State Schema (`state.py`)

```python
class GraphState(TypedDict):
    # Inputs
    job_description: str
    raw_resume: str
    company_name: str
    role_title: str
    # Guardrail
    guardrail_passed: bool
    guardrail_message: str
    # Agent outputs (Pydantic → dict)
    fit_score: Optional[dict]       # FitScoreOutput
    tailored_cv: Optional[dict]     # TailoredCVOutput
    cover_letter: Optional[dict]    # CoverLetterOutput
    # HITL
    awaiting_human: bool
    human_feedback: Optional[dict]  # HumanFeedback
    # Routing
    routing_decision: Optional[str]
    # Output
    final_package: Optional[str]
    # Debug
    agent_log: list[str]
    error: Optional[str]
```

### 5.6 Graph Configuration (`graph.py`)

- **Framework:** LangGraph `StateGraph`
- **Checkpointer:** `MemorySaver` (in-memory, per-session)
- **Interrupts:** `interrupt_before=["hitl_review", "low_fit_warning"]`
- **Resume pattern:** `graph.update_state(config, feedback)` → `graph.stream(None, config)`
- **Nodes:** 7 (`guardrail`, `fit_scorer`, `low_fit_warning`, `cv_tailor`, `cover_letter_writer`, `hitl_review`, `assembler`)
- **Conditional edges:** 4 routing functions with explicit mapping dicts

---

## 6. Guardrails

| Guard | Implementation | Triggers |
|-------|---------------|---------|
| Empty input | String `.strip()` check | JD or resume is blank |
| Minimum length | Word count check | JD < 30 words or resume < 50 words |
| Prompt injection | String match on 8 known patterns | `"ignore previous instructions"`, `"you are now"`, `"jailbreak"`, etc. |
| No fabrication | System prompt hard constraint | Agent 2 instructed never to invent experience |
| HITL gate (low fit) | Graph interrupt at `low_fit_warning` | Score < 40 |
| HITL gate (review) | Graph interrupt at `hitl_review` | Always, before final assembly |
| No auto-send | Architecture constraint | Final package is a markdown download only — no external API calls |

---

## 7. Human-in-the-Loop Design

Two interrupt points exist in the graph:

**HITL #1 — Low Fit Warning** (conditional)
Fires when `fit_score < 40`. Surfaces the score, recommendation, and list of gaps to the user. User chooses:
- **Proceed anyway** → resumes to `cv_tailor`
- **Abort** → routes to `END`, nothing generated

**HITL #2 — Full Review** (always)
Fires after all three agents complete. User sees:
- Fit score with strengths and gaps
- Keyword match analysis
- Tailored CV (editable inline)
- Cover letter (editable inline)

User chooses:
- **Approve** → assembler runs with agent outputs as-is
- **Edit + Approve** → assembler uses human-edited versions
- **Reject** → routes to `END`, nothing saved

---

## 8. Evaluation Plan

Five test cases in `evals/run_evals.py`:

| # | Name | Input | What is asserted |
|---|------|-------|-----------------|
| 1 | Happy path | Strong-match SWE candidate vs SWE JD | Guardrail passes, score ≥ 60, routing = `proceed`, all three agent outputs populated, final package assembled |
| 2 | Empty resume guardrail | Valid JD + empty resume | `guardrail_passed = False`, no agents run, message references resume |
| 3 | Prompt injection | JD containing `"ignore previous instructions"` | `guardrail_passed = False`, routing = `abort`, no LLM calls made |
| 4 | Low fit routing | Web developer vs neurosurgeon JD | Score < 40, `routing_decision = "low_fit_warning"`, graph paused at interrupt |
| 5 | Structured output validation | Data analyst vs data analyst JD | `FitScoreOutput` validates with Pydantic, score in 0–100, `TailoredCVOutput` has ≥ 3 bullets, cover letter word count 150–500 |

---

## 9. Observability

- **LangSmith tracing:** Enabled via `LANGCHAIN_TRACING_V2=true` in `.env`. All node executions, LLM calls, and tool calls are captured as spans in the `job-assistant-capstone` project.
- **Agent log:** Every node writes a human-readable string to `GraphState.agent_log`. The Streamlit UI exposes this in an expandable debug panel.
- **Error field:** `GraphState.error` captures exception strings from agent fallback paths without crashing the graph.

---

## 10. Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Orchestration | LangGraph | ≥ 0.2.0 |
| LLM framework | LangChain | ≥ 0.3.0 |
| LLM | OpenAI GPT-4o-mini | via `langchain-openai` |
| Embeddings | OpenAI text-embedding-3-small | via `langchain-openai` |
| Vector store | ChromaDB | ≥ 0.5.0 |
| Structured outputs | Pydantic v2 | ≥ 2.0.0 |
| UI | Streamlit | ≥ 1.38.0 |
| Tracing | LangSmith | via env vars |
| Runtime | Python 3.10+ | — |

---

## 11. File Structure

```
job_assistant/
├── PRD.md                          ← this document
├── README.md                       ← setup and run instructions
├── CONTRIBUTIONS.md                ← individual member contributions
├── requirements.txt
├── .env.example
├── state.py                        ← GraphState + Pydantic models
├── graph.py                        ← LangGraph StateGraph
├── main.py                         ← CLI runner with HITL loop
├── app.py                          ← Streamlit UI
├── agents/
│   ├── guardrail.py                ← input validation node
│   ├── fit_scorer.py               ← Agent 1
│   ├── cv_tailor.py                ← Agent 2
│   ├── cover_letter_writer.py      ← Agent 3
│   └── hitl_and_assembler.py       ← HITL nodes + assembler
├── tools/
│   └── tools.py                    ← extract_keywords, rewrite_bullets, validate_cover_letter
├── rag/
│   └── rag_setup.py                ← ChromaDB setup + retrieve_context()
└── evals/
    └── run_evals.py                ← 5 evaluation scenarios
```

---

## 12. Limitations & Known Issues

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| RAG knowledge base is static | Industry context may be outdated | Extend `KNOWLEDGE_BASE` list in `rag_setup.py` |
| No multi-session memory | Each run is independent | Use `MemorySaver` with persistent `thread_id` per user |
| Keyword extraction is rule-based | May miss domain-specific jargon | Replace with LLM-based extraction for production |
| Self-correction limited to 1 retry | Cover letter may still have minor issues | Increase retry count or add a critic agent |
| Prompt injection detection is string-match only | Sophisticated injections may bypass | Add LLM-based intent classifier as secondary check |
| GPT-4o-mini quality ceiling | Niche or highly technical roles may produce weak output | Swap to GPT-4o for high-stakes runs |

---

*Document prepared for Multi Agent Orchestration [AI/ML] capstone submission.*
