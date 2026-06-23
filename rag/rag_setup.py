"""
rag/rag_setup.py — RAG pipeline using Chroma (local, no account needed).

Embeds career/industry knowledge docs and exposes a retriever.
Agents call retrieve_context(query) to ground their outputs.
"""

import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import has_openai_api_key


# ── Knowledge base documents ──────────────────────────────────────────────────
# In a real project: load PDFs / web pages. Here we embed inline knowledge
# so the project works immediately without extra files.

KNOWLEDGE_BASE = [
    Document(
        page_content="""
        Software Engineering Job Market — Key Trends (2024-2025):
        Employers increasingly prioritise cloud experience (AWS, GCP, Azure), CI/CD pipelines,
        and system design skills over pure coding ability. Full-stack roles often require React + Node
        or Python FastAPI. Machine learning integration is expected even in non-ML roles.
        Remote-first companies look for strong async communication and self-management.
        Typical SWE interview process: resume screen → technical phone screen → take-home or
        live coding → system design round → culture/behavioural round.
        """,
        metadata={"source": "industry_context", "domain": "software_engineering"},
    ),
    Document(
        page_content="""
        Data Science & Analytics Roles — What Recruiters Look For:
        Strong SQL is non-negotiable. Python (pandas, scikit-learn) is the standard.
        Business impact framing is critical — quantify outcomes (increased revenue by X%, reduced churn by Y%).
        Visualisation skills (Tableau, Power BI, matplotlib) matter as much as modelling.
        Communication of technical findings to non-technical stakeholders is a top differentiator.
        Portfolio projects on GitHub or Kaggle significantly strengthen applications.
        Entry roles typically require 1-2 years experience or strong internship/project history.
        """,
        metadata={"source": "industry_context", "domain": "data_science"},
    ),
    Document(
        page_content="""
        Product Management — Essential Competencies:
        PMs need a combination of strategic thinking, customer empathy, and technical fluency.
        Key skills: roadmap prioritisation (RICE, MoSCoW), stakeholder management, A/B testing,
        metrics definition (North Star, OKRs, KPIs), user research, and cross-functional leadership.
        Strong PM cover letters emphasise specific product decisions and their measurable outcomes.
        Certifications (AIPMM, Pragmatic) are valued but not required. MBA from top school helps
        for senior roles at large companies.
        """,
        metadata={"source": "industry_context", "domain": "product_management"},
    ),
    Document(
        page_content="""
        Cover Letter Best Practices:
        The best cover letters are 3-4 paragraphs, 250-350 words. They open with a specific hook
        tied to the company (not "I am applying for..."). Paragraph 2 links your strongest experience
        directly to the role's top requirements using concrete metrics. Paragraph 3 explains genuine
        interest in this company specifically — mission, product, recent news. Closing requests a
        conversation, not "consideration of my application."
        Avoid generic phrases: "team player", "passionate", "hard worker", "fast learner."
        Use active verbs: led, built, shipped, reduced, increased, designed, launched.
        Tailor every letter — recruiters spot copy-paste immediately.
        """,
        metadata={"source": "cover_letter_guide", "domain": "general"},
    ),
    Document(
        page_content="""
        Resume Tailoring — ATS and Human Optimisation:
        Applicant Tracking Systems (ATS) scan for exact keyword matches from the JD.
        Use the exact phrases from the JD (e.g. if JD says "cross-functional collaboration",
        use that phrase, not "working across teams"). Quantify every bullet: instead of
        "improved performance", write "reduced API latency by 40% serving 2M daily requests."
        Standard ATS-friendly format: simple fonts, no tables/columns, PDF or Word.
        Keep to 1 page for < 5 years experience, 2 pages max otherwise.
        Summary section should mirror the exact role title from the JD.
        """,
        metadata={"source": "resume_guide", "domain": "general"},
    ),
    Document(
        page_content="""
        Startup vs Enterprise Job Applications:
        Startup applications: emphasise versatility, ownership, speed, and tolerance for ambiguity.
        Show you can wear multiple hats. Mention specific startup technologies (AWS Lambda, Stripe,
        Heroku, Vercel). Highlight scrappy wins: "built and shipped X in 2 weeks."
        Enterprise applications: emphasise process, scale, compliance, and stakeholder management.
        Use enterprise language: "drove alignment across 4 business units", "delivered within
        governance framework", "managed $2M budget."
        Research the company stage and tone-match your application accordingly.
        """,
        metadata={"source": "application_strategy", "domain": "general"},
    ),
    Document(
        page_content="""
        Marketing & Growth Roles — Key Requirements:
        Digital marketing: SEO/SEM, Google Analytics, Meta Ads, email marketing (Mailchimp/HubSpot),
        content strategy, conversion rate optimisation (CRO).
        Growth roles: SQL for self-serve analysis, A/B testing, funnel analysis, LTV/CAC modelling.
        Brand/communications: copywriting, brand voice consistency, PR, stakeholder communication.
        Strong portfolio of campaigns with measurable results (CTR, ROAS, conversion rates) is essential.
        """,
        metadata={"source": "industry_context", "domain": "marketing"},
    ),
]


_retriever = None


def _fallback_context(query: str) -> str:
    """Return relevant inline knowledge without embeddings when OpenAI is unavailable."""
    query_terms = {term.lower().strip(".,:/()") for term in query.split() if len(term) > 2}
    ranked = []
    for doc in KNOWLEDGE_BASE:
        text = doc.page_content.lower()
        score = sum(1 for term in query_terms if term in text)
        ranked.append((score, doc))
    ranked.sort(key=lambda item: item[0], reverse=True)
    docs = [doc for score, doc in ranked if score > 0][:3] or [doc for _, doc in ranked[:3]]
    return "\n\n---\n\n".join(d.page_content.strip() for d in docs)

def get_retriever(persist_directory: str = "./chroma_db"):
    """
    Build or load the Chroma vector store and return a retriever.
    Persists to disk so it only embeds once.
    """
    global _retriever
    if _retriever is not None:
        return _retriever
    if not has_openai_api_key():
        raise RuntimeError("OPENAI_API_KEY is required for Chroma/OpenAI embedding retrieval.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(KNOWLEDGE_BASE)

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        try:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings,
            )
            # Smoke-test: verify the store is readable before trusting it
            vectorstore.similarity_search("test", k=1)
        except Exception:
            import shutil
            shutil.rmtree(persist_directory, ignore_errors=True)
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory,
            )
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

    _retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3},
    )
    return _retriever


def retrieve_context(query: str, persist_directory: str = "./chroma_db") -> str:
    """
    Retrieve relevant context from the knowledge base for a given query.
    Returns a single string of concatenated relevant chunks.
    """
    try:
        retriever = get_retriever(persist_directory)
        docs = retriever.invoke(query)
        if not docs:
            return "No specific context found. Use general best practices."
        return "\n\n---\n\n".join(d.page_content.strip() for d in docs)
    except Exception as exc:
        return (
            "Embedding retrieval unavailable; using inline fallback context. "
            f"Reason: {exc}\n\n{_fallback_context(query)}"
        )
