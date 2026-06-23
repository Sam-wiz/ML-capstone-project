"""
main.py — CLI runner for the Job Application Assistant.

Usage:
    python main.py                          # uses built-in sample inputs
    python main.py --jd jd.txt --resume cv.txt   # load from files

HITL prompts appear in the terminal.
Set OPENAI_API_KEY in .env before running.
"""

import argparse
from pathlib import Path
from uuid import uuid4
from dotenv import load_dotenv
load_dotenv()

from graph import graph, print_graph_structure
from state import HumanFeedback
from config import has_openai_api_key

# ── Sample inputs (swap these for real content) ───────────────────────────────

SAMPLE_JD = """
Software Engineer — Backend (Python)
Acme Corp | London, UK | Full-time

About the role:
We are looking for a Backend Software Engineer to join our platform team.
You will design, build, and maintain scalable Python services that power our
B2B SaaS product used by 500+ enterprise customers.

Responsibilities:
- Design and implement RESTful APIs using Python (FastAPI or Django REST Framework)
- Work with PostgreSQL and Redis for data storage and caching
- Deploy and monitor services on AWS (EC2, Lambda, RDS)
- Collaborate cross-functionally with product managers and frontend engineers
- Write unit and integration tests; maintain >80% code coverage
- Participate in code reviews and contribute to engineering best practices

Requirements:
- 3+ years of backend Python experience
- Strong knowledge of SQL and database design
- Experience with AWS services (EC2, S3, Lambda, RDS)
- Familiarity with Docker and CI/CD pipelines (GitHub Actions or similar)
- Understanding of REST API design principles
- Excellent communication and collaboration skills

Nice to have:
- Experience with FastAPI
- Knowledge of Redis or other caching layers
- Prior experience at a SaaS company
"""

SAMPLE_RESUME = """
Jane Smith
jane.smith@email.com | github.com/janesmith | London, UK

PROFESSIONAL SUMMARY
Python developer with 4 years of experience building web applications and data pipelines.
Passionate about clean code and scalable architecture.

EXPERIENCE

Backend Developer — TechStart Ltd (2021–Present)
• Built and maintained Django REST APIs serving 50k daily requests
• Designed PostgreSQL schemas for a multi-tenant application
• Migrated monolith services to Docker containers, reducing deployment time by 60%
• Wrote comprehensive unit tests achieving 85% code coverage
• Collaborated with a cross-functional team of 8 engineers and 3 product managers

Junior Developer — DataCo (2020–2021)
• Developed Python scripts for ETL data pipelines processing 1M records/day
• Created internal dashboards using Flask and Chart.js
• Assisted in debugging and optimising slow SQL queries (reduced query time by 40%)

EDUCATION
BSc Computer Science — University of Manchester (2020) — 2:1

SKILLS
Python, Django, Flask, PostgreSQL, MySQL, Git, Docker, Linux, REST APIs, HTML/CSS
"""


def run_with_hitl():
    """Run the full graph with CLI-based human-in-the-loop."""
    if not has_openai_api_key():
        print("OPENAI_API_KEY is not configured. Add it to .env before running the full demo.")
        return

    print_graph_structure()

    thread_config = {"configurable": {"thread_id": f"demo-{uuid4()}"}}

    initial_state = {
        "job_description": SAMPLE_JD,
        "raw_resume": SAMPLE_RESUME,
        "company_name": "",
        "role_title": "",
        "guardrail_passed": False,
        "guardrail_message": "",
        "fit_score": None,
        "tailored_cv": None,
        "cover_letter": None,
        "human_feedback": None,
        "awaiting_human": False,
        "routing_decision": None,
        "final_package": None,
        "agent_log": [],
        "error": None,
    }

    print("=" * 60)
    print("Running graph... (this may take 20-40 seconds)")
    print("=" * 60)

    # ── First run: graph runs until first INTERRUPT ───────────────────────────
    events = list(graph.stream(initial_state, thread_config, stream_mode="values"))
    current_state = events[-1] if events else initial_state

    print("\n--- Agent Log ---")
    for entry in current_state.get("agent_log", []):
        print(f"  {entry}")

    # ── Check if aborted at guardrail ─────────────────────────────────────────
    if not current_state.get("guardrail_passed"):
        print(f"\n❌ Guardrail blocked: {current_state.get('guardrail_message')}")
        return

    # ── Handle low fit warning interrupt ─────────────────────────────────────
    snapshot = graph.get_state(thread_config)
    if snapshot.next and "low_fit_warning" in str(snapshot.next):
        fit = current_state.get("fit_score", {})
        print(f"\n⚠️  LOW FIT WARNING — Score: {fit.get('score')}/100")
        print(f"   Recommendation: {fit.get('recommendation')}")
        print(f"   Gaps: {fit.get('gaps', [])}")
        proceed = input("\nDo you want to proceed with tailoring anyway? (y/n): ").strip().lower()
        decision = "approve" if proceed == "y" else "reject"
        graph.update_state(
            thread_config,
            {"human_feedback": HumanFeedback(decision=decision).model_dump(), "awaiting_human": False},
        )
        events = list(graph.stream(None, thread_config, stream_mode="values"))
        current_state = events[-1] if events else current_state
        if decision == "reject":
            print("Application aborted by user.")
            return

    # ── HITL review interrupt ─────────────────────────────────────────────────
    snapshot = graph.get_state(thread_config)
    if snapshot.next and "hitl_review" in str(snapshot.next):
        fit = current_state.get("fit_score", {})
        cv = current_state.get("tailored_cv", {})
        cl = current_state.get("cover_letter", {})

        print("\n" + "=" * 60)
        print("HUMAN REVIEW — Please review the draft outputs below")
        print("=" * 60)

        print(f"\n📊 FIT SCORE: {fit.get('score')}/100 ({fit.get('recommendation')})")
        print(f"   Strengths: {fit.get('strengths', [])}")
        print(f"   Gaps: {fit.get('gaps', [])}")

        print("\n📄 TAILORED CV SUMMARY:")
        print(f"   {cv.get('summary', '')[:200]}...")
        print(f"   Bullets: {len(cv.get('tailored_bullets', []))} rewritten")

        print("\n✉️  COVER LETTER (first 300 chars):")
        print(f"   {cl.get('full_text', '')[:300]}...")

        print("\n" + "-" * 40)
        decision_input = input("Decision — (a)pprove / (e)dit / (r)eject: ").strip().lower()

        if decision_input == "r":
            feedback = HumanFeedback(decision="reject")
            graph.update_state(
                thread_config,
                {"human_feedback": feedback.model_dump(), "awaiting_human": False},
            )
            events = list(graph.stream(None, thread_config, stream_mode="values"))
            print("\nApplication rejected. Nothing sent.")
            return

        edited_cv = None
        edited_cl = None
        notes = None

        if decision_input == "e":
            print("\nEditing mode — press Enter to keep existing, or type replacement:")
            edited_cv_input = input("Paste edited CV summary (or Enter to keep): ").strip()
            edited_cl_input = input("Paste edited cover letter (or Enter to keep): ").strip()
            notes = input("Any notes to add: ").strip()
            edited_cv = edited_cv_input or None
            edited_cl = edited_cl_input or None

        feedback = HumanFeedback(
            decision="approve" if decision_input == "a" else "edit",
            edited_cv=edited_cv,
            edited_cover_letter=edited_cl,
            feedback_notes=notes,
        )

        graph.update_state(
            thread_config,
            {"human_feedback": feedback.model_dump(), "awaiting_human": False},
        )

        print("\nResuming graph after approval...")
        events = list(graph.stream(None, thread_config, stream_mode="values"))
        current_state = events[-1] if events else current_state

    # ── Print final output ────────────────────────────────────────────────────
    final = current_state.get("final_package")
    if final:
        print("\n" + "=" * 60)
        print("✅ FINAL APPLICATION PACKAGE")
        print("=" * 60)
        print(final)

        output_path = "output_package.md"
        with open(output_path, "w") as f:
            f.write(final)
        print(f"\n📁 Saved to {output_path}")
    else:
        print("\n⚠️  No final package generated. Check agent logs above.")

    print("\n--- Final Agent Log ---")
    for entry in current_state.get("agent_log", []):
        print(f"  {entry}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Job Application Assistant — multi-agent pipeline",
    )
    parser.add_argument("--jd", type=Path, default=None, metavar="FILE",
                        help="Path to a text file containing the job description")
    parser.add_argument("--resume", type=Path, default=None, metavar="FILE",
                        help="Path to a text file containing your resume")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.jd:
        if not args.jd.exists():
            print(f"Error: JD file not found: {args.jd}")
            raise SystemExit(1)
        SAMPLE_JD = args.jd.read_text(encoding="utf-8")
        print(f"Loaded JD from {args.jd} ({len(SAMPLE_JD.split())} words)")

    if args.resume:
        if not args.resume.exists():
            print(f"Error: Resume file not found: {args.resume}")
            raise SystemExit(1)
        SAMPLE_RESUME = args.resume.read_text(encoding="utf-8")
        print(f"Loaded resume from {args.resume} ({len(SAMPLE_RESUME.split())} words)")

    run_with_hitl()
