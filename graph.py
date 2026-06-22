"""
graph.py — Main LangGraph StateGraph.

Wires all nodes, edges, and conditional routing.
Exposes compile() → graph for use by main.py and app.py.

Graph flow:
  START
    → guardrail_node
    → [abort if failed] → END
    → fit_scorer_node
    → [low_fit_warning if score < 40] → low_fit_warning_node → INTERRUPT
        → [human says proceed] → cv_tailor_node
        → [human says abort]  → END
    → [score >= 40] → cv_tailor_node
    → cover_letter_writer_node
    → hitl_review_node  ← INTERRUPT (human reviews full draft)
    → [approve/edit] → assembler_node → END
    → [reject]        → END
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import GraphState
from agents.guardrail import guardrail_node
from agents.fit_scorer import fit_scorer_node
from agents.cv_tailor import cv_tailor_node
from agents.cover_letter_writer import cover_letter_writer_node
from agents.hitl_and_assembler import (
    low_fit_warning_node,
    hitl_review_node,
    assembler_node,
)


# ── Routing functions ─────────────────────────────────────────────────────────

def route_after_guardrail(state: GraphState) -> str:
    """After guardrail: proceed or abort."""
    if not state.get("guardrail_passed", False):
        return "abort"
    return "fit_scorer"


def route_after_fit_score(state: GraphState) -> str:
    """After fit scoring: low fit warning or proceed to CV tailoring."""
    if state.get("routing_decision") == "low_fit_warning":
        return "low_fit_warning"
    return "cv_tailor"


def route_after_low_fit_warning(state: GraphState) -> str:
    """
    After low fit warning (HITL has responded):
    If human said proceed → cv_tailor. If reject → abort.
    """
    feedback = state.get("human_feedback", {})
    if feedback and feedback.get("decision") == "reject":
        return "abort"
    return "cv_tailor"


def route_after_hitl_review(state: GraphState) -> str:
    """
    After HITL review: approve/edit → assemble. Reject → abort.
    """
    feedback = state.get("human_feedback", {})
    if feedback and feedback.get("decision") == "reject":
        return "abort"
    return "assembler"


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("guardrail", guardrail_node)
    builder.add_node("fit_scorer", fit_scorer_node)
    builder.add_node("low_fit_warning", low_fit_warning_node)
    builder.add_node("cv_tailor", cv_tailor_node)
    builder.add_node("cover_letter_writer", cover_letter_writer_node)
    builder.add_node("hitl_review", hitl_review_node)
    builder.add_node("assembler", assembler_node)

    # Entry edge
    builder.add_edge(START, "guardrail")

    # Guardrail routing
    builder.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "fit_scorer": "fit_scorer",
            "abort": END,
        },
    )

    # Fit scorer routing
    builder.add_conditional_edges(
        "fit_scorer",
        route_after_fit_score,
        {
            "low_fit_warning": "low_fit_warning",
            "cv_tailor": "cv_tailor",
        },
    )

    # Low fit warning → human decides → cv_tailor or abort
    builder.add_conditional_edges(
        "low_fit_warning",
        route_after_low_fit_warning,
        {
            "cv_tailor": "cv_tailor",
            "abort": END,
        },
    )

    # CV tailor → cover letter (always)
    builder.add_edge("cv_tailor", "cover_letter_writer")

    # Cover letter → HITL review
    builder.add_edge("cover_letter_writer", "hitl_review")

    # HITL review routing
    builder.add_conditional_edges(
        "hitl_review",
        route_after_hitl_review,
        {
            "assembler": "assembler",
            "abort": END,
        },
    )

    # Assembler → done
    builder.add_edge("assembler", END)

    # MemorySaver enables interrupt() / resume pattern
    memory = MemorySaver()
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["hitl_review", "low_fit_warning"],
    )


graph = build_graph()


# ── Graph visualisation helper (optional) ────────────────────────────────────

def print_graph_structure():
    """Print a simple text representation of the graph."""
    print("\n=== Job Application Assistant — Graph Structure ===")
    print("START")
    print("  → guardrail_node")
    print("     [abort if guardrail fails] → END")
    print("  → fit_scorer_node")
    print("     [score < 40] → low_fit_warning_node  ← INTERRUPT")
    print("        [reject] → END")
    print("        [proceed] → cv_tailor_node")
    print("     [score >= 40] → cv_tailor_node")
    print("  → cover_letter_writer_node")
    print("  → hitl_review_node  ← INTERRUPT")
    print("     [reject] → END")
    print("     [approve/edit] → assembler_node")
    print("  → END")
    print("===================================================\n")
