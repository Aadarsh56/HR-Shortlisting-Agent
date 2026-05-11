"""
LangGraph State Graph — Pipeline Orchestration

Connects all 4 agents:
  Parser Agent → Scorer Agent → Critic Agent → Report Agent

With a conditional edge: Critic can loop back to Scorer if confidence is low.
"""

from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph


# ──────────────────────────────────────────────
# State Definition (TypedDict for LangGraph)
# ──────────────────────────────────────────────

class PipelineState(TypedDict):
    """State passed between all agents in the pipeline."""
    jd_raw: str
    jd_parsed: dict | None
    resumes_raw: list[dict]
    candidates: list[dict]
    evaluations: list[dict]
    retry_candidates: list[str]
    retry_count: dict[str, int]
    reports: dict
    overrides: list[dict]
    errors: list[str]
    session_id: str


# ──────────────────────────────────────────────
# Import agent nodes
# ──────────────────────────────────────────────

from agents.parser_agent import parser_node
from agents.scorer_agent import scorer_node
from agents.critic_agent import critic_node
from agents.report_agent import report_node


# ──────────────────────────────────────────────
# Conditional Edge: Should Critic re-score?
# ──────────────────────────────────────────────

def should_rescore(state: PipelineState) -> str:
    """Determine if candidates need re-scoring based on Critic's confidence.

    Returns:
        'rescore' to loop back to Scorer, 'continue' to proceed to Report.
    """
    retry_candidates = state.get("retry_candidates", [])
    if retry_candidates:
        return "rescore"
    return "continue"


# ──────────────────────────────────────────────
# Build the Graph
# ──────────────────────────────────────────────

def build_pipeline(use_critic: bool = True) -> StateGraph:
    """Build and compile the LangGraph pipeline.

    Args:
        use_critic: If False, skip Critic Agent for faster runs (Parser→Scorer→Report).
    """
    graph = StateGraph(PipelineState)

    graph.add_node("parser", parser_node)
    graph.add_node("scorer", scorer_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reporter", report_node)

    graph.set_entry_point("parser")
    graph.add_edge("parser", "scorer")

    if use_critic:
        graph.add_edge("scorer", "critic")
        graph.add_conditional_edges(
            "critic",
            should_rescore,
            {"rescore": "scorer", "continue": "reporter"},
        )
    else:
        # Skip Critic — go straight to Report
        graph.add_edge("scorer", "reporter")

    graph.add_edge("reporter", END)
    return graph.compile()


def run_pipeline(
    jd_text: str,
    resumes: list[dict],
    session_id: str = None,
    use_critic: bool = True,
) -> dict:
    """Run the complete evaluation pipeline.

    Args:
        jd_text: Raw job description text.
        resumes: List of dicts with {filename, content, source}.
        session_id: Optional session identifier.
        use_critic: Whether to run the Critic validation agent.
    """
    import uuid

    if not session_id:
        session_id = str(uuid.uuid4())[:8]

    pipeline = build_pipeline(use_critic=use_critic)

    initial_state: PipelineState = {
        "jd_raw": jd_text,
        "jd_parsed": None,
        "resumes_raw": resumes,
        "candidates": [],
        "evaluations": [],
        "retry_candidates": [],
        "retry_count": {},
        "reports": {},
        "overrides": [],
        "errors": [],
        "session_id": session_id,
    }

    return pipeline.invoke(initial_state)

