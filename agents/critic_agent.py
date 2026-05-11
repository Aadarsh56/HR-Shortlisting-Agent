"""
Critic Agent — LangGraph Node

Validates scoring output for consistency, hallucinations, confidence, and bias.
Can trigger re-scoring if confidence is below threshold.
"""

import json
import os
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
MAX_RETRIES = int(os.getenv("MAX_SCORER_RETRIES", "2"))


def _load_prompt(filename: str) -> str:
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


def _get_llm():
    from . import get_langfuse_callbacks
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3")
    return ChatOllama(model=model, base_url=base_url, temperature=0, format="json",
                      callbacks=get_langfuse_callbacks(),
                      num_predict=400, num_ctx=4096)


def critique_evaluation(evaluation: dict, candidate_raw_text: str) -> dict:
    """Critique a single candidate's evaluation."""
    llm = _get_llm()
    system_prompt = _load_prompt("critic.txt")

    scores_summary = ""
    for ds in evaluation.get("dimension_scores", []):
        scores_summary += (
            f"- {ds['dimension']} ({ds['weight']*100:.0f}%): "
            f"Score={ds['score']}/10 — \"{ds['justification']}\"\n"
        )

    user_prompt = (
        f"EVALUATION TO REVIEW:\n"
        f"Candidate: {evaluation.get('candidate', {}).get('name', 'Unknown')}\n"
        f"Weighted Total: {evaluation.get('weighted_total', 0):.1f}/100\n\n"
        f"DIMENSION SCORES:\n{scores_summary}\n"
        f"RESUME TEXT (first 2000 chars):\n{candidate_raw_text[:2000]}\n\n"
        f"Review this evaluation for consistency, hallucinations, and bias."
    )

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        data = _extract_json(response.content)
        if data:
            return {
                "confidence": min(1.0, max(0.0, float(data.get("confidence", 0.7)))),
                "issues": data.get("issues", []),
                "bias_flags": data.get("bias_flags", []),
                "overall_assessment": data.get("overall_assessment", "Review complete."),
            }
    except Exception:
        pass

    return {"confidence": 0.75, "issues": [], "bias_flags": [],
            "overall_assessment": "Default confidence assigned."}


def critic_node(state: dict) -> dict:
    """LangGraph node: Validate all evaluations."""
    evaluations = state.get("evaluations", [])
    candidates = state.get("candidates", [])
    retry_count = dict(state.get("retry_count", {}))
    errors = list(state.get("errors", []))
    retry_candidates = []

    candidate_texts = {c.get("name", ""): c.get("raw_text", "") for c in candidates}

    for ev in evaluations:
        name = ev.get("candidate", {}).get("name", "Unknown")
        raw_text = candidate_texts.get(name, "")

        try:
            critique = critique_evaluation(ev, raw_text)
            ev["confidence"] = critique["confidence"]
            ev["bias_flags"] = critique.get("bias_flags", [])

            # Apply suggested corrections
            for issue in critique.get("issues", []):
                if issue.get("suggested_score") is not None:
                    for ds in ev.get("dimension_scores", []):
                        if ds["dimension"] == issue.get("dimension", ""):
                            ds["score"] = max(0, min(10, int(issue["suggested_score"])))

            # Check if re-scoring needed
            if critique["confidence"] < CONFIDENCE_THRESHOLD:
                retries = retry_count.get(name, 0)
                if retries < MAX_RETRIES:
                    retry_candidates.append(name)
                    retry_count[name] = retries + 1
        except Exception as e:
            ev["confidence"] = 0.7
            errors.append(f"Critic error for {name}: {str(e)}")

        # Recalculate weighted total
        total = sum(ds["score"] * ds["weight"] * 10 for ds in ev.get("dimension_scores", []))
        ev["weighted_total"] = round(total, 2)
        ev["recommendation"] = "Hire" if total >= 75 else "Maybe" if total >= 50 else "No Hire"

    return {**state, "evaluations": evaluations, "retry_candidates": retry_candidates,
            "retry_count": retry_count, "errors": errors}
