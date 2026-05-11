"""
Scorer Agent — LangGraph Node

Scores each candidate against the JD across 5 weighted dimensions.
Uses Llama 3 for evaluation with the mandatory rubric.
"""

import json
import os
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from models.schemas import (
    CandidateEvaluation,
    DimensionScore,
    Recommendation,
    SCORING_DIMENSIONS,
)

PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


def _load_prompt(filename: str) -> str:
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return {}


def _get_llm():
    from . import get_langfuse_callbacks
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3")
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0,
        format="json",
        num_predict=500,
        callbacks=get_langfuse_callbacks(),
        num_ctx=3072,  # Reduced from 4096 to speed up prefill evaluation
    )


def _compute_weighted_total(dimension_scores: list[DimensionScore]) -> float:
    """Compute the weighted total score (0-100 scale).

    Formula: sum(score × weight × 10) for each dimension.
    """
    total = 0.0
    for ds in dimension_scores:
        total += ds.score * ds.weight * 10
    return round(total, 2)


def _determine_recommendation(weighted_total: float) -> Recommendation:
    """Determine hire recommendation based on weighted total."""
    if weighted_total >= 75:
        return Recommendation.HIRE
    elif weighted_total >= 50:
        return Recommendation.MAYBE
    return Recommendation.NO_HIRE


def score_candidate(jd: dict, candidate: dict) -> CandidateEvaluation:
    """Score a single candidate against the JD.

    Args:
        jd: Parsed JobDescription as dict.
        candidate: Parsed CandidateProfile as dict.

    Returns:
        CandidateEvaluation with dimension scores.
    """
    llm = _get_llm()
    system_prompt = _load_prompt("scorer.txt")

    # Use anonymised text for scoring to prevent bias
    scoring_text = candidate.get("anonymised_text") or candidate.get("raw_text", "")

    # Build context for the scorer
    user_prompt = f"""
JOB DESCRIPTION:
- Title: {jd.get('title', 'N/A')}
- Required Skills: {', '.join(jd.get('required_skills', []))}
- Preferred Skills: {', '.join(jd.get('preferred_skills', []))}
- Min Experience: {jd.get('min_experience_years', 0)} years
- Education Required: {jd.get('education_requirement', 'Not specified')}
- Certifications: {', '.join(jd.get('certifications', []))}
- Domain: {jd.get('domain', 'Not specified')}

CANDIDATE PROFILE:
- Skills: {', '.join(candidate.get('skills', []))}
- Total Experience: {candidate.get('total_experience_years', 0)} years
- Education: {json.dumps(candidate.get('education', []))}
- Projects: {json.dumps(candidate.get('projects', []))}
- Certifications: {', '.join(candidate.get('certifications', []))}
- Summary: {candidate.get('summary', 'N/A')}

RESUME TEXT (for communication quality assessment):
{scoring_text[:3000]}

Score this candidate across all 5 dimensions.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    dimension_scores = []

    for attempt in range(3):
        try:
            response = llm.invoke(messages)
            data = _extract_json(response.content)

            if data and "dimension_scores" in data:
                for ds_data in data["dimension_scores"]:
                    # Coerce None/invalid values defensively
                    score_val = ds_data.get("score") or 5
                    weight_val = ds_data.get("weight") or 0.20
                    justification = ds_data.get("justification") or "No justification provided."
                    try:
                        score_val = max(0, min(10, int(float(score_val))))
                        weight_val = float(weight_val)
                    except (TypeError, ValueError):
                        score_val = 5
                        weight_val = 0.20

                    dimension_scores.append(DimensionScore(
                        dimension=str(ds_data.get("dimension") or "Unknown"),
                        weight=weight_val,
                        score=score_val,
                        justification=str(justification),
                    ))
                break
        except Exception:
            if attempt == 2:
                break
            continue

    # If LLM failed, generate default scores
    if not dimension_scores:
        for dim in SCORING_DIMENSIONS:
            dimension_scores.append(DimensionScore(
                dimension=dim["name"],
                weight=dim["weight"],
                score=5,
                justification="[Auto-generated] Scoring failed — default average score assigned.",
            ))

    # Ensure we have exactly 5 dimensions with correct weights
    dim_names = {ds.dimension for ds in dimension_scores}
    for dim in SCORING_DIMENSIONS:
        if dim["name"] not in dim_names:
            dimension_scores.append(DimensionScore(
                dimension=dim["name"],
                weight=dim["weight"],
                score=5,
                justification="[Auto-generated] Dimension not scored — default average assigned.",
            ))

    weighted_total = _compute_weighted_total(dimension_scores)
    recommendation = _determine_recommendation(weighted_total)

    from models.schemas import CandidateProfile, SkillsGap
    candidate_profile = CandidateProfile(**candidate)

    return CandidateEvaluation(
        candidate=candidate_profile,
        dimension_scores=dimension_scores,
        weighted_total=weighted_total,
        recommendation=recommendation,
        skills_gap=SkillsGap(),  # Will be filled by report agent
    )


def scorer_node(state: dict) -> dict:
    """LangGraph node: Score all candidates.

    Args:
        state: Current agent state.

    Returns:
        Updated state with evaluations.
    """
    jd = state.get("jd_parsed")
    candidates = state.get("candidates", [])
    errors = list(state.get("errors", []))
    retry_candidates = state.get("retry_candidates", [])

    if not jd:
        errors.append("Cannot score: JD not parsed.")
        return {**state, "errors": errors}

    evaluations = []
    existing_evals = state.get("evaluations", [])

    # If re-scoring specific candidates, keep existing evals for others
    if retry_candidates:
        for ev in existing_evals:
            name = ev.get("candidate", {}).get("name", "")
            if name not in retry_candidates:
                evaluations.append(ev)

    for candidate in candidates:
        name = candidate.get("name", "Unknown")

        # Only score if not already evaluated (unless it's a retry)
        if retry_candidates and name not in retry_candidates:
            continue

        if not retry_candidates:
            # Check if already evaluated
            already = any(
                ev.get("candidate", {}).get("name") == name
                for ev in evaluations
            )
            if already:
                continue

        try:
            evaluation = score_candidate(jd, candidate)
            evaluations.append(evaluation.model_dump())
        except Exception as e:
            errors.append(f"Scoring error for {name}: {str(e)}")

    return {
        **state,
        "evaluations": evaluations,
        "retry_candidates": [],  # Clear retries
        "errors": errors,
    }
