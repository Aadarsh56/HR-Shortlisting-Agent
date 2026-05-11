"""
Report Agent — LangGraph Node

Ranks candidates, computes skills gap, and generates reports in PDF/HTML/JSON.
"""

import uuid
from datetime import datetime

from models.schemas import SCORING_DIMENSIONS
from services.skills_gap import analyse_skills_gap
from services.report_generator import generate_pdf_report, generate_html_report, generate_json_report


def report_node(state: dict) -> dict:
    """LangGraph node: Rank candidates, fill skills gap, generate reports."""
    jd = state.get("jd_parsed", {})
    evaluations = list(state.get("evaluations", []))
    overrides = state.get("overrides", [])
    errors = list(state.get("errors", []))

    if not evaluations:
        errors.append("No evaluations to report.")
        return {**state, "errors": errors}

    # 1. Compute skills gap for each candidate
    required_skills = jd.get("required_skills", [])
    preferred_skills = jd.get("preferred_skills", [])

    for ev in evaluations:
        candidate = ev.get("candidate", {})
        candidate_skills = candidate.get("skills", [])

        try:
            gap = analyse_skills_gap(required_skills, preferred_skills, candidate_skills)
            ev["skills_gap"] = gap.model_dump()
        except Exception as e:
            ev["skills_gap"] = {
                "matched_skills": [], "missing_skills": [],
                "bonus_skills": [], "match_percentage": 0.0,
            }
            errors.append(f"Skills gap error for {candidate.get('name', '?')}: {e}")

    # 2. Rank by weighted_total (descending)
    evaluations.sort(key=lambda e: e.get("weighted_total", 0), reverse=True)
    for i, ev in enumerate(evaluations):
        ev["rank"] = i + 1

    # 3. Build report data
    hire_count = sum(1 for e in evaluations if e.get("recommendation") == "Hire")
    maybe_count = sum(1 for e in evaluations if e.get("recommendation") == "Maybe")
    no_hire_count = sum(1 for e in evaluations if e.get("recommendation") == "No Hire")

    import os
    session_id = state.get("session_id", str(uuid.uuid4())[:8])
    report_data = {
        "session_id": session_id,
        "jd": jd,
        "evaluations": evaluations,
        "overrides": overrides,
        "generated_at": datetime.now().isoformat(),
        "model_used": os.getenv("OLLAMA_MODEL", "llama3"),
        "total_candidates": len(evaluations),
        "hire_count": hire_count,
        "maybe_count": maybe_count,
        "no_hire_count": no_hire_count,
    }

    # 4. Generate reports
    reports = {}
    try:
        reports["pdf"] = generate_pdf_report(report_data)
    except Exception as e:
        errors.append(f"PDF generation error: {e}")

    try:
        reports["html"] = generate_html_report(report_data)
    except Exception as e:
        errors.append(f"HTML generation error: {e}")

    try:
        reports["json"] = generate_json_report(report_data)
    except Exception as e:
        errors.append(f"JSON generation error: {e}")

    return {
        **state,
        "evaluations": evaluations,
        "reports": reports,
        "errors": errors,
    }
