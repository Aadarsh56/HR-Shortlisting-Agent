"""
Unit tests for services and models.
Tests work without Ollama (no LLM calls).
"""
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.schemas import (
    JobDescription, CandidateProfile, DimensionScore,
    SkillsGap, SkillWithResource, CandidateEvaluation, Recommendation
)
from services.skills_gap import analyse_skills_gap, normalise_skill
from services.bias_detector import anonymise_text, generate_fairness_report
from services.linkedin_parser import parse_linkedin_json


# ── skills_gap tests ──────────────────────────────────────

def test_skills_gap_perfect_match():
    gap = analyse_skills_gap(
        required_skills=["python", "docker"],
        preferred_skills=["kubernetes"],
        candidate_skills=["python", "docker", "fastapi"],
    )
    assert gap.match_percentage == 100.0
    assert "python" in gap.matched_skills
    assert "docker" in gap.matched_skills
    assert len(gap.missing_skills) == 0
    assert "fastapi" in gap.bonus_skills
    print("[OK] test_skills_gap_perfect_match")


def test_skills_gap_partial_match():
    gap = analyse_skills_gap(
        required_skills=["python", "docker", "postgresql", "kafka"],
        preferred_skills=["kubernetes"],
        candidate_skills=["python", "docker"],
    )
    assert gap.match_percentage == 50.0
    assert len(gap.missing_skills) == 2
    print("[OK] test_skills_gap_partial_match")


def test_skills_gap_no_match():
    # Use skills from completely unrelated domains so even semantic matching won't fire
    gap = analyse_skills_gap(
        required_skills=["cobol", "fortran"],
        preferred_skills=[],
        candidate_skills=["adobe photoshop", "video editing"],
    )
    assert gap.match_percentage == 0.0
    assert len(gap.bonus_skills) > 0
    print("[OK] test_skills_gap_no_match")



def test_learning_resource_known_skill():
    gap = analyse_skills_gap(
        required_skills=["docker"],
        preferred_skills=[],
        candidate_skills=[],
    )
    assert len(gap.missing_skills) == 1
    resource = gap.missing_skills[0]
    assert "docker" in resource.resource.lower()
    print("[OK] test_learning_resource_known_skill")


# ── bias_detector tests ───────────────────────────────────

def test_anonymise_removes_email():
    text = "John Smith\nemail: john.smith@example.com\n5 years Python experience"
    result = anonymise_text(text)
    assert "john.smith@example.com" not in result.lower()
    assert "python" in result.lower()
    print("[OK] test_anonymise_removes_email")


def test_anonymise_replaces_pronouns():
    text = "He worked at Google. She built REST APIs. His team won an award."
    result = anonymise_text(text)
    assert " He " not in result
    assert " She " not in result
    print("[OK] test_anonymise_replaces_pronouns")


def test_fairness_report_insufficient_data():
    result = generate_fairness_report([{"weighted_total": 80}])
    assert result["status"] == "insufficient_data"
    print("[OK] test_fairness_report_insufficient_data")


def test_fairness_report_multiple_candidates():
    evals = [
        {"weighted_total": 85, "candidate": {"name": "A"}},
        {"weighted_total": 60, "candidate": {"name": "B"}},
        {"weighted_total": 45, "candidate": {"name": "C"}},
    ]
    result = generate_fairness_report(evals)
    assert result["status"] == "completed"
    assert result["total_candidates"] == 3
    assert result["mean_score"] > 0
    print("[OK] test_fairness_report_multiple_candidates")


# ── linkedin_parser tests ────────────────────────────────

def test_linkedin_parser_standard():
    data = {
        "full_name": "Priya Menon",
        "email": "priya@example.com",
        "skills": [{"name": "Python"}, {"name": "FastAPI"}],
        "experience": [{
            "title": "Backend Engineer",
            "company_name": "TechCorp",
            "duration": "2 years",
            "description": "Built payment APIs",
        }],
        "education": [{
            "school": "IIT Madras",
            "degree": "B.Tech",
            "field_of_study": "CS",
            "end_date": "2022",
        }],
        "certifications": [{"name": "AWS Developer"}],
    }
    profile = parse_linkedin_json(data)
    assert profile.name == "Priya Menon"
    assert "python" in profile.skills
    assert profile.email == "priya@example.com"
    assert len(profile.experience) == 1
    assert "AWS Developer" in profile.certifications
    print("[OK] test_linkedin_parser_standard")


# ── schema tests ─────────────────────────────────────────

def test_dimension_score_clamped():
    """Pydantic should reject score > 10 (ge=0, le=10 constraint)."""
    from pydantic import ValidationError
    try:
        ds = DimensionScore(dimension="Test", weight=0.3, score=15, justification="x")
        # If we reach here, Pydantic did NOT raise — test fails
        raise AssertionError("Pydantic should have raised ValidationError for score=15")

    except (ValidationError, ValueError):
        pass  # Expected — constraint enforced
    print("[OK] test_dimension_score_clamped")



def test_weighted_total_calculation():
    # Manual check: 5 dims each 5/10 with correct weights
    dims = [
        DimensionScore(dimension="Skills Match", weight=0.30, score=5, justification="ok"),
        DimensionScore(dimension="Experience Relevance", weight=0.25, score=5, justification="ok"),
        DimensionScore(dimension="Education & Certifications", weight=0.15, score=5, justification="ok"),
        DimensionScore(dimension="Projects / Portfolio", weight=0.20, score=5, justification="ok"),
        DimensionScore(dimension="Communication Quality", weight=0.10, score=5, justification="ok"),
    ]
    total = sum(d.score * d.weight * 10 for d in dims)
    assert abs(total - 50.0) < 0.01  # 50/100 → Maybe
    print("[OK] test_weighted_total_calculation")


# ── run all ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n[TEST] Running HR Shortlister Tests\n" + "-" * 40)
    test_skills_gap_perfect_match()
    test_skills_gap_partial_match()
    test_skills_gap_no_match()
    test_learning_resource_known_skill()
    test_anonymise_removes_email()
    test_anonymise_replaces_pronouns()
    test_fairness_report_insufficient_data()
    test_fairness_report_multiple_candidates()
    test_linkedin_parser_standard()
    test_dimension_score_clamped()
    test_weighted_total_calculation()
    print("\n[OK] All tests passed!\n")
