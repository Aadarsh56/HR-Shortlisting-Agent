"""
Pydantic schemas for the HR Resume Shortlisting Agent.

These models define the data contracts between all agents in the pipeline.
Every LLM output is validated against these schemas before proceeding.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum


from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class Recommendation(str, Enum):
    HIRE = "Hire"
    MAYBE = "Maybe"
    NO_HIRE = "No Hire"


class ResumeSource(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    LINKEDIN = "linkedin"


# ──────────────────────────────────────────────
# Job Description Models
# ──────────────────────────────────────────────

class JobDescription(BaseModel):
    """Structured representation of a parsed Job Description."""

    title: str = Field(description="Job title, e.g. 'Senior Backend Engineer'")
    required_skills: list[str] = Field(
        default_factory=list,
        description="Skills explicitly required by the JD"
    )
    preferred_skills: list[str] = Field(
        default_factory=list,
        description="Nice-to-have skills mentioned in the JD"
    )
    min_experience_years: int = Field(
        default=0,
        description="Minimum years of experience required"
    )
    max_experience_years: int | None = Field(
        default=None,
        description="Maximum years of experience (if specified)"
    )
    education_requirement: str = Field(
        default="",
        description="Minimum education level, e.g. 'Bachelor's in CS'"
    )
    certifications: list[str] = Field(
        default_factory=list,
        description="Required or preferred certifications"
    )
    responsibilities: list[str] = Field(
        default_factory=list,
        description="Key job responsibilities"
    )
    domain: str = Field(
        default="",
        description="Industry or domain, e.g. 'FinTech', 'Healthcare'"
    )


# ──────────────────────────────────────────────
# Candidate Profile Models
# ──────────────────────────────────────────────

class ExperienceEntry(BaseModel):
    """A single work experience entry."""

    company: str = ""
    title: str = ""
    duration: str = ""
    description: str = ""
    domain: str = ""


class EducationEntry(BaseModel):
    """A single education entry."""

    institution: str = ""
    degree: str = ""
    field: str = ""
    year: str = ""


class ProjectEntry(BaseModel):
    """A single project entry."""

    name: str = ""
    description: str = ""
    technologies: list[str] = Field(default_factory=list)
    url: str = ""


class CandidateProfile(BaseModel):
    """Structured representation of a parsed resume or LinkedIn profile."""

    name: str = Field(description="Candidate full name")
    email: str | None = Field(default=None, description="Email address")
    phone: str | None = Field(default=None, description="Phone number")
    linkedin_url: str | None = Field(default=None, description="LinkedIn profile URL")
    github_url: str | None = Field(default=None, description="GitHub profile URL")
    skills: list[str] = Field(
        default_factory=list,
        description="Technical and soft skills"
    )
    experience: list[ExperienceEntry] = Field(default_factory=list)
    total_experience_years: int = Field(
        default=0,
        description="Total years of professional experience"
    )
    education: list[EducationEntry] = Field(default_factory=list)
    projects: list[ProjectEntry] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    summary: str = Field(default="", description="Professional summary")
    source: ResumeSource = Field(
        default=ResumeSource.PDF,
        description="Source format of this profile"
    )
    raw_text: str = Field(
        default="",
        description="Original unprocessed text (for audit trail)"
    )
    anonymised_text: str = Field(
        default="",
        description="PII-stripped text used for scoring"
    )


# ──────────────────────────────────────────────
# Scoring Models
# ──────────────────────────────────────────────

SCORING_DIMENSIONS = [
    {"name": "Skills Match", "weight": 0.30},
    {"name": "Experience Relevance", "weight": 0.25},
    {"name": "Education & Certifications", "weight": 0.15},
    {"name": "Projects / Portfolio", "weight": 0.20},
    {"name": "Communication Quality", "weight": 0.10},
]


class DimensionScore(BaseModel):
    """Score for a single evaluation dimension."""

    dimension: str = Field(description="Dimension name, e.g. 'Skills Match'")
    weight: float = Field(description="Weight as decimal, e.g. 0.30")
    score: int = Field(ge=0, le=10, description="Score from 0 (poor) to 10 (excellent)")
    justification: str = Field(description="One-line evidence-based justification")


class SkillWithResource(BaseModel):
    """A missing skill with a learning resource recommendation."""

    skill: str
    resource: str = Field(description="Recommended learning resource or URL")
    estimated_time: str = Field(description="Estimated time to learn, e.g. '3 days'")


class SkillsGap(BaseModel):
    """Skills gap analysis for a single candidate."""

    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[SkillWithResource] = Field(default_factory=list)
    bonus_skills: list[str] = Field(
        default_factory=list,
        description="Extra skills the candidate has beyond JD requirements"
    )
    match_percentage: float = Field(
        default=0.0,
        description="Percentage of required skills matched (0-100)"
    )


class CandidateEvaluation(BaseModel):
    """Complete evaluation of a single candidate."""

    candidate: CandidateProfile
    dimension_scores: list[DimensionScore] = Field(default_factory=list)
    weighted_total: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Weighted total score (0-100)"
    )
    recommendation: Recommendation = Field(default=Recommendation.NO_HIRE)
    skills_gap: SkillsGap = Field(default_factory=SkillsGap)
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Critic agent confidence (0-1)"
    )
    bias_flags: list[str] = Field(
        default_factory=list,
        description="Any bias concerns flagged by the critic"
    )
    rank: int = Field(default=0, description="Position in ranked shortlist")


# ──────────────────────────────────────────────
# Override / Human-in-the-Loop Models
# ──────────────────────────────────────────────

class OverrideRecord(BaseModel):
    """Record of a human score override."""

    candidate_name: str
    dimension: str
    original_score: int
    new_score: int
    reason: str
    reviewer: str = "HR Reviewer"
    timestamp: datetime = Field(default_factory=datetime.now)
    flagged: bool = Field(default=False, description="Candidate flagged for further review")


# ──────────────────────────────────────────────
# Report / Session Models
# ──────────────────────────────────────────────

class ShortlistReport(BaseModel):
    """Complete shortlist report for a JD evaluation session."""

    session_id: str
    jd: JobDescription
    evaluations: list[CandidateEvaluation] = Field(default_factory=list)
    overrides: list[OverrideRecord] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)
    model_used: str = "llama3"
    total_candidates: int = 0
    hire_count: int = 0
    maybe_count: int = 0
    no_hire_count: int = 0


# AgentState is implemented as a TypedDict in agents/graph.py (LangGraph requirement).
# The Pydantic class was removed — use graph.PipelineState instead.
