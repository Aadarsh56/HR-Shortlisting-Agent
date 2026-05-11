"""
LinkedIn JSON parser service.

Converts LinkedIn profile export JSON into our CandidateProfile schema.
Supports the standard LinkedIn data export format and common scraper outputs.
"""

from models.schemas import (
    CandidateProfile,
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
    ResumeSource,
)


def parse_linkedin_json(data: dict) -> CandidateProfile:
    """Convert a LinkedIn JSON export/scrape into a CandidateProfile.

    Handles multiple common LinkedIn JSON structures:
    - Official LinkedIn data export
    - Custom simplified format

    Args:
        data: Dictionary of LinkedIn profile data.

    Returns:
        Structured CandidateProfile.
    """
    # Extract basic info
    name = (
        data.get("full_name")
        or data.get("name")
        or f"{data.get('first_name', '')} {data.get('last_name', '')}".strip()
        or "Unknown"
    )
    email = data.get("email") or data.get("email_address")
    phone = data.get("phone") or data.get("phone_number")

    # Extract skills
    skills_raw = data.get("skills", [])
    if isinstance(skills_raw, list):
        if skills_raw and isinstance(skills_raw[0], dict):
            skills = [s.get("name", s.get("skill", "")) for s in skills_raw]
        else:
            skills = [str(s) for s in skills_raw]
    else:
        skills = []

    # Extract experience
    experience = []
    for exp in data.get("experience", data.get("positions", [])):
        if isinstance(exp, dict):
            experience.append(ExperienceEntry(
                company=exp.get("company", exp.get("company_name", "")),
                title=exp.get("title", exp.get("position", "")),
                duration=exp.get("duration", exp.get("date_range", "")),
                description=exp.get("description", exp.get("summary", "")),
                domain=exp.get("industry", exp.get("domain", "")),
            ))

    # Calculate total experience years (rough estimate)
    total_years = data.get("total_experience_years", 0)
    if not total_years and experience:
        total_years = len(experience) * 2  # rough approximation

    # Extract education
    education = []
    for edu in data.get("education", []):
        if isinstance(edu, dict):
            education.append(EducationEntry(
                institution=edu.get("school", edu.get("institution", "")),
                degree=edu.get("degree", edu.get("degree_name", "")),
                field=edu.get("field_of_study", edu.get("field", "")),
                year=str(edu.get("end_date", edu.get("year", ""))),
            ))

    # Extract projects
    projects = []
    for proj in data.get("projects", []):
        if isinstance(proj, dict):
            projects.append(ProjectEntry(
                name=proj.get("title", proj.get("name", "")),
                description=proj.get("description", ""),
                technologies=proj.get("technologies", []),
                url=proj.get("url", proj.get("link", "")),
            ))

    # Extract certifications
    certifications = []
    for cert in data.get("certifications", data.get("licenses", [])):
        if isinstance(cert, dict):
            certifications.append(cert.get("name", cert.get("title", "")))
        elif isinstance(cert, str):
            certifications.append(cert)

    # Build summary from headline and about
    summary = data.get("summary", data.get("about", data.get("headline", "")))

    # Build raw text for audit
    raw_text = f"Name: {name}\n"
    raw_text += f"Summary: {summary}\n"
    raw_text += f"Skills: {', '.join(skills)}\n"
    for exp in experience:
        raw_text += f"Experience: {exp.title} at {exp.company} ({exp.duration})\n"
    for edu in education:
        raw_text += f"Education: {edu.degree} in {edu.field} from {edu.institution}\n"

    return CandidateProfile(
        name=name,
        email=email,
        phone=phone,
        skills=[s.lower().strip() for s in skills if s],
        experience=experience,
        total_experience_years=total_years,
        education=education,
        projects=projects,
        certifications=certifications,
        summary=summary,
        source=ResumeSource.LINKEDIN,
        raw_text=raw_text,
    )
