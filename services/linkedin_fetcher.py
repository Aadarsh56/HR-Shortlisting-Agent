"""
LinkedIn Profile Fetcher — using linkedin-api (Free, unofficial API)

Uses the linkedin-api Python library which authenticates with your own
LinkedIn credentials and fetches any public profile via LinkedIn's
internal mobile API (same API the LinkedIn app uses).

Setup (one-time):
  1. Add to .env:
       LINKEDIN_EMAIL=your-linkedin-email@gmail.com
       LINKEDIN_PASSWORD=your-linkedin-password

  2. The credentials are only used to authenticate with LinkedIn.
     They are never stored beyond the session.

IMPORTANT NOTES:
  - This uses LinkedIn's unofficial internal API
  - Only fetch profiles of candidates who have consented (e.g. applied for a role)
  - Do not use for mass scraping — LinkedIn may temporarily lock the account
  - Use a secondary LinkedIn account if concerned about rate limits

Supported profile fields:
  - Name, headline, summary
  - Skills (with endorsement counts)
  - Experience (company, title, dates, description)
  - Education (school, degree, field, dates)
  - Certifications, languages, projects
  - Contact info (email if shared publicly)
"""

from __future__ import annotations

import os
import re
from datetime import datetime

from models.schemas import (
    CandidateProfile,
    EducationEntry,
    ExperienceEntry,
    ProjectEntry,
    ResumeSource,
)


# ── Singleton client ────────────────────────────────────────────────────────
_client = None


def _get_client():
    """Authenticate once and reuse the client for the session."""
    global _client
    if _client is not None:
        return _client, ""

    email    = os.getenv("LINKEDIN_EMAIL", "").strip()
    password = os.getenv("LINKEDIN_PASSWORD", "").strip()

    if not email or not password:
        return None, (
            "LINKEDIN_EMAIL and LINKEDIN_PASSWORD are not set in .env\n\n"
            "Add these lines to your .env file:\n"
            "  LINKEDIN_EMAIL=your-email@gmail.com\n"
            "  LINKEDIN_PASSWORD=your-password"
        )

    try:
        from linkedin_api import Linkedin
        _client = Linkedin(email, password)
        return _client, ""
    except Exception as e:
        err = str(e)
        if "CHALLENGE" in err.upper() or "checkpoint" in err.lower():
            return None, (
                "LinkedIn security challenge detected.\n"
                "1. Open linkedin.com in a browser and solve the CAPTCHA\n"
                "2. Then try again here"
            )
        if "Wrong email or password" in err or "401" in err:
            return None, "Wrong LinkedIn email or password. Check your .env file."
        return None, f"LinkedIn login error: {err}"


def is_available() -> bool:
    """Check if credentials are configured."""
    return bool(
        os.getenv("LINKEDIN_EMAIL", "").strip() and
        os.getenv("LINKEDIN_PASSWORD", "").strip()
    )


# ── URL helpers ─────────────────────────────────────────────────────────────

_LI_PATTERN = re.compile(
    r"(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9\-_%]+)/?",
    re.IGNORECASE,
)


def extract_username(url_or_username: str) -> str | None:
    """Extract LinkedIn username from a URL or return as-is if already username."""
    url_or_username = url_or_username.strip()
    m = _LI_PATTERN.search(url_or_username)
    if m:
        return m.group(1).rstrip("/")
    # If no URL pattern, assume it's already a username
    if re.match(r"^[a-zA-Z0-9\-_%]+$", url_or_username):
        return url_or_username
    return None


# ── Date helpers ────────────────────────────────────────────────────────────

def _li_date_to_str(date_dict: dict | None) -> str:
    if not date_dict:
        return ""
    year  = date_dict.get("year", "")
    month = date_dict.get("month", "")
    return f"{year}-{month:02d}" if (year and month) else str(year)


def _calc_years(positions: list[dict]) -> int:
    total_months = 0
    now = datetime.now()
    for pos in positions:
        td = pos.get("timePeriod", {})
        start = td.get("startDate", {})
        end   = td.get("endDate", {})
        sy = start.get("year",  now.year)
        sm = start.get("month", 1)
        ey = end.get("year",  now.year) if end else now.year
        em = end.get("month", now.month) if end else now.month
        total_months += max(0, (ey - sy) * 12 + (em - sm))
    return max(0, round(total_months / 12))


# ── Main fetch function ─────────────────────────────────────────────────────

def fetch_linkedin_profile(url_or_username: str) -> tuple[CandidateProfile | None, str]:
    """Fetch a LinkedIn profile and return a structured CandidateProfile.

    Args:
        url_or_username: LinkedIn profile URL or username
                         e.g. "https://linkedin.com/in/johndoe/" or "johndoe"

    Returns:
        (CandidateProfile, "")  on success
        (None, error_message)   on failure
    """
    username = extract_username(url_or_username)
    if not username:
        return None, f"Cannot extract LinkedIn username from: '{url_or_username}'"

    client, err = _get_client()
    if err:
        return None, err

    try:
        data = client.get_profile(username)
    except Exception as e:
        err_msg = str(e)
        if "RESTRICTED" in err_msg.upper() or "403" in err_msg:
            return None, f"Profile '{username}' is private or restricted."
        if "NOT_FOUND" in err_msg.upper() or "404" in err_msg:
            return None, f"Profile '{username}' not found. Check the username."
        return None, f"Error fetching '{username}': {err_msg}"

    if not data:
        return None, f"No data returned for '{username}'. Profile may be private."

    return _map_to_profile(data, username)


def _map_to_profile(data: dict, username: str) -> tuple[CandidateProfile, str]:
    """Map linkedin-api response to CandidateProfile."""

    # ── Basic info ──
    first = data.get("firstName", "")
    last  = data.get("lastName",  "")
    name  = f"{first} {last}".strip() or username

    headline = data.get("headline", "")
    summary  = data.get("summary", "")

    linkedin_url = f"https://www.linkedin.com/in/{username}"

    # ── Contact info ──
    contact = data.get("contact_info", {})
    email   = contact.get("email_address")
    phone   = None
    phones  = contact.get("phone_numbers", [])
    if phones:
        phone = phones[0].get("number") if isinstance(phones[0], dict) else str(phones[0])

    # Extract GitHub from websites
    github_url = None
    websites = contact.get("websites", [])
    for site in websites:
        url = site.get("url", "") if isinstance(site, dict) else str(site)
        if "github.com" in url.lower():
            github_url = url
            break

    # ── Skills ──
    skills_raw = data.get("skills", [])
    skills = []
    for s in skills_raw:
        if isinstance(s, dict):
            name_val = s.get("name", "")
        else:
            name_val = str(s)
        if name_val:
            skills.append(name_val.lower().strip())

    # ── Experience ──
    experiences: list[ExperienceEntry] = []
    positions = data.get("experience", [])
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        td    = pos.get("timePeriod", {})
        start = _li_date_to_str(td.get("startDate"))
        end   = _li_date_to_str(td.get("endDate")) if td.get("endDate") else "Present"
        experiences.append(ExperienceEntry(
            company     = pos.get("companyName", ""),
            title       = pos.get("title", ""),
            duration    = f"{start} — {end}" if start else end,
            description = pos.get("description", ""),
            domain      = pos.get("industries", [""])[0] if pos.get("industries") else "",
        ))

    total_years = _calc_years(
        [{"timePeriod": p.get("timePeriod", {})} for p in positions]
    )

    # ── Education ──
    education: list[EducationEntry] = []
    for edu in data.get("education", []):
        if not isinstance(edu, dict):
            continue
        td  = edu.get("timePeriod", {})
        end_date = td.get("endDate", {})
        education.append(EducationEntry(
            institution = edu.get("schoolName", ""),
            degree      = edu.get("degreeName", ""),
            field       = edu.get("fieldOfStudy", ""),
            year        = str(end_date.get("year", "")) if end_date else "",
        ))

    # ── Certifications ──
    certifications = [
        c.get("name", "") for c in data.get("certifications", [])
        if isinstance(c, dict) and c.get("name")
    ]

    # ── Projects ──
    projects: list[ProjectEntry] = []
    for proj in data.get("projects", []):
        if not isinstance(proj, dict):
            continue
        projects.append(ProjectEntry(
            name        = proj.get("title", ""),
            description = proj.get("description", ""),
            technologies= [],
            url         = proj.get("url", ""),
        ))

    # ── Build raw_text ──
    raw_lines = [
        f"Name: {name}",
        f"Headline: {headline}",
        f"Summary: {summary}",
        f"Skills: {', '.join(skills)}",
    ]
    for exp in experiences:
        raw_lines.append(f"Experience: {exp.title} at {exp.company} ({exp.duration})")
        if exp.description:
            raw_lines.append(f"  {exp.description[:200]}")
    for edu in education:
        raw_lines.append(
            f"Education: {edu.degree} in {edu.field} from {edu.institution} ({edu.year})"
        )

    profile = CandidateProfile(
        name                   = name,
        email                  = email,
        phone                  = phone,
        linkedin_url           = linkedin_url,
        github_url             = github_url,
        skills                 = skills,
        experience             = experiences,
        total_experience_years = total_years,
        education              = education,
        projects               = projects,
        certifications         = certifications,
        summary                = summary or headline,
        source                 = ResumeSource.LINKEDIN,
        raw_text               = "\n".join(raw_lines),
    )
    return profile, ""
