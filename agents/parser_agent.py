"""
Parser Agent — LangGraph Node

Handles two tasks:
1. Parse the Job Description into structured JobDescription
2. Parse each resume into structured CandidateProfile

Uses Llama 3 via Ollama for structured extraction.
"""

import json
import os
import re

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from models.schemas import (
    CandidateProfile,
    EducationEntry,
    ExperienceEntry,
    JobDescription,
    ProjectEntry,
    ResumeSource,
)
from services.bias_detector import anonymise_text

# Load prompts
PROMPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")


def _extract_linkedin_url(text: str) -> str | None:
    """Regex fallback: extract LinkedIn URL from raw resume text."""
    patterns = [
        r'https?://(?:www\.)?linkedin\.com/in/[\w\-]+/?',
        r'linkedin\.com/in/[\w\-]+/?',
        r'(?i)linkedin[:\s]+(?:https?://)?(?:www\.)?linkedin\.com/in/([\w\-]+)',
        r'(?i)linkedin[:\s]+([\w\-]+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            url = m.group(0)
            if not url.startswith('http'):
                url = 'https://' + url
            # Validate it looks like a real profile URL
            if 'linkedin.com/in/' in url:
                return url.rstrip('/')
    return None


def _extract_github_url(text: str) -> str | None:
    """Regex fallback: extract GitHub URL from raw resume text."""
    patterns = [
        r'https?://(?:www\.)?github\.com/[\w\-]+/?',
        r'github\.com/[\w\-]+/?',
        r'(?i)github[:\s]+(?:https?://)?(?:www\.)?github\.com/([\w\-]+)',
        r'(?i)github[:\s]+([\w\-]+)',
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            url = m.group(0).strip()
            if not url.startswith('http'):
                if 'github.com/' in url:
                    url = 'https://' + url
                else:
                    # Just a username — build URL
                    username = re.search(r'[\w\-]+$', url)
                    if username:
                        url = f'https://github.com/{username.group(0)}'
            if 'github.com/' in url:
                return url.rstrip('/')
    return None




def _load_prompt(filename: str) -> str:
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _sanitise_input(text: str) -> str:
    """Basic input sanitisation to prevent prompt injection."""
    dangerous_patterns = [
        "ignore previous instructions",
        "ignore all previous",
        "disregard above",
        "<|im_start|>",
        "<|im_end|>",
        "<<SYS>>",
        "[INST]",
        "<system>",
    ]
    cleaned = text
    for pattern in dangerous_patterns:
        cleaned = cleaned.replace(pattern, "[REDACTED]")
        cleaned = cleaned.replace(pattern.upper(), "[REDACTED]")
    return cleaned


def _extract_json(text: str) -> dict:
    """Extract JSON from LLM response text, handling markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding JSON object by finding first { and last }
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(text[start_idx:end_idx+1])
        except json.JSONDecodeError:
            pass

    return {}


def _to_str(val) -> str:
    """Safely convert LLM output to string, handling nulls and lists."""
    if val is None:
        return ""
    if isinstance(val, list):
        return "; ".join(str(v) for v in val)
    return str(val).strip()

def _get_llm():
    """Get the Ollama LLM instance."""
    from . import get_langfuse_callbacks
    
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3")
    return ChatOllama(
        model=model,
        base_url=base_url,
        temperature=0,
        format="json",
        num_predict=2048,  # Increased to prevent JSON truncation
        callbacks=get_langfuse_callbacks(),
        num_ctx=4096,      # Reduced from 8192 for faster prompt evaluation
    )


# ── Regex-based fallback extractors (used when LLM returns blanks) ─────────────

_EMAIL_RE = re.compile(r'[\w.+\-]+@[\w\-]+\.[a-z]{2,}', re.IGNORECASE)
_PHONE_RE = re.compile(
    r'(?:\+?\d{1,3}[\s\-]?)?'
    r'(?:\(?\d{2,4}\)?[\s\-]?)'
    r'\d{3,4}[\s\-]?\d{3,5}'
)
# Heuristic: first 1-3 non-empty lines from the top that look like a human name
_NAME_BLACKLIST = re.compile(
    r'resume|curriculum|vitae|cv|profile|summary|objective|contact|address|'
    r'linkedin|github|www|http|@|\d{5}|page',
    re.IGNORECASE,
)


def _extract_email(text: str) -> str | None:
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None


def _extract_phone(text: str) -> str | None:
    m = _PHONE_RE.search(text)
    return m.group(0).strip() if m else None


def _extract_name_heuristic(text: str) -> str | None:
    """Try to find the candidate name from the first few meaningful lines."""
    for line in text.splitlines()[:25]:
        line = line.strip()
        # Must be 2-5 words, no digits, no blacklisted terms, reasonable length
        words = line.split()
        if 2 <= len(words) <= 5 and not _NAME_BLACKLIST.search(line):
            if not re.search(r'[\d@#$%^&*(){}\[\]|<>/]', line):
                if all(len(w) >= 2 for w in words):
                    return line
    return None


# Common tech skill keywords for regex-based skills extraction fallback
_SKILL_KEYWORDS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi", "spring",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "linux",
    "sql", "postgresql", "mysql", "mongodb", "redis", "kafka", "elasticsearch",
    "git", "ci/cd", "machine learning", "deep learning", "nlp", "llm",
    "langchain", "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy",
    "spark", "hadoop", "rest api", "graphql", "agile", "scrum", "html", "css",
    "excel", "power bi", "tableau", "matlab", "r",
]
_SKILL_RE = re.compile(
    r'\b(' + '|'.join(re.escape(s) for s in _SKILL_KEYWORDS) + r')\b',
    re.IGNORECASE,
)


def _extract_skills_from_text(text: str) -> list[str]:
    """Regex fallback: extract known skill keywords from raw resume text."""
    found = dict.fromkeys(
        m.group(0).lower() for m in _SKILL_RE.finditer(text)
    )
    return list(found.keys())


def validate_model_exists() -> tuple[bool, str]:
    """Check whether the configured Ollama model is available locally.

    Returns:
        (True, "") if the model is available.
        (False, error_message) if the model is missing or Ollama is unreachable.
    """
    import urllib.request
    import urllib.error

    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_MODEL", "llama3")

    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        available = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
        # Also include full names with tags
        available_full = [m.get("name", "") for m in data.get("models", [])]
        model_base = model.split(":")[0]

        if model in available_full or model_base in available:
            return True, ""

        pull_cmd = f"ollama pull {model}"
        return False, (
            f"Model '{model}' is not downloaded.\n\n"
            f"Run this command in a terminal to download it:\n\n"
            f"    {pull_cmd}\n\n"
            f"Available models: {', '.join(available_full) if available_full else 'none'}"
        )
    except urllib.error.URLError:
        return False, (
            f"Cannot connect to Ollama at {base_url}.\n\n"
            "Make sure Ollama is running. Open a terminal and run:\n\n"
            "    ollama serve"
        )
    except Exception as e:
        return False, f"Model check failed: {e}"


def parse_jd(jd_text: str) -> JobDescription:
    """Parse a Job Description into structured format using Llama 3.

    Args:
        jd_text: Raw job description text.

    Returns:
        Parsed JobDescription object.
    """
    llm = _get_llm()
    system_prompt = _load_prompt("jd_parser.txt")
    sanitised = _sanitise_input(jd_text)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Parse this Job Description:\n\n{sanitised}"),
    ]

    # Try up to 2 times
    for attempt in range(2):
        try:
            response = llm.invoke(messages)
            data = _extract_json(response.content)

            if data:
                # Coerce None values — LLM sometimes returns null for optional fields
                raw_min = data.get("min_experience_years")
                raw_max = data.get("max_experience_years")
                return JobDescription(
                    title=data.get("title") or "Untitled Position",
                    required_skills=[
                        s.lower().strip()
                        for s in (data.get("required_skills") or [])
                        if isinstance(s, str)
                    ],
                    preferred_skills=[
                        s.lower().strip()
                        for s in (data.get("preferred_skills") or [])
                        if isinstance(s, str)
                    ],
                    min_experience_years=int(raw_min) if raw_min is not None else 0,
                    max_experience_years=int(raw_max) if raw_max is not None else None,
                    education_requirement=data.get("education_requirement") or "",
                    certifications=[c for c in (data.get("certifications") or []) if c],
                    responsibilities=[r for r in (data.get("responsibilities") or []) if r],
                    domain=data.get("domain") or "",
                )
        except Exception as e:
            if attempt == 1:
                return JobDescription(title="Parse Failed")
            continue

    return JobDescription(title="Parse Failed")


def parse_resume(resume_text: str, source: str = "pdf") -> CandidateProfile:
    """Parse a resume into structured format using Llama 3.

    Args:
        resume_text: Raw resume text content.
        source: Source format (pdf, docx, linkedin).

    Returns:
        Parsed CandidateProfile object.
    """
    llm = _get_llm()
    system_prompt = _load_prompt("resume_parser.txt")
    sanitised = _sanitise_input(resume_text)

    # Truncate very long resumes
    if len(sanitised) > 50000:
        sanitised = sanitised[:50000] + "\n[TRUNCATED]"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Parse this resume:\n\n{sanitised}"),
    ]

    for attempt in range(3):
        try:
            response = llm.invoke(messages)
            data = _extract_json(response.content)

            if data:
                # Build experience entries
                experience = []
                for exp in data.get("experience", []):
                    if isinstance(exp, dict):
                        experience.append(ExperienceEntry(
                            company=_to_str(exp.get("company")),
                            title=_to_str(exp.get("title")),
                            duration=_to_str(exp.get("duration")),
                            description=_to_str(exp.get("description")),
                            domain=_to_str(exp.get("domain")),
                        ))

                # Build education entries
                education = []
                for edu in data.get("education", []):
                    if isinstance(edu, dict):
                        education.append(EducationEntry(
                            institution=_to_str(edu.get("institution")),
                            degree=_to_str(edu.get("degree")),
                            field=_to_str(edu.get("field")),
                            year=_to_str(edu.get("year")),
                        ))

                # Build project entries
                projects = []
                for proj in data.get("projects", []):
                    if isinstance(proj, dict):
                        techs = proj.get("technologies")
                        if not isinstance(techs, list):
                            techs = [str(techs)] if techs else []
                        projects.append(ProjectEntry(
                            name=_to_str(proj.get("name")),
                            description=_to_str(proj.get("description")),
                            technologies=[str(t) for t in techs],
                            url=_to_str(proj.get("url")),
                        ))

                # ── Regex fallbacks for any fields the LLM left blank ────────
                llm_name = data.get("name")
                llm_email = data.get("email")
                llm_phone = data.get("phone")
                llm_skills = [
                    s.lower().strip()
                    for s in (data.get("skills") or [])
                    if isinstance(s, str)
                ]

                final_name  = llm_name  or _extract_name_heuristic(resume_text) or "Unknown Candidate"
                final_email = llm_email or _extract_email(resume_text)
                final_phone = llm_phone or _extract_phone(resume_text)
                # If LLM extracted 0 skills, fall back to keyword scan
                final_skills = llm_skills if llm_skills else _extract_skills_from_text(resume_text)

                # Anonymise for bias-free scoring
                anonymised = anonymise_text(resume_text)

                # LinkedIn/GitHub — use LLM output first, regex fallback second
                linkedin = data.get("linkedin_url") or _extract_linkedin_url(resume_text)
                github = data.get("github_url") or _extract_github_url(resume_text)

                raw_exp = data.get("total_experience_years")
                
                final_summary = _to_str(data.get("summary"))
                
                return CandidateProfile(
                    name=_to_str(final_name),
                    email=_to_str(final_email) or None,
                    phone=_to_str(final_phone) or None,
                    linkedin_url=_to_str(linkedin) or None,
                    github_url=_to_str(github) or None,
                    skills=final_skills,
                    experience=experience,
                    total_experience_years=int(raw_exp) if raw_exp is not None else 0,
                    education=education,
                    projects=projects,
                    certifications=[_to_str(c) for c in (data.get("certifications") or []) if c],
                    summary=final_summary,
                    source=ResumeSource(source) if source in ["pdf", "docx", "linkedin"] else ResumeSource.PDF,
                    raw_text=resume_text,
                    anonymised_text=anonymised,
                )
        except Exception as e:
            if attempt == 2:
                return CandidateProfile(
                    name="Parse Failed",
                    raw_text=resume_text,
                    anonymised_text=anonymise_text(resume_text),
                    source=ResumeSource(source) if source in ["pdf", "docx", "linkedin"] else ResumeSource.PDF,
                )
            continue

    return CandidateProfile(name="Parse Failed", raw_text=resume_text)


def parser_node(state: dict) -> dict:
    """LangGraph node: Parse JD and all resumes.

    Args:
        state: Current agent state dict.

    Returns:
        Updated state with jd_parsed and candidates.
    """
    errors = list(state.get("errors", []))

    # 1. Parse JD
    jd_parsed = None
    jd_raw = state.get("jd_raw", "")
    if jd_raw:
        try:
            jd_parsed = parse_jd(jd_raw)
        except Exception as e:
            errors.append(f"JD parsing error: {str(e)}")

    # 2. Parse all resumes
    candidates = []
    for resume_info in state.get("resumes_raw", []):
        filename = resume_info.get("filename", "unknown")
        content = resume_info.get("content", "")
        source = resume_info.get("source", "pdf")

        if not content or content.startswith("[ERROR]"):
            errors.append(f"Skipped {filename}: {content}")
            continue

        try:
            profile = parse_resume(content, source)
            if profile.name == "Parse Failed":
                errors.append(f"Failed to parse: {filename}")
                # Don't add failed parses to candidates list
                continue
            candidates.append(profile)
        except Exception as e:
            errors.append(f"Error parsing {filename}: {str(e)}")

    return {
        **state,
        "jd_parsed": jd_parsed.model_dump() if jd_parsed else None,
        "candidates": [c.model_dump() for c in candidates],
        "errors": errors,
    }
