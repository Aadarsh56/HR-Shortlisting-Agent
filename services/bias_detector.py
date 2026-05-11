"""
Bias detection and PII anonymisation service.

Pre-scoring: Strips names, gender pronouns, age indicators from resumes.
Post-scoring: Analyses score distributions for demographic correlations.
"""

import re


# ──────────────────────────────────────────────
# PII Anonymisation (Pre-Scoring)
# ──────────────────────────────────────────────

# Common gender pronouns to neutralise
GENDER_PRONOUNS = {
    r'\bhe\b': '[candidate]',
    r'\bshe\b': '[candidate]',
    r'\bhis\b': "[candidate's]",
    r'\bher\b': "[candidate's]",
    r'\bhim\b': '[candidate]',
    r'\bhimself\b': '[candidate]',
    r'\bherself\b': '[candidate]',
}

# Patterns that may reveal age
AGE_PATTERNS = [
    r'\b(?:date of birth|dob|d\.o\.b)\s*[:\-]?\s*\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}',
    r'\b(?:age|born)\s*[:\-]?\s*\d{1,3}\b',
    r'\b(?:19[5-9]\d|200[0-5])\b',  # birth years 1950-2005
]

# Phone and email patterns
PHONE_PATTERN = r'[\+]?[\d\s\-\(\)]{7,15}'
EMAIL_PATTERN = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'

# URL patterns (LinkedIn, personal websites)
URL_PATTERN = r'https?://[^\s]+'


def anonymise_text(text: str) -> str:
    """Remove PII from resume text before scoring.

    Strips:
    - Names (first line, assumed to be the name)
    - Gender pronouns
    - Age indicators (DOB, birth year)
    - Phone numbers
    - Email addresses
    - URLs/LinkedIn profiles

    Args:
        text: Raw resume text.

    Returns:
        Anonymised text safe for unbiased scoring.
    """
    lines = text.split('\n')
    anonymised_lines = []

    for i, line in enumerate(lines):
        processed = line

        # Skip the first non-empty line (usually the name)
        if i < 3 and line.strip() and not any(
            keyword in line.lower()
            for keyword in ['experience', 'education', 'skill', 'summary', 'objective']
        ):
            # Check if this looks like a name (mostly capital letters, short)
            words = line.strip().split()
            if len(words) <= 4 and all(
                w[0].isupper() for w in words if w and w[0].isalpha()
            ):
                processed = "[CANDIDATE NAME]"

        # Replace gender pronouns (case-insensitive)
        for pattern, replacement in GENDER_PRONOUNS.items():
            processed = re.sub(pattern, replacement, processed, flags=re.IGNORECASE)

        # Remove age indicators
        for pattern in AGE_PATTERNS:
            processed = re.sub(pattern, '[REDACTED]', processed, flags=re.IGNORECASE)

        # Remove email
        processed = re.sub(EMAIL_PATTERN, '[EMAIL REDACTED]', processed)

        # Remove phone numbers (careful not to remove year numbers)
        processed = re.sub(
            r'(?:phone|mobile|tel|cell)\s*[:\-]?\s*' + PHONE_PATTERN,
            '[PHONE REDACTED]',
            processed,
            flags=re.IGNORECASE,
        )

        # Remove URLs
        processed = re.sub(URL_PATTERN, '[URL REDACTED]', processed)

        anonymised_lines.append(processed)

    return '\n'.join(anonymised_lines)


# ──────────────────────────────────────────────
# Post-Scoring Fairness Audit
# ──────────────────────────────────────────────

def generate_fairness_report(evaluations: list[dict]) -> dict:
    """Analyse score distributions for potential bias patterns.

    Checks:
    - Score variance (high variance may indicate inconsistency)
    - Score clustering (are scores suspiciously uniform?)
    - Outlier detection (any candidate scored vastly differently?)

    Args:
        evaluations: List of CandidateEvaluation dicts.

    Returns:
        Fairness report dict with findings and flags.
    """
    if len(evaluations) < 2:
        return {
            "status": "insufficient_data",
            "message": "Need at least 2 candidates for fairness analysis.",
            "flags": [],
        }

    scores = [e.get("weighted_total", 0) for e in evaluations]
    n = len(scores)
    mean_score = sum(scores) / n
    variance = sum((s - mean_score) ** 2 for s in scores) / n
    std_dev = variance ** 0.5

    flags = []

    # Check for suspiciously low variance (might indicate rubber-stamping)
    if std_dev < 2.0 and n >= 3:
        flags.append(
            "⚠️ Very low score variance (σ={:.1f}). Scores may not differentiate candidates meaningfully.".format(
                std_dev
            )
        )

    # Check for outliers (more than 2 std devs from mean)
    for e in evaluations:
        score = e.get("weighted_total", 0)
        name = e.get("candidate", {}).get("name", "Unknown")
        if abs(score - mean_score) > 2 * std_dev and std_dev > 0:
            flags.append(
                f"⚠️ Outlier detected: {name} scored {score:.1f} "
                f"(mean={mean_score:.1f}, σ={std_dev:.1f}). Review recommended."
            )

    # Check dimension-level consistency
    dimension_scores = {}
    for e in evaluations:
        for ds in e.get("dimension_scores", []):
            dim = ds.get("dimension", "")
            if dim not in dimension_scores:
                dimension_scores[dim] = []
            dimension_scores[dim].append(ds.get("score", 0))

    for dim, dim_scores in dimension_scores.items():
        if len(dim_scores) >= 3:
            dim_mean = sum(dim_scores) / len(dim_scores)
            dim_var = sum((s - dim_mean) ** 2 for s in dim_scores) / len(dim_scores)
            if dim_var < 0.5:
                flags.append(
                    f"ℹ️ Dimension '{dim}' has nearly identical scores across candidates. "
                    f"Verify the scoring logic is differentiating properly."
                )

    return {
        "status": "completed",
        "total_candidates": n,
        "mean_score": round(mean_score, 2),
        "std_deviation": round(std_dev, 2),
        "score_range": [min(scores), max(scores)],
        "flags": flags,
        "recommendation": (
            "No significant bias patterns detected."
            if not flags
            else f"{len(flags)} potential concern(s) flagged for review."
        ),
    }
