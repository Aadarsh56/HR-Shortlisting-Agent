"""
Skills gap analysis service.

For each candidate, identifies matched, missing, and bonus skills
relative to the JD requirements. Generates learning resource
recommendations for missing skills.

Matching cascade:
  Tier 1 — BGE semantic embedding similarity (cosine >= 0.72)
  Tier 2 — Exact string match
  Tier 3 — Substring containment
  Tier 4 — Alias / abbreviation map
"""

import re
from models.schemas import SkillsGap, SkillWithResource

# Lazy import: only load embedding model when first needed
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from services.embedder import semantic_skill_match as _ssm
            _embedder = _ssm
        except Exception:
            _embedder = None
    return _embedder

# ──────────────────────────────────────────────
# Learning Resource Database
# ──────────────────────────────────────────────

# Curated learning resources for common tech skills
LEARNING_RESOURCES = {
    "python": {"resource": "Python Official Tutorial — docs.python.org", "time": "1 week"},
    "javascript": {"resource": "JavaScript.info — Modern JS Tutorial", "time": "1 week"},
    "typescript": {"resource": "TypeScript Handbook — typescriptlang.org", "time": "5 days"},
    "react": {"resource": "React Official Tutorial — react.dev", "time": "1 week"},
    "angular": {"resource": "Angular Tour of Heroes — angular.io", "time": "1 week"},
    "vue": {"resource": "Vue.js Guide — vuejs.org", "time": "5 days"},
    "node.js": {"resource": "Node.js Getting Started — nodejs.org", "time": "3 days"},
    "nodejs": {"resource": "Node.js Getting Started — nodejs.org", "time": "3 days"},
    "docker": {"resource": "Docker Getting Started — docs.docker.com", "time": "3 days"},
    "kubernetes": {"resource": "Kubernetes Basics — kubernetes.io/docs", "time": "2 weeks"},
    "aws": {"resource": "AWS Cloud Practitioner — aws.amazon.com/training", "time": "2 weeks"},
    "azure": {"resource": "Azure Fundamentals — learn.microsoft.com", "time": "2 weeks"},
    "gcp": {"resource": "Google Cloud Skills Boost — cloudskillsboost.google", "time": "2 weeks"},
    "sql": {"resource": "SQLBolt Interactive Tutorial — sqlbolt.com", "time": "3 days"},
    "postgresql": {"resource": "PostgreSQL Tutorial — postgresqltutorial.com", "time": "1 week"},
    "mongodb": {"resource": "MongoDB University — university.mongodb.com", "time": "1 week"},
    "redis": {"resource": "Redis University — university.redis.com", "time": "3 days"},
    "git": {"resource": "Git Handbook — guides.github.com", "time": "2 days"},
    "ci/cd": {"resource": "GitHub Actions Docs — docs.github.com/actions", "time": "3 days"},
    "terraform": {"resource": "Terraform Getting Started — developer.hashicorp.com", "time": "1 week"},
    "graphql": {"resource": "GraphQL Official Tutorial — graphql.org/learn", "time": "3 days"},
    "rest api": {"resource": "RESTful API Design — restfulapi.net", "time": "2 days"},
    "rest apis": {"resource": "RESTful API Design — restfulapi.net", "time": "2 days"},
    "fastapi": {"resource": "FastAPI Tutorial — fastapi.tiangolo.com", "time": "3 days"},
    "django": {"resource": "Django Official Tutorial — djangoproject.com", "time": "1 week"},
    "flask": {"resource": "Flask Quickstart — flask.palletsprojects.com", "time": "3 days"},
    "spring boot": {"resource": "Spring Boot Guides — spring.io/guides", "time": "1 week"},
    "java": {"resource": "Java Tutorial — dev.java/learn", "time": "2 weeks"},
    "c++": {"resource": "LearnCpp.com — Complete C++ Tutorial", "time": "3 weeks"},
    "rust": {"resource": "The Rust Book — doc.rust-lang.org/book", "time": "3 weeks"},
    "go": {"resource": "Go Tour — go.dev/tour", "time": "1 week"},
    "machine learning": {"resource": "Andrew Ng ML Course — coursera.org", "time": "8 weeks"},
    "deep learning": {"resource": "fast.ai Practical DL — course.fast.ai", "time": "6 weeks"},
    "nlp": {"resource": "Hugging Face NLP Course — huggingface.co/learn", "time": "4 weeks"},
    "data science": {"resource": "Kaggle Learn — kaggle.com/learn", "time": "4 weeks"},
    "pandas": {"resource": "Pandas Getting Started — pandas.pydata.org", "time": "3 days"},
    "spark": {"resource": "Apache Spark Quick Start — spark.apache.org", "time": "1 week"},
    "kafka": {"resource": "Kafka Quickstart — kafka.apache.org", "time": "1 week"},
    "linux": {"resource": "Linux Journey — linuxjourney.com", "time": "1 week"},
    "agile": {"resource": "Agile Manifesto & Scrum Guide — scrumguides.org", "time": "2 days"},
    "communication": {"resource": "Coursera Business Communication — coursera.org", "time": "2 weeks"},
    "leadership": {"resource": "Coursera Leadership Specialisation — coursera.org", "time": "4 weeks"},
}


def normalise_skill(skill: str) -> str:
    """Normalise a skill name for consistent comparison.

    Args:
        skill: Raw skill string.

    Returns:
        Lowercase, stripped skill name.
    """
    return skill.lower().strip().replace(".", "").replace("-", " ")


def get_learning_resource(skill: str) -> SkillWithResource:
    """Get a learning resource recommendation for a missing skill.

    Args:
        skill: The skill name to find a resource for.

    Returns:
        SkillWithResource with learning link and time estimate.
    """
    normalised = normalise_skill(skill)

    if normalised in LEARNING_RESOURCES:
        info = LEARNING_RESOURCES[normalised]
        return SkillWithResource(
            skill=skill,
            resource=info["resource"],
            estimated_time=info["time"],
        )

    # Generic fallback
    return SkillWithResource(
        skill=skill,
        resource=f"Search: '{skill} tutorial' on Google / YouTube",
        estimated_time="1-2 weeks (estimate)",
    )



# ──────────────────────────────────────────────
# Skill Alias Map — common variations
# ──────────────────────────────────────────────
SKILL_ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "node": "node.js",
    "nodejs": "node.js",
    "postgres": "postgresql",
    "pg": "postgresql",
    "k8s": "kubernetes",
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "genai": "generative ai",
    "llm": "large language models",
    "llms": "large language models",
    "nlp": "natural language processing",
    "rest": "rest api",
    "rest apis": "rest api",
    "restful": "rest api",
    "restful api": "rest api",
    "rest api design": "rest api",
    "unit test": "unit testing",
    "pytest": "unit testing",
    "jest": "unit testing",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    "github actions": "ci/cd",
    "jenkins": "ci/cd",
    "rabbitmq": "message queues",
    "kafka": "message queues",
    "message queue": "message queues",
    "aws ec2": "aws",
    "aws s3": "aws",
    "aws lambda": "aws",
    "amazon web services": "aws",
    "gcp": "google cloud",
    "google cloud platform": "google cloud",
    "version control": "git",
    "github": "git",
    "gitlab": "git",
    "react.js": "react",
    "reactjs": "react",
    "vue.js": "vue",
    "vuejs": "vue",
    "langchain": "langchain/langgraph or similar frameworks",
    "langgraph": "langchain/langgraph or similar frameworks",
}


def _resolve_alias(skill: str) -> str:
    """Resolve a skill through the alias map."""
    norm = normalise_skill(skill)
    if norm in SKILL_ALIASES:
        return SKILL_ALIASES[norm]
        
    for k, v in SKILL_ALIASES.items():
        norm = re.sub(r'\b' + re.escape(k) + r'\b', v, norm)
        
    return norm


def _skills_match(jd_norm: str, cand_norm: str) -> bool:
    """4-tier skill matching cascade.

    Tier 1: Exact match
    Tier 2: Substring containment
    Tier 3: Alias / abbreviation map
    Tier 4: BGE semantic embedding similarity
    """
    # Tier 1: Exact
    if jd_norm == cand_norm:
        return True
    # Tier 2: Substring
    if jd_norm in cand_norm or cand_norm in jd_norm:
        return True
    # Tier 3: Alias resolution
    jd_resolved = _resolve_alias(jd_norm)
    cand_resolved = _resolve_alias(cand_norm)
    if jd_resolved == cand_resolved:
        return True
    if jd_resolved in cand_resolved or cand_resolved in jd_resolved:
        return True
    return False


def _skills_match_semantic(jd_skill: str, candidate_skills: list[str]) -> bool:
    """Tier 4: semantic embedding fallback using BGE.

    Called only when tiers 1-3 fail for a required skill.
    Returns True if any candidate skill scores >= SKILL_MATCH_THRESHOLD.
    """
    fn = _get_embedder()
    if fn is None:
        return False
    try:
        matched, _sim = fn(jd_skill, candidate_skills)
        return matched
    except Exception:
        return False


def analyse_skills_gap(
    required_skills: list[str],
    preferred_skills: list[str],
    candidate_skills: list[str],
) -> SkillsGap:
    """Compute the skills gap between JD requirements and candidate profile.

    Matching cascade:
      Tier 1: BGE semantic embedding similarity (cosine >= 0.72)
      Tier 2: Exact string match
      Tier 3: Substring containment
      Tier 4: Alias / abbreviation map (pytest=unit testing, k8s=kubernetes ...)

    Tier 1 catches cases like:
      'PyTorch' vs 'deep learning frameworks'  (sim=0.74)
      'agile methodology' vs 'scrum'           (sim=0.78)
    """
    # Normalise all skills
    req_normalised  = {normalise_skill(s): s for s in required_skills}
    pref_normalised = {normalise_skill(s): s for s in preferred_skills}
    cand_normalised = {normalise_skill(s): s for s in candidate_skills}
    cand_keys = list(cand_normalised.keys())

    matched = []
    matched_req_norms = set()

    # Tier 1: Semantic embedding (First Priority)
    if candidate_skills:
        for req_norm, req_original in req_normalised.items():
            if _skills_match_semantic(req_norm, cand_keys):
                matched.append(req_original)
                matched_req_norms.add(req_norm)

    # Tiers 2-4: Keyword cascade fallback for unmatched skills
    still_missing_norms = [
        (norm, orig) for norm, orig in req_normalised.items()
        if norm not in matched_req_norms
    ]
    for req_norm, req_original in still_missing_norms:
        for cand_norm in cand_keys:
            if _skills_match(req_norm, cand_norm):
                matched.append(req_original)
                matched_req_norms.add(req_norm)
                break

    # Missing: required skills not matched by any tier
    missing = []
    for req_norm, req_original in req_normalised.items():
        if req_norm not in matched_req_norms:
            missing.append(get_learning_resource(req_original))

    # Bonus: candidate skills not matching any JD skill (all tiers)
    all_jd = list(req_normalised.keys()) + list(pref_normalised.keys())
    bonus = []
    for cand_norm, cand_original in cand_normalised.items():
        is_covered = False
        if candidate_skills:
            is_covered = _skills_match_semantic(cand_norm, list(req_normalised.keys()) + list(pref_normalised.keys()))
        if not is_covered:
            is_covered = any(_skills_match(jd_n, cand_norm) for jd_n in all_jd)
        if not is_covered:
            bonus.append(cand_original)

    total_required = len(req_normalised)
    match_pct = (len(matched) / total_required * 100) if total_required > 0 else 0

    return SkillsGap(
        matched_skills=matched,
        missing_skills=missing,
        bonus_skills=bonus,
        match_percentage=round(match_pct, 1),
    )

