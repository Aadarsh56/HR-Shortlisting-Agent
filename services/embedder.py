"""
Semantic Embedding Service — BGE (BAAI/bge-small-en-v1.5)

Uses SentenceTransformers with the BGE-small model to provide semantic
similarity matching. BGE-small is:
  - 130MB download — fits in RAM without using any VRAM
  - #1 ranked English retrieval model on the MTEB benchmark
  - Fully local: no API keys, no data leaves the machine

Two primary uses in this pipeline:
  1. Pre-filtering: rank resumes by semantic similarity to JD before LLM
  2. Semantic skills matching: fallback when keyword/alias matching fails
     e.g. "PyTorch" ↔ "deep learning frameworks" (cosine sim ≈ 0.74)
"""

from __future__ import annotations

import os
import numpy as np

# ── Singleton: load model once per process ─────────────────────────────────
_model = None
_model_name = "BAAI/bge-small-en-v1.5"


def _get_model():
    """Load BGE model once and cache it for the process lifetime."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer(_model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
    return _model


def is_available() -> bool:
    """Check if the embedding model can be loaded."""
    try:
        _get_model()
        return True
    except Exception:
        return False


# ── Core functions ─────────────────────────────────────────────────────────

def embed(texts: list[str]) -> np.ndarray:
    """Embed a list of strings into a 2D numpy array of shape (N, 384).

    BGE requires a query prefix for retrieval tasks:
      - Documents (resumes/JDs): no prefix needed
      - Short phrases (skills): no prefix needed

    Args:
        texts: List of strings to embed.

    Returns:
        np.ndarray of shape (len(texts), 384), L2-normalised.
    """
    model = _get_model()
    # BGE best practice: normalise embeddings for cosine similarity
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalised vectors.

    Since vectors are already normalised, this reduces to a dot product.
    Returns a float in the range [-1, 1] where 1 = identical.
    """
    return float(np.dot(a, b))


def similarity_matrix(vecs_a: np.ndarray, vecs_b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between two sets of vectors.

    Args:
        vecs_a: Shape (M, D)
        vecs_b: Shape (N, D)

    Returns:
        Shape (M, N) — similarity_matrix[i, j] = similarity(a[i], b[j])
    """
    return np.dot(vecs_a, vecs_b.T)


# ── Resume pre-filtering ────────────────────────────────────────────────────

def rank_resumes_by_similarity(
    jd_text: str,
    resumes: list[dict],
    top_k: int | None = None,
) -> list[dict]:
    """Rank resumes by semantic similarity to the job description.

    Uses BGE embeddings to compute cosine similarity between the JD and
    each resume. Returns all resumes sorted by similarity (highest first),
    with a 'semantic_similarity' float added to each dict.

    If top_k is set, returns only the top_k most similar resumes.
    This is the pre-filtering step that prevents unnecessary LLM calls.

    Args:
        jd_text:  Full job description text.
        resumes:  List of resume dicts with a 'content' key.
        top_k:    If set, only return the top_k most similar resumes.

    Returns:
        Sorted list of resume dicts with 'semantic_similarity' score added.
    """
    if not resumes:
        return []

    # Embed JD and all resumes in a single batched call (fast)
    texts = [jd_text] + [r.get("content", "") for r in resumes]
    vectors = embed(texts)

    jd_vec = vectors[0]
    resume_vecs = vectors[1:]

    # Compute similarity for each resume
    enriched = []
    for i, resume in enumerate(resumes):
        sim = cosine_similarity(jd_vec, resume_vecs[i])
        enriched.append({**resume, "semantic_similarity": round(float(sim), 4)})

    # Sort descending by similarity
    enriched.sort(key=lambda r: r["semantic_similarity"], reverse=True)

    if top_k is not None:
        return enriched[:top_k]
    return enriched


# ── Semantic skill matching ─────────────────────────────────────────────────

# Cache skill embeddings to avoid re-computing the same skills repeatedly
_skill_cache: dict[str, np.ndarray] = {}
SKILL_MATCH_THRESHOLD = float(os.getenv("SKILL_SEMANTIC_THRESHOLD", "0.72"))


def semantic_skill_match(jd_skill: str, candidate_skills: list[str]) -> tuple[bool, float]:
    """Check if any candidate skill semantically matches a JD skill.

    This is the 4th-tier fallback after:
      Tier 1: Exact match
      Tier 2: Substring match
      Tier 3: Alias map
      Tier 4: THIS — semantic similarity via BGE embeddings

    Example matches that tiers 1-3 miss:
      "PyTorch" ↔ "deep learning frameworks"     (sim ≈ 0.74)
      "LLM fine-tuning" ↔ "model training"        (sim ≈ 0.81)
      "data wrangling" ↔ "ETL pipelines"          (sim ≈ 0.69)
      "agile methodology" ↔ "scrum"               (sim ≈ 0.78)

    Args:
        jd_skill:         A single required/preferred skill from the JD.
        candidate_skills: All skills extracted from the candidate's resume.

    Returns:
        (matched: bool, best_similarity: float)
    """
    if not candidate_skills:
        return False, 0.0

    # Gather all unique skills to embed (use cache for efficiency)
    to_embed = []
    for skill in [jd_skill] + candidate_skills:
        if skill not in _skill_cache:
            to_embed.append(skill)

    if to_embed:
        vecs = embed(to_embed)
        for skill, vec in zip(to_embed, vecs):
            _skill_cache[skill] = vec

    jd_vec = _skill_cache[jd_skill]
    cand_vecs = np.array([_skill_cache[s] for s in candidate_skills])

    # Find the highest similarity between jd_skill and any candidate skill
    similarities = np.dot(cand_vecs, jd_vec)
    best_sim = float(np.max(similarities))

    return best_sim >= SKILL_MATCH_THRESHOLD, round(best_sim, 3)


def compute_jd_resume_similarity(jd_text: str, resume_text: str) -> float:
    """Compute semantic similarity between a JD and a single resume.

    Returns a float in [0, 1] representing how semantically aligned
    the resume is with the job description.

    Args:
        jd_text:     Full job description text.
        resume_text: Full resume text (raw or anonymised).

    Returns:
        Cosine similarity score in [0, 1].
    """
    vecs = embed([jd_text, resume_text])
    return round(float(cosine_similarity(vecs[0], vecs[1])), 4)
