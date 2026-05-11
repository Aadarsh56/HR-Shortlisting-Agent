"""
SQLite database layer for persisting evaluations, overrides, and audit logs.

Uses SQLAlchemy ORM for structured data storage.
"""

import json
import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "hr_shortlister.db")

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


# ──────────────────────────────────────────────
# ORM Models
# ──────────────────────────────────────────────

class EvaluationRecord(Base):
    """Stores a complete candidate evaluation."""

    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    candidate_name = Column(String)
    candidate_email = Column(String, nullable=True)
    source = Column(String)
    dimension_scores_json = Column(Text)  # JSON serialised
    weighted_total = Column(Float)
    recommendation = Column(String)
    skills_gap_json = Column(Text)  # JSON serialised
    confidence = Column(Float)
    bias_flags_json = Column(Text)
    rank = Column(Integer)
    created_at = Column(DateTime, default=datetime.now)


class OverrideLog(Base):
    """Stores human score overrides with audit trail."""

    __tablename__ = "overrides"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    candidate_name = Column(String)
    dimension = Column(String)
    original_score = Column(Integer)
    new_score = Column(Integer)
    reason = Column(Text)
    reviewer = Column(String)
    flagged = Column(Integer, default=0)  # SQLite boolean
    created_at = Column(DateTime, default=datetime.now)


class SessionRecord(Base):
    """Stores session metadata."""

    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, unique=True, index=True)
    jd_title = Column(String)
    jd_raw = Column(Text)
    jd_parsed_json = Column(Text)
    total_candidates = Column(Integer, default=0)
    model_used = Column(String, default="llama3")
    created_at = Column(DateTime, default=datetime.now)


# ──────────────────────────────────────────────
# Create Tables
# ──────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(engine)


# ──────────────────────────────────────────────
# CRUD Operations
# ──────────────────────────────────────────────

def get_session() -> Session:
    """Get a new database session."""
    return SessionLocal()


def save_session_record(session_id: str, jd_title: str, jd_raw: str,
                        jd_parsed: dict, total_candidates: int):
    """Save a new evaluation session."""
    db = get_session()
    try:
        record = SessionRecord(
            session_id=session_id,
            jd_title=jd_title,
            jd_raw=jd_raw,
            jd_parsed_json=json.dumps(jd_parsed),
            total_candidates=total_candidates,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()


def save_evaluation(session_id: str, evaluation: dict):
    """Save a candidate evaluation."""
    db = get_session()
    try:
        record = EvaluationRecord(
            session_id=session_id,
            candidate_name=evaluation.get("candidate", {}).get("name", "Unknown"),
            candidate_email=evaluation.get("candidate", {}).get("email"),
            source=evaluation.get("candidate", {}).get("source", "pdf"),
            dimension_scores_json=json.dumps(evaluation.get("dimension_scores", [])),
            weighted_total=evaluation.get("weighted_total", 0),
            recommendation=evaluation.get("recommendation", "No Hire"),
            skills_gap_json=json.dumps(evaluation.get("skills_gap", {})),
            confidence=evaluation.get("confidence", 0),
            bias_flags_json=json.dumps(evaluation.get("bias_flags", [])),
            rank=evaluation.get("rank", 0),
        )
        db.add(record)
        db.commit()
    finally:
        db.close()


def save_override(session_id: str, override: dict):
    """Save a human score override."""
    db = get_session()
    try:
        record = OverrideLog(
            session_id=session_id,
            candidate_name=override["candidate_name"],
            dimension=override["dimension"],
            original_score=override["original_score"],
            new_score=override["new_score"],
            reason=override["reason"],
            reviewer=override.get("reviewer", "HR Reviewer"),
            flagged=1 if override.get("flagged", False) else 0,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()


def get_overrides(session_id: str) -> list[dict]:
    """Get all overrides for a session."""
    db = get_session()
    try:
        records = db.query(OverrideLog).filter(
            OverrideLog.session_id == session_id
        ).all()
        return [
            {
                "candidate_name": r.candidate_name,
                "dimension": r.dimension,
                "original_score": r.original_score,
                "new_score": r.new_score,
                "reason": r.reason,
                "reviewer": r.reviewer,
                "flagged": bool(r.flagged),
                "timestamp": r.created_at.isoformat() if r.created_at else "",
            }
            for r in records
        ]
    finally:
        db.close()


# Note: init_db() is called explicitly from app.py on startup.
# Do NOT call it here — avoids double init and import-time side effects.
