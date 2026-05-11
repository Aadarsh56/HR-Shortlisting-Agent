"""
HR Resume Shortlisting Agent — Streamlit UI
Professional candidate evaluation dashboard powered by local AI.
"""

import json
import os
import uuid
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from services.file_parser import detect_and_parse_bytes
from services.linkedin_parser import parse_linkedin_json
from agents.graph import run_pipeline
from agents.parser_agent import validate_model_exists, parse_resume, _extract_linkedin_url, _extract_github_url
from models.database import save_override, get_overrides, init_db
from services.report_generator import generate_pdf_report, generate_html_report, generate_json_report
from services.bias_detector import generate_fairness_report

init_db()

# ── Security: Optional Password Gate ──────────────────────────────────────
# Set APP_PASSWORD in .streamlit/secrets.toml to enable authentication.
# If not set, the app runs in open mode (suitable for localhost-only use).
try:
    _app_password = st.secrets.get("APP_PASSWORD", "")
except Exception:
    _app_password = ""   # secrets.toml not present — open mode
if _app_password:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if not st.session_state.authenticated:
        st.set_page_config(page_title="HR Shortlisting Agent — Login", layout="centered")
        st.markdown("## HR Shortlisting Agent")
        st.markdown("Enter the access password to continue.")
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if pwd == _app_password:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()

# ── Security: Rate Limiting ────────────────────────────────────────────────
# Max 10 pipeline runs per session to prevent runaway GPU usage.
MAX_RUNS_PER_SESSION = 10
if "run_count" not in st.session_state:
    st.session_state.run_count = 0

st.set_page_config(
    page_title="HR Shortlisting Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Base Colors & Layout */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    section[data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid #1e293b !important; }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Page header */
    .page-header {
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid #1e293b;
        margin-bottom: 2rem;
    }
    .page-header h1 {
        font-size: 2rem; font-weight: 700; color: #f8fafc; margin: 0 0 0.5rem 0;
        display: flex; align-items: center; gap: 0.75rem;
    }
    .page-header p { font-size: 0.95rem; color: #94a3b8; margin: 0; font-weight: 400; }
    
    /* Top pills */
    .header-pill {
        display: inline-flex; align-items: center; gap: 0.4rem;
        padding: 0.35rem 0.85rem; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;
        background: rgba(16, 185, 129, 0.1); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.2);
        margin-right: 0.5rem; margin-top: 1rem;
    }

    /* Metric cards */
    .metric-row { display: flex; gap: 1rem; margin-bottom: 2rem; }
    .metric-card {
        flex: 1; background: #111827; border: 1px solid #1e293b; border-radius: 12px;
        padding: 1.5rem; text-align: center; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-value { font-size: 2.5rem; font-weight: 700; line-height: 1; margin-bottom: 0.5rem; }
    .metric-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; }
    .total  .metric-value { color: #3b82f6; }
    .hire   .metric-value { color: #10b981; }
    .maybe  .metric-value { color: #f59e0b; }
    .nohire .metric-value { color: #ef4444; }

    /* Section headers */
    .section-title {
        font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.1em;
        color: #94a3b8; margin: 2rem 0 1rem 0; padding-bottom: 0.5rem; border-bottom: 1px solid #1e293b;
    }

    /* Skill tags */
    .skill-tag { display: inline-block; padding: 4px 10px; border-radius: 6px; margin: 3px; font-size: 0.8rem; font-weight: 500; }
    .skill-matched { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.2); }
    .skill-missing  { background: rgba(239,68,68,0.1);  color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }
    .skill-bonus    { background: rgba(99,102,241,0.1); color: #818cf8; border: 1px solid rgba(99,102,241,0.2); }

    /* Candidate card */
    .candidate-header {
        display: flex; align-items: center; justify-content: space-between;
        padding: 1.25rem 1.5rem; background: #111827; border: 1px solid #1e293b;
        border-radius: 12px; margin-bottom: 1rem; transition: all 0.2s ease;
    }
    .candidate-header:hover { border-color: #3b82f6; background: #1e293b; transform: translateY(-2px); }
    .cand-name { font-size: 1.1rem; font-weight: 600; color: #f8fafc; margin-bottom: 0.2rem; }
    .cand-score { font-size: 1.5rem; font-weight: 700; color: #10b981; margin-right: 1.5rem; }
    
    .badge {
        display: inline-block; padding: 4px 12px; border-radius: 9999px;
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em;
    }
    .badge-hire   { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.2); }
    .badge-maybe  { background: rgba(245,158,11,0.1); color: #f59e0b; border: 1px solid rgba(245,158,11,0.2); }
    .badge-nohire { background: rgba(239,68,68,0.1);  color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }

    /* Sidebar tweaks */
    .sidebar-label { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; margin: 1.5rem 0 0.5rem 0; }
    .pipeline-box { background: #111827; border: 1px solid #1e293b; border-radius: 8px; padding: 1rem; font-size: 0.85rem; color: #94a3b8; line-height: 1.8; }

    /* Tabs Override */
    button[data-baseweb="tab"] { font-size: 0.9rem !important; font-weight: 500 !important; color: #94a3b8 !important; border-radius: 6px !important; }
    button[data-baseweb="tab"][aria-selected="true"] { color: #10b981 !important; background: rgba(16,185,129,0.1) !important; }
    div[data-baseweb="tab-highlight"] { background-color: #10b981 !important; }
    
    /* Dataframe */
    .stDataFrame { border: 1px solid #1e293b !important; border-radius: 8px !important; }

    /* Progress */
    .stProgress > div > div { background: linear-gradient(90deg, #10b981, #059669) !important; }
</style>
""", unsafe_allow_html=True)

# Session State
if "results" not in st.session_state:
    st.session_state.results = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]




# ══════════════════════════════════════════════
# CHART FUNCTIONS
# ══════════════════════════════════════════════

def create_radar_chart(evaluations):
    """Radar chart comparing candidates across 5 dimensions."""
    fig = go.Figure()
    dims = ["Skills Match", "Experience Relevance", "Education & Certifications",
            "Projects / Portfolio", "Communication Quality"]
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6",
              "#06b6d4", "#f97316", "#ec4899"]
    for i, ev in enumerate(evaluations):
        scores = []
        for d in dims:
            s = next((ds["score"] for ds in ev.get("dimension_scores", [])
                      if ds["dimension"] == d), 5)
            scores.append(s)
        scores.append(scores[0])  # close the polygon
        fig.add_trace(go.Scatterpolar(
            r=scores, theta=dims + [dims[0]],
            fill='toself', name=ev.get("candidate", {}).get("name", "?"),
            line=dict(color=colors[i % len(colors)]),
            fillcolor=colors[i % len(colors)].replace(")", ",0.1)").replace("rgb", "rgba")
                      if "rgb" in colors[i % len(colors)] else None,
            opacity=0.85,
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], showticklabels=True,
                            gridcolor="#334155", linecolor="#475569"),
            angularaxis=dict(gridcolor="#334155", linecolor="#475569"),
            bgcolor="#1e293b",
        ),
        paper_bgcolor="#0f172a", font=dict(color="#f1f5f9"),
        legend=dict(bgcolor="#1e293b", bordercolor="#475569"),
        title=dict(text="📊 Candidate Comparison — 5 Dimensions", font=dict(size=16)),
        height=500,
    )
    return fig


def create_ranking_bars(evaluations):
    """Horizontal bar chart of candidate rankings."""
    names = [ev.get("candidate", {}).get("name", "?") for ev in evaluations]
    scores = [ev.get("weighted_total", 0) for ev in evaluations]
    colors = ["#10b981" if s >= 75 else "#f59e0b" if s >= 50 else "#ef4444" for s in scores]
    recs = [ev.get("recommendation", "?") for ev in evaluations]

    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.1f} — {r}" for s, r in zip(scores, recs)],
        textposition='auto', textfont=dict(color="white", size=12),
    ))
    fig.update_layout(
        title=dict(text="🏆 Candidate Rankings", font=dict(size=16, color="#f1f5f9")),
        xaxis=dict(title="Score (0-100)", range=[0, 100], gridcolor="#334155",
                   color="#94a3b8"),
        yaxis=dict(autorange="reversed", color="#f1f5f9"),
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font=dict(color="#f1f5f9"), height=max(300, len(names) * 60 + 100),
    )
    return fig


def create_heatmap(evaluations):
    """Heatmap: candidates × dimensions."""
    dims = ["Skills Match", "Experience Relevance", "Education & Certifications",
            "Projects / Portfolio", "Communication Quality"]
    names = [ev.get("candidate", {}).get("name", "?") for ev in evaluations]
    z = []
    for ev in evaluations:
        row = []
        for d in dims:
            s = next((ds["score"] for ds in ev.get("dimension_scores", [])
                      if ds["dimension"] == d), 0)
            row.append(s)
        z.append(row)

    fig = go.Figure(go.Heatmap(
        z=z, x=[d.split(" ")[0] for d in dims], y=names,
        colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
        zmin=0, zmax=10, text=z, texttemplate="%{text}",
        textfont=dict(size=14, color="white"),
    ))
    fig.update_layout(
        title=dict(text="🔥 Score Heatmap", font=dict(size=16, color="#f1f5f9")),
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font=dict(color="#f1f5f9"), height=max(300, len(names) * 50 + 150),
        xaxis=dict(color="#94a3b8"), yaxis=dict(autorange="reversed", color="#f1f5f9"),
    )
    return fig


def create_skills_gap_chart(evaluation):
    """Bar chart showing matched vs missing vs bonus skills for one candidate."""
    gap = evaluation.get("skills_gap", {})
    matched = gap.get("matched_skills", [])
    missing_raw = gap.get("missing_skills", [])
    bonus = gap.get("bonus_skills", [])
    missing = [m.get("skill", m) if isinstance(m, dict) else m for m in missing_raw]

    categories = ["Matched ✓", "Missing ✗", "Bonus ★"]
    values = [len(matched), len(missing), len(bonus)]
    colors = ["#10b981", "#ef4444", "#8b5cf6"]

    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker=dict(color=colors, line=dict(width=0)),
        text=values, textposition='auto', textfont=dict(size=16, color="white"),
    ))
    name = evaluation.get("candidate", {}).get("name", "?")
    pct = gap.get("match_percentage", 0)
    fig.update_layout(
        title=dict(text=f"🧩 {name} — Skills Gap ({pct:.0f}% match)",
                   font=dict(size=14, color="#f1f5f9")),
        paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
        font=dict(color="#f1f5f9"), height=300,
        xaxis=dict(color="#94a3b8"), yaxis=dict(title="Count", color="#94a3b8", gridcolor="#334155"),
    )
    return fig


def create_score_gauge(score, name):
    """Gauge chart for a single candidate's weighted total."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title=dict(text=name, font=dict(size=14, color="#f1f5f9")),
        number=dict(suffix="/100", font=dict(size=20, color="#f1f5f9")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#94a3b8"),
            bar=dict(color="#3b82f6"),
            bgcolor="#1e293b",
            steps=[
                dict(range=[0, 50], color="rgba(239,68,68,0.2)"),
                dict(range=[50, 75], color="rgba(245,158,11,0.2)"),
                dict(range=[75, 100], color="rgba(16,185,129,0.2)"),
            ],
            threshold=dict(line=dict(color="#f1f5f9", width=2), thickness=0.75, value=score),
        ),
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a", font=dict(color="#f1f5f9"), height=250,
    )
    return fig


# ══════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════

st.markdown("""
<div class="page-header">
    <h1><span style="color:#10b981">🏢</span> HR Shortlisting Agent</h1>
    <p>AI-powered candidate evaluation &nbsp;&middot;&nbsp; 100% local &nbsp;&middot;&nbsp; no data leaves your machine</p>
    <div style="margin-top: 1rem;">
        <span class="header-pill">✨ Llama 3.2 local</span>
        <span class="header-pill">🔗 LangGraph pipeline</span>
        <span class="header-pill">🛡️ Bias detection on</span>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Job Description", "Upload Resumes", "Results", "Skills Gap", "Overrides"
])


# ── TAB 1: Job Description ──
with tab1:
    st.subheader("Enter or Upload Job Description")
    jd_method = st.radio("Input method:", ["Paste Text", "Upload File"], horizontal=True)

    if jd_method == "Paste Text":
        jd_text = st.text_area(
            "Job Description", height=300,
            placeholder="Paste the full job description here...",
            key="jd_text_input",
        )
    else:
        jd_file = st.file_uploader("Upload JD (TXT/PDF/DOCX)", type=["txt", "pdf", "docx"])
        jd_text = ""
        if jd_file:
            if jd_file.name.endswith(".txt"):
                jd_text = jd_file.read().decode("utf-8", errors="ignore")
            else:
                jd_text = detect_and_parse_bytes(jd_file.read(), jd_file.name)
            st.text_area("Extracted JD Text", jd_text, height=200, disabled=True)

    if jd_text:
        st.success(f"JD loaded — {len(jd_text)} characters")
        st.session_state.jd_text = jd_text

        # Quick JD parse preview (regex-based, instant — no LLM needed)
        # Extracts required/preferred skill keywords for quick sanity check.
        with st.expander("JD Preview — verify before analysis"):
            st.caption("This is a quick keyword scan, not the full LLM parse. Run Analysis for the complete extraction.")
            import re as _re
            # Find common tech keywords as a quick skill preview
            TECH_PATTERN = _re.compile(
                r'\b(python|java(?:script)?|typescript|react|angular|vue|node\.?js|'
                r'django|flask|fastapi|spring|aws|azure|gcp|docker|kubernetes|k8s|'
                r'sql|postgres|mysql|mongodb|redis|kafka|git|ci/?cd|ml|'
                r'machine learning|deep learning|nlp|llm|langchain|pytorch|tensorflow|'
                r'rust|go|c\+\+|scala|spark|hadoop|pandas|numpy|scikit|'
                r'rest api|graphql|agile|scrum|terraform|linux)\b',
                _re.IGNORECASE
            )
            found_skills = list(dict.fromkeys(
                m.group(0).lower() for m in TECH_PATTERN.finditer(jd_text)
            ))
            exp_match = _re.search(r'(\d+)\+?\s*(?:years?|yrs?)', jd_text, _re.IGNORECASE)
            min_exp = exp_match.group(1) + "+ years" if exp_match else "Not specified"

            c_s, c_e = st.columns([3, 1])
            with c_s:
                st.markdown("**Detected skill keywords**")
                if found_skills:
                    tags = "".join(
                        f'<span class="skill-tag skill-neutral">{s}</span>'
                        for s in found_skills[:30]
                    )
                    st.markdown(tags, unsafe_allow_html=True)
                else:
                    st.caption("No recognisable tech keywords found — check JD text")
            with c_e:
                st.markdown("**Min experience**")
                st.markdown(f"`{min_exp}`")

# ── TAB 2: Upload Resumes ──
with tab2:
    st.markdown('<div class="section-title">Resume Sources</div>', unsafe_allow_html=True)

    # ── Section 1: PDF / DOCX upload ──────────────────────────────────────
    st.markdown("**PDF / DOCX Resumes**")
    uploaded_files = st.file_uploader(
        "Upload one or more resume files",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="resume_upload",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) ready")

        # ── Resume Parsed Preview ──────────────────────────────────────────
        # Let user verify parser is working before running the full pipeline.
        if "resume_previews" not in st.session_state:
            st.session_state.resume_previews = {}

        preview_btn = st.button(
            "Preview Parsed Resumes (check parser output)",
            key="preview_btn",
            help="Runs the AI parser on each resume and shows extracted fields. "
                 "Does NOT run scoring — just lets you verify extraction is correct."
        )

        if preview_btn:
            model_ok, model_err = validate_model_exists()
            if not model_ok:
                st.warning(f"Model not ready — cannot parse: {model_err}")
            else:
                previews = {}
                # Cache raw bytes so Run Analysis doesn't get empty bytes
                if "resume_bytes_cache" not in st.session_state:
                    st.session_state.resume_bytes_cache = {}
                parse_progress = st.progress(0, text="Parsing resumes...")
                for i, f in enumerate(uploaded_files):
                    parse_progress.progress(
                        int((i + 1) / len(uploaded_files) * 100),
                        text=f"Parsing {f.name}..."
                    )
                    raw_bytes = f.read()
                    st.session_state.resume_bytes_cache[f.name] = raw_bytes  # cache!
                    raw_text  = detect_and_parse_bytes(raw_bytes, f.name)
                    try:
                        profile = parse_resume(raw_text, f.name)
                        previews[f.name] = {"ok": True, "profile": profile, "raw_text": raw_text}
                    except Exception as e:
                        previews[f.name] = {"ok": False, "error": str(e), "raw_text": raw_text}
                parse_progress.empty()
                st.session_state.resume_previews = previews

        # Show previews if available
        if st.session_state.resume_previews:
            st.markdown('<div class="section-title">Parsed Resume Preview</div>', unsafe_allow_html=True)
            for fname, result in st.session_state.resume_previews.items():
                if not result["ok"]:
                    with st.expander(f"{fname}  —  Parse failed"):
                        st.error(f"Parser error: {result['error']}")
                    continue

                p = result["profile"]
                raw_text = result.get("raw_text", "")

                # Determine link status
                li_url = p.linkedin_url or _extract_linkedin_url(raw_text)
                gh_url = p.github_url  or _extract_github_url(raw_text)

                rec_label = f"Hire" if hasattr(p, 'recommendation') else ""

                with st.expander(
                    f"{p.name or fname}  —  {len(p.skills)} skills  —  "
                    f"{p.total_experience_years} yr(s) exp"
                ):
                    # Row 1: Contact + Links
                    r1_left, r1_right = st.columns(2)

                    with r1_left:
                        st.markdown("**Contact**")
                        if p.email:
                            st.markdown(f"Email: {p.email}")
                        else:
                            st.caption("Email: not found in resume")

                        if p.phone:
                            st.markdown(f"Phone: {p.phone}")
                        else:
                            st.caption("Phone: not found in resume")

                    with r1_right:
                        st.markdown("**Profile Links**")

                        # LinkedIn
                        if li_url:
                            st.markdown(f"LinkedIn: [{li_url}]({li_url})")
                        else:
                            st.markdown(
                                '<div style="font-size:0.82rem;color:#f87171;padding:2px 0">'
                                'LinkedIn: link not found in resume — candidate may need to add it'
                                '</div>',
                                unsafe_allow_html=True
                            )

                        # GitHub
                        if gh_url:
                            st.markdown(f"GitHub: [{gh_url}]({gh_url})")
                        else:
                            st.markdown(
                                '<div style="font-size:0.82rem;color:#94a3b8;padding:2px 0">'
                                'GitHub: not found in resume'
                                '</div>',
                                unsafe_allow_html=True
                            )

                    st.divider()

                    # Row 2: Skills
                    st.markdown("**Extracted Skills**")
                    if p.skills:
                        tags_html = "".join(
                            f'<span class="skill-tag skill-neutral">{s}</span>'
                            for s in p.skills
                        )
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.caption("No skills extracted — check if PDF is machine-readable")

                    st.divider()

                    # Row 3: Experience + Education side by side
                    r3_left, r3_right = st.columns(2)

                    with r3_left:
                        st.markdown(f"**Experience ({p.total_experience_years} yr)**")
                        if p.experience:
                            for exp in p.experience[:4]:
                                st.markdown(
                                    f'<div style="font-size:0.82rem;margin:4px 0">'
                                    f'<strong>{exp.title}</strong> at {exp.company}'
                                    f'<br><span style="color:#64748b">{exp.duration}</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                        else:
                            st.caption("No experience entries extracted")

                    with r3_right:
                        st.markdown("**Education**")
                        if p.education:
                            for edu in p.education[:3]:
                                st.markdown(
                                    f'<div style="font-size:0.82rem;margin:4px 0">'
                                    f'<strong>{edu.degree}</strong> — {edu.field}'
                                    f'<br><span style="color:#64748b">{edu.institution} ({edu.year})</span>'
                                    f'</div>',
                                    unsafe_allow_html=True
                                )
                        else:
                            st.caption("No education entries extracted")

                    # Certifications if any
                    if p.certifications:
                        st.markdown("**Certifications**")
                        st.caption(" · ".join(p.certifications))

                    # Summary
                    if p.summary:
                        st.markdown("**Summary**")
                        st.caption(p.summary[:400] + ("..." if len(p.summary) > 400 else ""))

                    # ── Raw text debug panel (collapsed) ──────────────────
                    st.divider()
                    char_count = len(raw_text)
                    with st.expander(
                        f"🔍 Raw extracted text ({char_count:,} chars) — inspect if parsing looks wrong",
                        expanded=False,
                    ):
                        if char_count < 50:
                            st.error(
                                "⚠️ Very little text extracted. The PDF may be image-based (scanned) "
                                "or use incompatible font encoding. "
                                "Try re-exporting from Word/Google Docs as a native PDF."
                            )
                        else:
                            st.caption(
                                "This is the exact text the LLM parser receives. "
                                "If your name or skills are missing here, the PDF format is the cause — not the AI."
                            )
                            st.text_area(
                                "Raw text",
                                value=raw_text[:3000] + ("\n\n[truncated…]" if char_count > 3000 else ""),
                                height=300,
                                disabled=True,
                                key=f"raw_txt_{fname.replace('.', '_')}",
                            )

    st.divider()

    # ── Section 2: LinkedIn Fetcher (Free — linkedin-api library) ──────────
    st.markdown("**LinkedIn Profile Fetcher**")

    from services.linkedin_fetcher import is_available as li_available

    if not li_available():
        st.markdown(
            '<div class="warn-block">'
            '<strong>LinkedIn credentials not configured</strong><br>'
            'This uses the free <code>linkedin-api</code> library — no subscription needed.<br><br>'
            'Add your LinkedIn login to <code>.env</code>:<br>'
            '<code>LINKEDIN_EMAIL=your-email@gmail.com</code><br>'
            '<code>LINKEDIN_PASSWORD=your-password</code><br><br>'
            'Use a secondary LinkedIn account if you prefer to keep your main account separate.'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="info-block">'
            'LinkedIn credentials configured — paste profile URLs below to fetch any public profile automatically.'
            '</div>',
            unsafe_allow_html=True
        )

    if "fetched_linkedin_profiles" not in st.session_state:
        st.session_state.fetched_linkedin_profiles = []

    # URL input — one per line
    li_urls_text = st.text_area(
        "LinkedIn Profile URLs (one per line)",
        height=110,
        placeholder=(
            "https://www.linkedin.com/in/johndoe/\n"
            "https://www.linkedin.com/in/janedoe/\n"
            "williamhgates"
        ),
        disabled=not li_available(),
        key="li_urls_text",
    )

    col_fetch, col_clear = st.columns([2, 1])
    with col_fetch:
        fetch_btn = st.button(
            "Fetch LinkedIn Profiles",
            disabled=not li_available() or not li_urls_text.strip(),
            type="primary" if li_available() else "secondary",
        )
    with col_clear:
        if st.button("Clear queue", disabled=not st.session_state.fetched_linkedin_profiles):
            st.session_state.fetched_linkedin_profiles = []
            st.rerun()

    if fetch_btn and li_urls_text.strip():
        from services.linkedin_fetcher import fetch_linkedin_profile, extract_username

        urls = [u.strip() for u in li_urls_text.strip().splitlines() if u.strip()]
        fetched, errors = [], []

        prog = st.progress(0, text="Fetching profiles...")
        for i, url in enumerate(urls):
            prog.progress(int((i + 1) / len(urls) * 100), text=f"Fetching {i+1}/{len(urls)}: {url}")
            profile, err = fetch_linkedin_profile(url)
            if err:
                errors.append(f"{url}  →  {err}")
            else:
                fetched.append(profile)

        prog.empty()
        # Merge with existing queue (avoid duplicates by linkedin_url)
        existing_urls = {p.linkedin_url for p in st.session_state.fetched_linkedin_profiles}
        new_profiles  = [p for p in fetched if p.linkedin_url not in existing_urls]
        st.session_state.fetched_linkedin_profiles += new_profiles

        if new_profiles:
            st.success(f"{len(new_profiles)} profile(s) fetched and added to queue.")
        elif fetched:
            st.info("All fetched profiles were already in the queue.")
        for e in errors:
            st.warning(e)

    # ── Profile cards ──────────────────────────────────────────────────────
    if st.session_state.fetched_linkedin_profiles:
        st.markdown(
            f'<div class="info-block">'
            f'<strong>{len(st.session_state.fetched_linkedin_profiles)}</strong> '
            f'LinkedIn profile(s) queued for analysis</div>',
            unsafe_allow_html=True
        )
        for idx, p in enumerate(st.session_state.fetched_linkedin_profiles):
            with st.expander(
                f"{p.name}  |  {len(p.skills)} skills  |  "
                f"{p.total_experience_years} yr exp"
            ):
                c_l, c_r = st.columns(2)
                with c_l:
                    st.markdown("**Contact**")
                    st.markdown(p.email or "_Email not public_")
                    if p.phone:
                        st.markdown(p.phone)

                    # LinkedIn link
                    if p.linkedin_url:
                        st.markdown(f"[View LinkedIn profile]({p.linkedin_url})")
                    else:
                        st.markdown(
                            '<div style="font-size:0.82rem;color:#f87171">'
                            'LinkedIn URL could not be accessed</div>',
                            unsafe_allow_html=True,
                        )

                    # GitHub link
                    if p.github_url:
                        st.markdown(f"[GitHub]({p.github_url})")
                    else:
                        st.markdown(
                            '<div style="font-size:0.82rem;color:#94a3b8">'
                            'GitHub not found in profile</div>',
                            unsafe_allow_html=True,
                        )

                with c_r:
                    st.markdown("**Skills**")
                    if p.skills:
                        tags = "".join(
                            f'<span class="skill-neutral skill-tag">{s}</span>'
                            for s in p.skills[:18]
                        )
                        st.markdown(tags, unsafe_allow_html=True)
                    else:
                        st.caption("No skills extracted")

                if p.experience:
                    st.caption(
                        f"Latest role: {p.experience[0].title} at "
                        f"{p.experience[0].company} ({p.experience[0].duration})"
                    )
                if p.summary:
                    st.caption(p.summary[:200])

                # Remove individual profile button
                if st.button(f"Remove from queue", key=f"rm_li_{idx}"):
                    st.session_state.fetched_linkedin_profiles.pop(idx)
                    st.rerun()

    # ── Manual JSON fallback (collapsed) ──────────────────────────────────
    with st.expander("Manual JSON entry (fallback if linkedin-api fails)"):
        st.caption(
            "Use this if automatic fetching fails. "
            "Fill in the template below for each candidate."
        )
        num_li = st.number_input("Number of profiles", min_value=1, max_value=10, value=1, step=1)
        li_raw_inputs = []
        for idx in range(int(num_li)):
            raw = st.text_area(
                f"Profile {idx + 1}",
                height=140,
                key=f"li_json_{idx}",
                placeholder=(
                    '{\n'
                    '  "name": "Jane Doe",\n'
                    '  "email": "jane@example.com",\n'
                    '  "linkedin_url": "https://linkedin.com/in/janedoe",\n'
                    '  "github_url": "https://github.com/janedoe",\n'
                    '  "skills": ["Python", "Machine Learning", "Docker"],\n'
                    '  "experience": [{"title": "ML Engineer", "company": "Acme", "duration": "2021-Present"}],\n'
                    '  "education": [{"degree": "B.Tech", "field": "CS", "institution": "IIT", "year": "2021"}],\n'
                    '  "summary": "3 years ML experience."\n'
                    '}'
                ),
            )
            if raw.strip():
                li_raw_inputs.append((idx + 1, raw.strip()))

        if st.button("Add manual profiles to queue", disabled=not li_raw_inputs):
            parsed, errors = [], []
            for idx, raw in li_raw_inputs:
                try:
                    profile = parse_linkedin_json(json.loads(raw))
                    parsed.append(profile)
                except json.JSONDecodeError as e:
                    errors.append(f"Profile {idx}: Invalid JSON — {e}")
                except Exception as e:
                    errors.append(f"Profile {idx}: {e}")
            st.session_state.fetched_linkedin_profiles += parsed
            if parsed:
                st.success(f"{len(parsed)} profile(s) added.")
            for e in errors:
                st.warning(e)


    st.divider()
    linkedin_json = ""  # backward-compat placeholder




    # ── Run Analysis Button ────────────────────────────────────────────────
    if st.button("Run Analysis", type="primary", width="stretch"):
        jd = st.session_state.get("jd_text", "")
        li_profiles = st.session_state.get("fetched_linkedin_profiles", [])
        li_json_txt = linkedin_json if "linkedin_json" in dir() else ""

        if not jd:
            st.error("Please enter a Job Description first (Job Description tab).")
        elif not uploaded_files and not li_profiles and not li_json_txt:
            st.error("Please upload at least one resume or fetch a LinkedIn profile.")
        else:
            resumes = []

            # Source A: PDF / DOCX files
            if uploaded_files:
                bytes_cache = st.session_state.get("resume_bytes_cache", {})
                for f in uploaded_files:
                    # Use cached bytes if available (avoids empty read after preview)
                    if f.name in bytes_cache:
                        raw_bytes = bytes_cache[f.name]
                    else:
                        f.seek(0)
                        raw_bytes = f.read()
                    content = detect_and_parse_bytes(raw_bytes, f.name)
                    ext = os.path.splitext(f.name)[1].lower().strip(".")
                    resumes.append({
                        "filename": f.name,
                        "content": content,
                        "source": ext if ext in ["pdf", "docx"] else "pdf",
                    })

            # Source B: LinkedIn profiles fetched via RapidAPI
            for p in li_profiles:
                resumes.append({
                    "filename": f"linkedin_{p.name.replace(' ', '_')}.json",
                    "content": p.raw_text,
                    "source": "linkedin",
                    "candidate_profile": p,   # pre-parsed — skip parser agent
                })

            # Source C: Manual LinkedIn JSON paste (fallback)
            if li_json_txt and li_json_txt.strip():
                try:
                    li_data = json.loads(li_json_txt)
                    li_profile = parse_linkedin_json(li_data)
                    resumes.append({
                        "filename": "linkedin_manual.json",
                        "content": li_profile.raw_text,
                        "source": "linkedin",
                    })
                except json.JSONDecodeError:
                    st.error("Invalid LinkedIn JSON format.")

            if resumes:
                critic_on = st.session_state.get("use_critic", False)
                current_model = st.session_state.get("selected_model", "llama3.2:3b")
                total_resumes = len(resumes)

                # Security: Rate limit check
                if st.session_state.run_count >= MAX_RUNS_PER_SESSION:
                    st.markdown(
                        '<div class="warn-block"><strong>Rate limit reached</strong><br>'
                        f'Maximum {MAX_RUNS_PER_SESSION} analysis runs per session. '
                        'Refresh the page to start a new session.</div>',
                        unsafe_allow_html=True
                    )
                    st.stop()

                # Pre-flight check: model must be available
                model_ok, model_err = validate_model_exists()
                if not model_ok:
                    st.markdown(
                        f'<div class="warn-block"><strong>Model not available</strong><br><br>'
                        f'<pre style="margin:0;background:transparent;color:#fbbf24;font-size:0.8rem">{model_err}</pre></div>',
                        unsafe_allow_html=True
                    )
                    st.stop()

                progress_bar = st.progress(0, text="Starting pipeline...")
                status_box   = st.empty()

                try:
                    # ── Semantic pre-filtering (BGE) ──────────────────────
                    EMBED_THRESHOLD = 5
                    try:
                        from services.embedder import rank_resumes_by_similarity, is_available as embed_ok
                        if embed_ok() and total_resumes >= EMBED_THRESHOLD:
                            status_box.info(f"Computing semantic similarity for {total_resumes} resumes (BGE)...")
                            progress_bar.progress(5, text="Embedding resumes...")
                            resumes = rank_resumes_by_similarity(jd, resumes)
                            sim_info = "  |  ".join(
                                f"{r.get('filename','?')} {r.get('semantic_similarity', 0):.2f}"
                                for r in resumes
                            )
                            status_box.info(f"Similarity: {sim_info}")
                            st.session_state.semantic_scores = {
                                r.get("filename", f"resume_{i}"): r.get("semantic_similarity", 0)
                                for i, r in enumerate(resumes)
                            }
                    except Exception:
                        pass  # Fallback — embedder not installed

                    for i, r in enumerate(resumes):
                        pct = int(10 + (i / max(total_resumes, 1)) * 75)
                        fname = r.get("filename", f"Resume {i+1}")
                        sim = r.get("semantic_similarity")
                        sim_label = f"  (sim: {sim:.2f})" if sim is not None else ""
                        status_box.info(f"Processing {i+1}/{total_resumes}: **{fname}**{sim_label}")
                        progress_bar.progress(pct, text=f"Parsing {fname}...")

                    status_box.info(f"Scoring with {current_model}...")
                    progress_bar.progress(90, text="AI scoring in progress...")

                    results = run_pipeline(
                        jd, resumes,
                        st.session_state.session_id,
                        use_critic=critic_on,
                    )
                    progress_bar.progress(100, text="Complete!")
                    status_box.empty()
                    st.session_state.results = results
                    st.session_state.run_count += 1
                    n = len(results.get("evaluations", []))
                    st.success(f"Analysis complete — {n} candidate(s) evaluated.")
                    if results.get("errors"):
                        with st.expander("Warnings"):
                            for err in results["errors"]:
                                st.warning(err)
                except Exception as e:
                    progress_bar.empty()
                    status_box.empty()
                    st.error(f"Pipeline error: {str(e)}")



# ── TAB 3: Results & Charts ──
with tab3:
    results = st.session_state.results
    if not results or not results.get("evaluations"):
        st.markdown('<div class="info-block">No results yet. Go to Upload Resumes, add a Job Description and candidates, then click Run Analysis.</div>', unsafe_allow_html=True)
    else:
        evaluations = results["evaluations"]

        # Metric cards
        total = len(evaluations)
        hire = sum(1 for e in evaluations if e.get("recommendation") == "Hire")
        maybe = sum(1 for e in evaluations if e.get("recommendation") == "Maybe")
        nohire = sum(1 for e in evaluations if e.get("recommendation") == "No Hire")

        st.markdown(f'''
        <div class="metric-row">
            <div class="metric-card total"><div class="metric-value">{total}</div><div class="metric-label">Resumes uploaded</div></div>
            <div class="metric-card hire"><div class="metric-value">{hire}</div><div class="metric-label">Shortlisted</div></div>
            <div class="metric-card maybe"><div class="metric-value">{maybe}</div><div class="metric-label">Maybe</div></div>
            <div class="metric-card nohire"><div class="metric-value">{nohire}</div><div class="metric-label">No hire</div></div>
        </div>
        ''', unsafe_allow_html=True)

        st.divider()

        # Charts row 1: Ranking bars + Radar
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(create_ranking_bars(evaluations), width="stretch")
        with col_b:
            st.plotly_chart(create_radar_chart(evaluations), width="stretch")

        # Chart row 2: Heatmap
        st.plotly_chart(create_heatmap(evaluations), width="stretch")

        # Gauge charts row
        st.markdown('<div class="section-title">Individual Score Gauges</div>', unsafe_allow_html=True)
        gauge_cols = st.columns(min(len(evaluations), 4))
        for i, ev in enumerate(evaluations[:8]):
            with gauge_cols[i % len(gauge_cols)]:
                st.plotly_chart(
                    create_score_gauge(ev.get("weighted_total", 0),
                                       ev.get("candidate", {}).get("name", "?")),
                    width="stretch",
                )

        # Detailed cards
        st.divider()
        st.markdown('<div class="section-title">Detailed Evaluations</div>', unsafe_allow_html=True)
        sem_scores = st.session_state.get("semantic_scores", {})
        for ev in evaluations:
            cand = ev.get("candidate", {})
            rec = ev.get("recommendation", "?")
            fname = cand.get("source_file", cand.get("name", "?"))
            sem_sim = sem_scores.get(fname, ev.get("semantic_similarity"))
            sem_label = f"  |  Semantic sim: {sem_sim:.2f}" if sem_sim is not None else ""

            score = ev.get('weighted_total', 0)
            score_color = "#10b981" if score >= 75 else "#f59e0b" if score >= 50 else "#ef4444"
            badge_class = "badge-hire" if rec == "Hire" else "badge-maybe" if rec == "Maybe" else "badge-nohire"
            
            st.markdown(f'''
            <div class="candidate-header">
                <div style="display: flex; align-items: center;">
                    <div style="width:36px; height:36px; border-radius:50%; background:#1e293b; border:1px solid #334155; color:#94a3b8; display:flex; align-items:center; justify-content:center; font-weight:600; margin-right:1.25rem;">{ev.get('rank', '?')}</div>
                    <div>
                        <div class="cand-name">{cand.get('name', '?')}</div>
                        <div style="font-size:0.85rem; color:#94a3b8;">{cand.get('job_title', 'Developer')} - {cand.get('total_experience_years', '?')} yrs exp</div>
                    </div>
                </div>
                <div style="display:flex; align-items:center; gap:1.5rem;">
                    <div style="font-size:1.5rem; font-weight:700; color:{score_color};">{score:.2f}</div>
                    <div class="badge {badge_class}">{rec}</div>
                </div>
            </div>
            ''', unsafe_allow_html=True)

            with st.expander(f"View breakdown..."):
                links_html = []
                if cand.get("email"):
                    links_html.append(f'<a href="mailto:{cand["email"]}">{cand["email"]}</a>')
                if cand.get("phone"):
                    links_html.append(f'<span>{cand["phone"]}</span>')
                if cand.get("linkedin_url"):
                    links_html.append(f'<a href="{cand["linkedin_url"]}" target="_blank">LinkedIn Profile</a>')
                if cand.get("github_url"):
                    links_html.append(f'<a href="{cand["github_url"]}" target="_blank">GitHub</a>')
                if links_html:
                    st.markdown(f'<div class="profile-links">{" &nbsp;|&nbsp; ".join(links_html)}</div>', unsafe_allow_html=True)

                # Score table
                df = pd.DataFrame(ev.get("dimension_scores", []))
                if not df.empty:
                    df["weight"] = df["weight"].apply(lambda w: f"{w*100:.0f}%")
                    df["score"] = df["score"].apply(lambda s: f"{s}/10")
                    st.dataframe(df[["dimension", "weight", "score", "justification"]],
                                 width="stretch", hide_index=True)

                # Confidence & bias
                conf = ev.get("confidence", 0)
                st.metric("Confidence", f"{conf:.0%}")
                flags = ev.get("bias_flags", [])
                if flags:
                    for f in flags:
                        st.warning(f"⚠️ {f}")

        # Fairness report
        st.divider()
        st.subheader("🛡️ Fairness Audit")
        fairness = generate_fairness_report(evaluations)
        if fairness.get("flags"):
            for flag in fairness["flags"]:
                st.warning(flag)
        else:
            st.success("✅ No significant bias patterns detected.")
        st.json(fairness)

        # Download buttons
        st.divider()
        st.subheader("📥 Download Reports")
        dcol1, dcol2, dcol3 = st.columns(3)
        report_data = {
            "session_id": st.session_state.session_id,
            "jd": results.get("jd_parsed", {}),
            "evaluations": evaluations,
            "overrides": get_overrides(st.session_state.session_id),
            "generated_at": __import__("datetime").datetime.now().isoformat(),
            "model_used": "llama3",
            "total_candidates": total, "hire_count": hire,
            "maybe_count": maybe, "no_hire_count": nohire,
        }
        with dcol1:
            if st.button("📄 Generate PDF", width="stretch"):
                path = generate_pdf_report(report_data)
                with open(path, "rb") as f:
                    st.download_button("⬇️ Download PDF", f.read(), "shortlist_report.pdf",
                                       "application/pdf", width="stretch")
        with dcol2:
            if st.button("🌐 Generate HTML", width="stretch"):
                path = generate_html_report(report_data)
                with open(path, "r", encoding="utf-8") as f:
                    st.download_button("⬇️ Download HTML", f.read(), "shortlist_report.html",
                                       "text/html", width="stretch")
        with dcol3:
            json_str = json.dumps(report_data, indent=2, default=str)
            st.download_button("📊 Download JSON", json_str, "shortlist_report.json",
                               "application/json", width="stretch")

# ── TAB 4: Skills Gap ──
with tab4:
    results = st.session_state.results
    if not results or not results.get("evaluations"):
        st.info("📭 Run analysis first to see skills gap data.")
    else:
        evaluations = results["evaluations"]
        st.subheader("🧩 Skills Gap Analysis — Per Candidate")

        for ev in evaluations:
            cand = ev.get("candidate", {})
            gap = ev.get("skills_gap", {})
            matched = gap.get("matched_skills", [])
            missing_raw = gap.get("missing_skills", [])
            bonus = gap.get("bonus_skills", [])
            cand_name = cand.get("name", "?")
            cand_skills = cand.get("skills", [])

            # Parse-failed guard
            if cand_name == "Parse Failed" or not cand_skills:
                st.warning(
                    f"⚠️ **{cand_name}**: Resume parsing returned no skills — matching impossible. "
                    "Check that the PDF has selectable text (not a scanned image), or try a different model."
                )
                st.divider()
                continue

            st.markdown(f"### {cand_name} — {gap.get('match_percentage', 0):.0f}% match")

            # Show extracted skills
            with st.expander(f"📄 Extracted skills from resume ({len(cand_skills)} found)"):
                tags = " ".join(
                    f'<span style="background:#1e3a5f;color:#60a5fa;padding:3px 9px;border-radius:5px;'
                    f'margin:2px;display:inline-block;font-size:0.82rem">{s}</span>'
                    for s in cand_skills
                )
                st.markdown(tags, unsafe_allow_html=True)

            # Skills gap bar chart
            st.plotly_chart(create_skills_gap_chart(ev), width="stretch")

            # Skill tags with counts
            col_m, col_x, col_b = st.columns(3)
            with col_m:
                st.markdown(f"**✓ Matched Skills ({len(matched)})**")
                tags = "".join(f'<span class="skill-matched">{s}</span>' for s in matched)
                st.markdown(tags or "_None_", unsafe_allow_html=True)
            with col_x:
                st.markdown(f"**✗ Missing Skills ({len(missing_raw)})**")
                for m in missing_raw:
                    skill = m.get("skill", m) if isinstance(m, dict) else m
                    resource = m.get("resource", "") if isinstance(m, dict) else ""
                    time_est = m.get("estimated_time", "") if isinstance(m, dict) else ""
                    st.markdown(f'<span class="skill-missing">{skill}</span>', unsafe_allow_html=True)
                    if resource:
                        st.caption(f"📚 {resource} ({time_est})")
                if not missing_raw:
                    st.markdown("_None — perfect match!_")
            with col_b:
                st.markdown(f"**★ Bonus Skills ({len(bonus)})**")
                tags = "".join(f'<span class="skill-bonus">{s}</span>' for s in bonus)
                st.markdown(tags or "_None_", unsafe_allow_html=True)

            st.divider()

# ── TAB 5: Overrides ──
with tab5:
    results = st.session_state.results
    if not results or not results.get("evaluations"):
        st.info("📭 Run analysis first to override scores.")
    else:
        evaluations = results["evaluations"]
        st.subheader("✏️ Human-in-the-Loop Score Override")
        st.caption("Adjust scores and provide justification. All changes are logged.")

        candidate_names = [ev.get("candidate", {}).get("name", "?") for ev in evaluations]
        selected = st.selectbox("Select Candidate", candidate_names)

        dims = ["Skills Match", "Experience Relevance", "Education & Certifications",
                "Projects / Portfolio", "Communication Quality"]

        if selected:
            ev = next(e for e in evaluations if e.get("candidate", {}).get("name") == selected)
            sel_dim = st.selectbox("Dimension", dims)
            current = 5
            justification = "No justification provided."
            for ds in ev.get("dimension_scores", []):
                if ds["dimension"] == sel_dim:
                    current = ds["score"]
                    justification = ds.get("justification", justification)
                    break

            st.markdown(f"**AI Original Score:** {current}/10")
            st.info(f"**AI Justification:** {justification}")

            new_score = st.slider("New Score", 0, 10, current)
            reason = st.text_input("Reason for override")
            reviewer = st.text_input("Reviewer Name", "HR Reviewer")
            flagged = st.checkbox("🚩 Flag candidate for further review")

            if st.button("💾 Save Override", type="primary"):
                if not reason:
                    st.error("Please provide a reason.")
                else:
                    override = {
                        "candidate_name": selected, "dimension": sel_dim,
                        "original_score": current, "new_score": new_score,
                        "reason": reason, "reviewer": reviewer, "flagged": flagged,
                    }
                    save_override(st.session_state.session_id, override)

                    # Update in-memory evaluation
                    for ds in ev.get("dimension_scores", []):
                        if ds["dimension"] == sel_dim:
                            ds["score"] = new_score
                    total = sum(ds["score"] * ds["weight"] * 10
                                for ds in ev.get("dimension_scores", []))
                    ev["weighted_total"] = round(total, 2)
                    ev["recommendation"] = ("Hire" if total >= 75 else
                                            "Maybe" if total >= 50 else "No Hire")
                    st.success(f"✅ Override saved. New total: {total:.1f}/100")

        # Show override history
        st.divider()
        st.subheader("📜 Override Audit Log")
        overrides = get_overrides(st.session_state.session_id)
        if overrides:
            st.dataframe(pd.DataFrame(overrides), width="stretch", hide_index=True)
        else:
            st.info("No overrides recorded yet.")

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style="display:flex; align-items:center; gap:0.75rem; margin-bottom:1.5rem;">
        <div style="background:#10b981; padding:0.4rem; border-radius:8px;">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
        </div>
        <div>
            <h2 style="margin:0; font-size:1.1rem; color:#f8fafc; font-weight:600;">HR Shortlisting</h2>
            <div style="font-size:0.75rem; color:#94a3b8;">Agent v1.0</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Model selector
    MODEL_OPTIONS = {
        "llama3.2:3b   Fast  (~2GB VRAM)":    "llama3.2:3b",
        "llama3.2:1b   Fastest (~1GB VRAM)":  "llama3.2:1b",
        "llama3        Best quality (~4.5GB)": "llama3",
        "phi3:mini     Fast  (~2.3GB VRAM)":  "phi3:mini",
        "gemma2:2b     Fast  (~1.6GB VRAM)":  "gemma2:2b",
    }
    selected_model_label = st.selectbox(
        "Select Ollama model",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed"
    )
    selected_model = MODEL_OPTIONS[selected_model_label]
    os.environ["OLLAMA_MODEL"] = selected_model
    st.session_state.selected_model = selected_model
    
    # Check model status
    model_ok, _ = validate_model_exists()
    status_color = "#10b981" if model_ok else "#ef4444"
    
    st.markdown(f"""
    <div style="background:#111827; border:1px solid #1e293b; border-radius:8px; padding:0.85rem; margin-bottom:1.5rem;">
        <div style="display:flex; align-items:center; gap:0.5rem; color:{status_color}; font-size:0.85rem; font-weight:600; margin-bottom:0.25rem;">
            <div style="width:6px; height:6px; border-radius:50%; background:{status_color};"></div>
            {selected_model}
        </div>
        <div style="font-size:0.75rem; color:#94a3b8;">RTX 3050 · 100% local</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">PIPELINE</div>', unsafe_allow_html=True)
    
    use_critic = st.toggle("Enable Critic validation", value=False)
    st.session_state.use_critic = use_critic
    
    critic_color = "#10b981" if use_critic else "#64748b"
    critic_icon = "✓" if use_critic else "○"
    critic_text = "Critic agent" if use_critic else "Critic (off)"

    st.markdown(f"""
    <div style="background:#111827; border:1px solid #1e293b; border-radius:8px; padding:1rem; font-size:0.85rem;">
        <div style="display:flex; align-items:center; gap:0.5rem; color:#10b981; margin-bottom:0.5rem;"><span style="font-weight:700;">✓</span> Parser agent</div>
        <div style="display:flex; align-items:center; gap:0.5rem; color:#10b981; margin-bottom:0.5rem;"><span style="font-weight:700;">✓</span> Scorer agent</div>
        <div style="display:flex; align-items:center; gap:0.5rem; color:{critic_color}; margin-bottom:0.5rem;"><span style="font-weight:700;">{critic_icon}</span> {critic_text}</div>
        <div style="display:flex; align-items:center; gap:0.5rem; color:#10b981;"><span style="font-weight:700;">✓</span> Report agent</div>
    </div>
    """, unsafe_allow_html=True)

# ── Floating HR Assistant Chatbot ──
st.markdown("""
<style>
/* Target the parent container of the popover to make it float */
div[data-testid="stElementContainer"]:has(div[data-testid="stPopover"]), 
div.element-container:has(div[data-testid="stPopover"]) {
    position: fixed !important;
    bottom: 30px !important;
    right: 30px !important;
    width: auto !important;
    z-index: 999999 !important;
}

/* Style the popover button to be a circle icon */
div[data-testid="stPopover"] > button {
    width: 65px !important;
    height: 65px !important;
    border-radius: 50% !important;
    background-color: #3b82f6 !important;
    color: white !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.4) !important;
    border: none !important;
    font-size: 32px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
}
div[data-testid="stPopover"] > button:hover {
    background-color: #2563eb !important;
    transform: scale(1.05);
    transition: all 0.2s ease-in-out;
}
div[data-testid="stPopover"] > button p {
    font-size: 30px !important;
    margin: 0 !important;
    line-height: 1 !important;
}

/* Adjust popover content body */
div[data-testid="stPopoverBody"] {
    width: 350px !important;
    height: 500px !important;
    max-height: 80vh !important;
    overflow-y: hidden !important;
    background-color: #0f172a !important;
    border: 1px solid #1e2d45 !important;
    box-shadow: 0 10px 25px rgba(0,0,0,0.5) !important;
    border-radius: 12px !important;
    padding: 15px !important;
}
</style>
""", unsafe_allow_html=True)

with st.popover("✨"):
    st.markdown("### ✨ AI Assistant")
    st.caption("I can analyze the candidates for you. Ask me anything!")
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        
    # Container for chat history
    chat_container = st.container(height=340)
    with chat_container:
        for msg in st.session_state.chat_messages:
            avatar = "✨" if msg["role"] == "assistant" else None
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])
                
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        chat_container.chat_message("user").markdown(prompt)
        
        with chat_container.chat_message("assistant", avatar="✨"):
            if not st.session_state.results or not st.session_state.results.get("evaluations"):
                err_msg = "Please run an analysis first so I have candidate data to search through."
                st.markdown(err_msg)
                st.session_state.chat_messages.append({"role": "assistant", "content": err_msg})
            else:
                with st.spinner("Thinking..."):
                    from langchain_ollama import ChatOllama
                    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
                    import json
                    
                    evals = st.session_state.results["evaluations"]
                    candidates_context = []
                    for ev in evals:
                        cand = ev.get("candidate", {})
                        candidates_context.append({
                            "name": cand.get("name"),
                            "skills": cand.get("skills"),
                            "experience_years": cand.get("total_experience_years"),
                            "total_score": ev.get("weighted_total"),
                            "recommendation": ev.get("recommendation"),
                        })
                    
                    context_str = json.dumps(candidates_context, indent=2)
                    sys_prompt = f"""You are a helpful HR Assistant. Use the provided candidate data to answer the user's questions. Be concise and helpful.
CANDIDATE DATA:
{context_str}"""
                    
                    msgs = [SystemMessage(content=sys_prompt)]
                    for m in st.session_state.chat_messages[-5:-1]:
                        if m["role"] == "user":
                            msgs.append(HumanMessage(content=m["content"]))
                        else:
                            msgs.append(AIMessage(content=m["content"]))
                    
                    msgs.append(HumanMessage(content=prompt))
                    
                    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                    model = os.getenv("OLLAMA_MODEL", "llama3")
                    
                    from agents import get_langfuse_callbacks
                    chat_llm = ChatOllama(
                        model=model, 
                        base_url=base_url, 
                        temperature=0.2,
                        callbacks=get_langfuse_callbacks()
                    )
                    
                    try:
                        response = chat_llm.invoke(msgs)
                        answer = response.content
                    except Exception as e:
                        answer = f"Sorry, I encountered an error: {str(e)}"
                    
                    st.markdown(answer)
                    st.session_state.chat_messages.append({"role": "assistant", "content": answer})

