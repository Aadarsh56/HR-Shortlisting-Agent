"""
Report generator service.

Produces shortlist reports in three formats:
- PDF (via ReportLab) — professional, print-ready
- HTML (via Jinja2) — styled, embeddable
- JSON — machine-readable
"""

import json
import os
from datetime import datetime

from jinja2 import Template
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Colour helpers
# ──────────────────────────────────────────────

def _recommendation_color(rec: str):
    """Get colour for a recommendation label."""
    mapping = {
        "Hire": colors.HexColor("#10b981"),
        "Maybe": colors.HexColor("#f59e0b"),
        "No Hire": colors.HexColor("#ef4444"),
    }
    return mapping.get(rec, colors.grey)


def _score_color(score: float):
    """Get colour based on score value (0-100)."""
    if score >= 75:
        return colors.HexColor("#10b981")
    elif score >= 50:
        return colors.HexColor("#f59e0b")
    return colors.HexColor("#ef4444")


# ──────────────────────────────────────────────
# PDF Report (ReportLab)
# ──────────────────────────────────────────────

def generate_pdf_report(report_data: dict, filename: str = None) -> str:
    """Generate a professional PDF shortlist report.

    Args:
        report_data: ShortlistReport as a dict.
        filename: Optional filename; auto-generated if not provided.

    Returns:
        Absolute path to the generated PDF file.
    """
    _ensure_output_dir()
    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shortlist_report_{ts}.pdf"

    filepath = os.path.join(OUTPUT_DIR, filename)
    doc = SimpleDocTemplate(filepath, pagesize=A4,
                            leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=20 * mm, bottomMargin=20 * mm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Title'],
        fontSize=22, spaceAfter=12,
        textColor=colors.HexColor("#1e293b"),
    )
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'],
        fontSize=14, spaceAfter=8, spaceBefore=16,
        textColor=colors.HexColor("#334155"),
    )
    body_style = ParagraphStyle(
        'CustomBody', parent=styles['Normal'],
        fontSize=10, spaceAfter=6,
        textColor=colors.HexColor("#475569"),
    )

    story = []

    # ── Title Page ──
    story.append(Paragraph("HR Resume Shortlist Report", title_style))
    jd = report_data.get("jd", {})
    story.append(Paragraph(f"Position: <b>{jd.get('title', 'N/A')}</b>", body_style))
    story.append(Paragraph(
        f"Generated: {report_data.get('generated_at', datetime.now().isoformat())[:19]}",
        body_style,
    ))
    story.append(Paragraph(
        f"Total Candidates: {report_data.get('total_candidates', 0)} | "
        f"Model: {report_data.get('model_used', 'llama3')}",
        body_style,
    ))
    story.append(Spacer(1, 12))

    # ── Summary Table ──
    story.append(Paragraph("Candidate Rankings", heading_style))

    table_data = [["Rank", "Candidate", "Score", "Recommendation"]]
    evaluations = report_data.get("evaluations", [])
    for ev in evaluations:
        cand = ev.get("candidate", {})
        table_data.append([
            str(ev.get("rank", "-")),
            cand.get("name", "Unknown"),
            f"{ev.get('weighted_total', 0):.1f}/100",
            ev.get("recommendation", "N/A"),
        ])

    if len(table_data) > 1:
        summary_table = Table(table_data, colWidths=[50, 200, 80, 100])
        table_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [
                colors.HexColor("#f8fafc"), colors.HexColor("#e2e8f0")
            ]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ])
        summary_table.setStyle(table_style)
        story.append(summary_table)
    story.append(Spacer(1, 20))

    # ── Per-Candidate Detail ──
    for ev in evaluations:
        cand = ev.get("candidate", {})
        name = cand.get("name", "Unknown")
        rec = ev.get("recommendation", "N/A")
        total = ev.get("weighted_total", 0)

        story.append(Paragraph(
            f"#{ev.get('rank', '-')} — {name} | "
            f"Score: {total:.1f}/100 | Recommendation: {rec}",
            heading_style,
        ))

        # Dimension scores table
        dim_data = [["Dimension", "Weight", "Score", "Justification"]]
        for ds in ev.get("dimension_scores", []):
            dim_data.append([
                ds.get("dimension", ""),
                f"{ds.get('weight', 0) * 100:.0f}%",
                f"{ds.get('score', 0)}/10",
                Paragraph(ds.get("justification", ""), body_style),
            ])

        if len(dim_data) > 1:
            dim_table = Table(dim_data, colWidths=[120, 60, 60, 230])
            dim_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#334155")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(dim_table)

        # Skills gap
        sgap = ev.get("skills_gap", {})
        matched = sgap.get("matched_skills", [])
        missing = sgap.get("missing_skills", [])
        bonus = sgap.get("bonus_skills", [])

        if matched or missing or bonus:
            story.append(Spacer(1, 6))
            story.append(Paragraph("Skills Gap Analysis", ParagraphStyle(
                'SkillsHeader', parent=body_style,
                fontSize=11, textColor=colors.HexColor("#1e293b"),
                fontName='Helvetica-Bold',
            )))
            if matched:
                story.append(Paragraph(
                    f"✓ Matched: {', '.join(matched)}", body_style
                ))
            if missing:
                missing_strs = [
                    f"{m.get('skill', m) if isinstance(m, dict) else m}"
                    for m in missing
                ]
                story.append(Paragraph(
                    f"✗ Missing: {', '.join(missing_strs)}", body_style
                ))
            if bonus:
                story.append(Paragraph(
                    f"★ Bonus: {', '.join(bonus)}", body_style
                ))

        story.append(Spacer(1, 16))

    # ── Overrides section ──
    overrides = report_data.get("overrides", [])
    if overrides:
        story.append(Paragraph("Score Overrides (Audit Log)", heading_style))
        ovr_data = [["Candidate", "Dimension", "Old", "New", "Reason"]]
        for o in overrides:
            ovr_data.append([
                o.get("candidate_name", ""),
                o.get("dimension", ""),
                str(o.get("original_score", "")),
                str(o.get("new_score", "")),
                Paragraph(o.get("reason", ""), body_style),
            ])
        ovr_table = Table(ovr_data, colWidths=[100, 100, 40, 40, 190])
        ovr_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#92400e")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(ovr_table)

    doc.build(story)
    return filepath


# ──────────────────────────────────────────────
# HTML Report (Jinja2)
# ──────────────────────────────────────────────

def generate_html_report(report_data: dict, filename: str = None) -> str:
    """Generate a styled HTML shortlist report.

    Args:
        report_data: ShortlistReport as a dict.
        filename: Optional filename.

    Returns:
        Absolute path to the generated HTML file.
    """
    _ensure_output_dir()
    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shortlist_report_{ts}.html"

    filepath = os.path.join(OUTPUT_DIR, filename)

    # Load template
    template_path = os.path.join(TEMPLATE_DIR, "report.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            template_str = f.read()
    else:
        template_str = _get_default_html_template()

    template = Template(template_str)
    html_content = template.render(report=report_data)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)

    return filepath


def _get_default_html_template() -> str:
    """Fallback HTML template if templates/report.html doesn't exist."""
    return """<!DOCTYPE html>
<html><head><title>Shortlist Report</title>
<style>body{font-family:sans-serif;padding:20px;}</style>
</head><body>
<h1>Shortlist Report</h1>
<p>{{ report.total_candidates }} candidates evaluated.</p>
</body></html>"""


# ──────────────────────────────────────────────
# JSON Report
# ──────────────────────────────────────────────

def generate_json_report(report_data: dict, filename: str = None) -> str:
    """Generate a JSON shortlist report.

    Args:
        report_data: ShortlistReport as a dict.
        filename: Optional filename.

    Returns:
        Absolute path to the generated JSON file.
    """
    _ensure_output_dir()
    if not filename:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shortlist_report_{ts}.json"

    filepath = os.path.join(OUTPUT_DIR, filename)

    # Remove raw_text and anonymised_text to keep the JSON clean
    clean_data = json.loads(json.dumps(report_data, default=str))
    for ev in clean_data.get("evaluations", []):
        cand = ev.get("candidate", {})
        cand.pop("raw_text", None)
        cand.pop("anonymised_text", None)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(clean_data, f, indent=2, default=str)

    return filepath
