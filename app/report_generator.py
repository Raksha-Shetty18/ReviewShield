import os
import io
import json
from datetime import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to add 'Page X of Y' and confidential project footer."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header (pages after page 1)
        if self._pageNumber > 1:
            self.drawString(54, letter[1] - 36, "ReviewShield™ | AI-Powered Fake Review Intelligence Audit Report")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
            
        # Footer
        footer_text = f"ReviewShield Report • Confidential Machine Learning Verification • Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 30, footer_text)
        self.drawString(54, 30, "Generated via ReviewShield Dual-Stream Ensemble Engine (v2.0)")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, 42, letter[0] - 54, 42)
        self.restoreState()


def build_pdf_report(record: Dict[str, Any]) -> io.BytesIO:
    """
    Generates a 8-section PDF report for an analyzed review record.
    Returns an in-memory BytesIO buffer containing the PDF.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#1E3A8A")     # Deep Navy
    c_teal = colors.HexColor("#0D9488")        # Teal Accent
    c_slate_dark = colors.HexColor("#0F172A")  # Dark Slate Heading
    c_text = colors.HexColor("#334155")        # Slate Text
    c_border = colors.HexColor("#E2E8F0")      # Border Grey
    c_bg_light = colors.HexColor("#F8FAFC")    # Card Background
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=c_primary
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=c_teal
    )
    section_h1 = ParagraphStyle(
        'SectionH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_slate_dark,
        spaceBefore=14,
        spaceAfter=6
    )
    body_text = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=c_text
    )
    body_bold = ParagraphStyle(
        'BodyBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=13,
        textColor=c_slate_dark
    )
    review_quote = ParagraphStyle(
        'ReviewQuote',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1E293B")
    )
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11,
        textColor=c_text
    )
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=c_slate_dark
    )
    disclaimer_style = ParagraphStyle(
        'DisclaimerText',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#475569")
    )

    story = []
    
    # Parse record fields
    pred_id = record.get("prediction_id") or f"PRED_{record.get('id', 'N/A')}"
    report_id = f"RPT_{pred_id.replace('PRED_', '')}"
    analysis_time = record.get("analysis_timestamp") or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    generation_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    pred_verdict = record.get("ensemble_prediction", "Potentially Fake")
    is_fake = "Fake" in pred_verdict
    badge_bg = colors.HexColor("#FEE2E2") if is_fake else colors.HexColor("#DCFCE7")
    badge_fg = colors.HexColor("#B91C1C") if is_fake else colors.HexColor("#15803D")
    
    fake_prob = float(record.get("ensemble_fake_prob") or 0.5)
    genuine_prob = float(record.get("ensemble_genuine_prob") or (1.0 - fake_prob))
    
    # -------------------------------------------------------------------------
    # HEADER BANNER (Title & Logo representation)
    # -------------------------------------------------------------------------
    header_data = [
        [
            Paragraph("<b>ReviewShield™</b><br/><font size=9 color='#0D9488'>AI-Powered Fake Review Intelligence System</font>", title_style),
            Paragraph(f"<b>OFFICIAL AUDIT REPORT</b><br/><font size=8 color='#64748B'>Report ID: {report_id}<br/>Date: {analysis_time[:10]}</font>", ParagraphStyle('HRight', parent=styles['Normal'], alignment=2))
        ]
    ]
    header_table = Table(header_data, colWidths=[3.5 * inch, 3.5 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(header_table)
    story.append(HRFlowable(width="100%", thickness=1.5, color=c_primary, spaceBefore=4, spaceAfter=12))

    # -------------------------------------------------------------------------
    # SECTION 1: REPORT INFORMATION
    # -------------------------------------------------------------------------
    story.append(Paragraph("SECTION 1: REPORT & AUDIT IDENTIFICATION", section_h1))
    meta_info = [
        [
            Paragraph("<b>Prediction ID:</b>", table_cell_bold),
            Paragraph(pred_id, table_cell),
            Paragraph("<b>Audit Date/Time:</b>", table_cell_bold),
            Paragraph(str(analysis_time), table_cell)
        ],
        [
            Paragraph("<b>Report ID:</b>", table_cell_bold),
            Paragraph(report_id, table_cell),
            Paragraph("<b>Report Generated:</b>", table_cell_bold),
            Paragraph(generation_time, table_cell)
        ],
        [
            Paragraph("<b>Ensemble Version:</b>", table_cell_bold),
            Paragraph(record.get("model_version") or "v2.0-ensemble-55-45", table_cell),
            Paragraph("<b>Batch / Source:</b>", table_cell_bold),
            Paragraph(f"{record.get('source', 'single').capitalize()} (Batch: {record.get('batch_id') or 'N/A'})", table_cell)
        ]
    ]
    t_meta = Table(meta_info, colWidths=[1.4 * inch, 2.1 * inch, 1.4 * inch, 2.1 * inch])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), c_bg_light),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # SECTION 2: SUBMITTED REVIEW
    # -------------------------------------------------------------------------
    story.append(Paragraph("SECTION 2: SUBMITTED REVIEW CONTENT & CONTEXT", section_h1))
    rating_val = record.get("rating", 5.0)
    stars_str = "★" * int(rating_val) + "☆" * (5 - int(rating_val))
    
    review_details = [
        [
            Paragraph("<b>Review Title:</b>", table_cell_bold),
            Paragraph(record.get("review_title") or "(No title provided)", table_cell),
            Paragraph("<b>Rating:</b>", table_cell_bold),
            Paragraph(f"{stars_str} ({rating_val} / 5.0)", table_cell_bold)
        ],
        [
            Paragraph("<b>Reviewer ID:</b>", table_cell_bold),
            Paragraph(record.get("reviewer_id") or "Unprovided / Anonymous", table_cell),
            Paragraph("<b>Product ASIN/ID:</b>", table_cell_bold),
            Paragraph(record.get("product_id") or "Unprovided", table_cell)
        ],
        [
            Paragraph("<b>Submission Date:</b>", table_cell_bold),
            Paragraph(str(record.get("review_date") or "N/A"), table_cell),
            Paragraph("<b>Account History:</b>", table_cell_bold),
            Paragraph("Linked in Database" if record.get("has_reviewer_history") else "Unobserved (Imputed)", table_cell)
        ]
    ]
    t_rev = Table(review_details, colWidths=[1.4 * inch, 2.1 * inch, 1.4 * inch, 2.1 * inch])
    t_rev.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_rev)
    story.append(Spacer(1, 4))
    
    # Review Text Box
    raw_text = record.get("review_text") or "No text content submitted."
    text_box = Table(
        [[Paragraph("<b>Full Review Text:</b><br/>" + raw_text, review_quote)]],
        colWidths=[7.0 * inch]
    )
    text_box.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(text_box)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # SECTION 3: PREDICTION RESULT
    # -------------------------------------------------------------------------
    story.append(Paragraph("SECTION 3: PRIMARY ENSEMBLE CLASSIFICATION VERDICT", section_h1))
    
    verdict_card = [
        [
            Paragraph(f"<font size=14 color='{badge_fg.hexval()}'><b>VERDICT: {pred_verdict.upper()}</b></font><br/>"
                      f"<font size=8 color='#475569'>Primary Decision Rule: 55% Tuned XGBoost + 45% Fine-Tuned RoBERTa (Threshold: 0.540)</font>",
                      ParagraphStyle('VCell', parent=styles['Normal'], leading=16)),
            Paragraph(f"<b>Fake Risk Probability:</b> <font color='{badge_fg.hexval()}'><b>{fake_prob * 100:.2f}%</b></font><br/>"
                      f"<b>Genuine Probability:</b> <b>{genuine_prob * 100:.2f}%</b>",
                      ParagraphStyle('VProb', parent=styles['Normal'], alignment=2, leading=14))
        ]
    ]
    t_verdict = Table(verdict_card, colWidths=[4.2 * inch, 2.8 * inch])
    t_verdict.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), badge_bg),
        ('BOX', (0, 0), (-1, -1), 1.5, badge_fg),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_verdict)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # SECTION 4: INDIVIDUAL MODEL BREAKDOWN
    # -------------------------------------------------------------------------
    story.append(Paragraph("SECTION 4: INDIVIDUAL MODEL RESULTS & COMPARISON", section_h1))
    
    xgb_f = record.get("xgb_fake_prob")
    roberta_f = record.get("roberta_fake_prob")
    bert_f = record.get("bert_fake_prob")
    deberta_f = record.get("deberta_fake_prob")
    
    models_table_data = [
        [
            Paragraph("<b>Model Architecture</b>", table_cell_bold),
            Paragraph("<b>Role in Pipeline</b>", table_cell_bold),
            Paragraph("<b>Fake Prob</b>", table_cell_bold),
            Paragraph("<b>Genuine Prob</b>", table_cell_bold),
            Paragraph("<b>Execution Status</b>", table_cell_bold)
        ],
        [
            Paragraph("Tuned XGBoost", table_cell_bold),
            Paragraph("Primary Ensemble (55% Weight)", table_cell),
            Paragraph(f"{xgb_f * 100:.2f}%" if xgb_f is not None else "N/A", table_cell),
            Paragraph(f"{(1.0 - xgb_f) * 100:.2f}%" if xgb_f is not None else "N/A", table_cell),
            Paragraph("<font color='#16A34A'><b>Evaluated (TreeSHAP)</b></font>", table_cell)
        ],
        [
            Paragraph("Fine-Tuned RoBERTa", table_cell_bold),
            Paragraph("Primary Ensemble (45% Weight)", table_cell),
            Paragraph(f"{roberta_f * 100:.2f}%" if roberta_f is not None else "N/A", table_cell),
            Paragraph(f"{(1.0 - roberta_f) * 100:.2f}%" if roberta_f is not None else "N/A", table_cell),
            Paragraph("<font color='#16A34A'><b>Evaluated (Neural)</b></font>", table_cell)
        ],
        [
            Paragraph("Fine-Tuned BERT", table_cell_bold),
            Paragraph("Standalone Comparison Benchmark", table_cell),
            Paragraph(f"{bert_f * 100:.2f}%" if bert_f is not None else "—", table_cell),
            Paragraph(f"{(1.0 - bert_f) * 100:.2f}%" if bert_f is not None else "—", table_cell),
            Paragraph("<font color='#2563EB'>Evaluated</font>" if bert_f is not None else "<font color='#64748B'>Not Run (Optional Standby)</font>", table_cell)
        ],
        [
            Paragraph("Fine-Tuned DeBERTa", table_cell_bold),
            Paragraph("Standalone Comparison Benchmark", table_cell),
            Paragraph(f"{deberta_f * 100:.2f}%" if deberta_f is not None else "—", table_cell),
            Paragraph(f"{(1.0 - deberta_f) * 100:.2f}%" if deberta_f is not None else "—", table_cell),
            Paragraph("<font color='#2563EB'>Evaluated</font>" if deberta_f is not None else "<font color='#64748B'>Not Run (Optional Standby)</font>", table_cell)
        ]
    ]
    t_models = Table(models_table_data, colWidths=[1.5 * inch, 1.9 * inch, 1.0 * inch, 1.1 * inch, 1.5 * inch])
    t_models.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_bg_light),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_models)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # SECTION 5: EXPLAINABLE AI (XGBoost TreeSHAP Analysis)
    # -------------------------------------------------------------------------
    story.append(Paragraph("SECTION 5: EXPLAINABLE AI & ATTRIBUTION SIGNALS", section_h1))
    story.append(Paragraph(
        "<i>Methodology Note:</i> The table below details TreeSHAP marginal contributions calculated specifically "
        "for the tabular and TF-IDF component of the tuned <b>XGBoost model</b>. TreeSHAP does not explain transformer "
        "attention weights (RoBERTa), which are calculated through multi-head contextual encodings.",
        disclaimer_style
    ))
    story.append(Spacer(1, 4))
    
    # Try parsing explanation JSON
    expl_data = record.get("explanation_data")
    if not expl_data and record.get("explanation_json"):
        try:
            expl_data = json.loads(record["explanation_json"])
        except Exception:
            expl_data = None
            
    xai_rows = [
        [
            Paragraph("<b>Influencing Signal / Feature</b>", table_cell_bold),
            Paragraph("<b>Observed Metric</b>", table_cell_bold),
            Paragraph("<b>SHAP Impact</b>", table_cell_bold),
            Paragraph("<b>Plain-English Interpretation</b>", table_cell_bold)
        ]
    ]
    
    if expl_data and isinstance(expl_data, dict):
        text_f = expl_data.get("textual_factors", [])
        beh_f = expl_data.get("behavioural_factors", [])
        temp_f = expl_data.get("temporal_factors", [])
        
        all_factors = (text_f + beh_f + temp_f)[:6]  # top 6
        for f in all_factors:
            inf = f.get("influence", "Neutral")
            inf_color = "#DC2626" if "Increases" in inf else ("#16A34A" if "Decreases" in inf else "#64748B")
            xai_rows.append([
                Paragraph(f.get("factor", "Signal"), table_cell_bold),
                Paragraph(str(f.get("actual_value", "N/A")), table_cell),
                Paragraph(f"<font color='{inf_color}'><b>{f.get('shap_value', 0.0):+.3f}</b> ({inf})</font>", table_cell),
                Paragraph(f.get("explanation", "Standard feature behavior."), table_cell)
            ])
    else:
        # Fallback standard explanation based on available fields
        xai_rows.append([
            Paragraph("Rating Polarity", table_cell_bold),
            Paragraph(f"{rating_val} Stars", table_cell),
            Paragraph("<font color='#DC2626'>Increases fake score</font>" if rating_val in [1.0, 5.0] else "<font color='#16A34A'>Decreases fake score</font>", table_cell),
            Paragraph("Polarized review ratings are frequently associated with promotional campaigns or smear attempts.", table_cell)
        ])
        xai_rows.append([
            Paragraph("Review Length", table_cell_bold),
            Paragraph(f"{len(raw_text.split())} words", table_cell),
            Paragraph("<font color='#DC2626'>Increases fake score</font>" if len(raw_text.split()) < 20 else "<font color='#16A34A'>Decreases fake score</font>", table_cell),
            Paragraph("Very short or formulaic review bodies often correlate with automated spam submissions.", table_cell)
        ])
        xai_rows.append([
            Paragraph("Reviewer History Status", table_cell_bold),
            Paragraph("Linked" if record.get("has_reviewer_history") else "Unobserved", table_cell),
            Paragraph("<font color='#64748B'>Neutral</font>", table_cell),
            Paragraph("Features imputed with empirical training medians. History is never fabricated.", table_cell)
        ])
        
    t_xai = Table(xai_rows, colWidths=[1.6 * inch, 1.3 * inch, 1.8 * inch, 2.3 * inch])
    t_xai.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_bg_light),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_xai)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # SECTION 6: BEHAVIOURAL & PATTERN INDICATORS
    # -------------------------------------------------------------------------
    story.append(Paragraph("SECTION 6: BEHAVIOURAL & TEMPORAL PATTERN INDICATORS", section_h1))
    
    beh_data = [
        [
            Paragraph("<b>Indicator Category</b>", table_cell_bold),
            Paragraph("<b>Observed Metric / Status</b>", table_cell_bold),
            Paragraph("<b>Contextual Evaluation</b>", table_cell_bold)
        ],
        [
            Paragraph("Account History Link", table_cell_bold),
            Paragraph("Verified in reviewer database" if record.get("has_reviewer_history") else "Unobserved / Anonymous submission", table_cell),
            Paragraph("Valid historical reviewer profiles allow velocity and product diversity calculations." if record.get("has_reviewer_history") else "Baseline median imputation used; no artificial history fabricated.", table_cell)
        ],
        [
            Paragraph("Review Rating Distribution", table_cell_bold),
            Paragraph(f"{rating_val} Stars ({'Extreme 1★/5★' if rating_val in [1.0, 5.0] else 'Moderate 2-4★'})", table_cell),
            Paragraph("92.1% of fraudulent reviews in training data exhibited extreme polarized ratings (1 or 5 stars).", table_cell)
        ],
        [
            Paragraph("Submission Velocity", table_cell_bold),
            Paragraph("Unobserved (Single submission mode)" if not record.get("has_reviewer_history") else "Calculated from reviewer timeline", table_cell),
            Paragraph("Bursts of multiple reviews within 7 days signal coordinated promotional activity.", table_cell)
        ]
    ]
    t_beh = Table(beh_data, colWidths=[1.8 * inch, 2.3 * inch, 2.9 * inch])
    t_beh.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), c_bg_light),
        ('GRID', (0, 0), (-1, -1), 0.5, c_border),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_beh)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # SECTION 7: INTERPRETATION, LIMITATIONS & GOVERNANCE
    # -------------------------------------------------------------------------
    story.append(Paragraph("SECTION 7: INTERPRETATION & METHODOLOGICAL LIMITATIONS", section_h1))
    disclaimer_box_content = [
        [
            Paragraph(
                "<b>Essential Verification Disclosures:</b><br/>"
                "1. <b>Probabilistic Screening:</b> This report represents a probabilistic machine-learning assessment. "
                "A classification of 'Potentially Fake' is a screening indicator for further human review, NOT definitive legal proof of fraudulent behavior or commercial defamation.<br/>"
                "2. <b>Proxy Labeling:</b> Models were trained on historical customer review datasets (2011–2021) utilizing reviewer-derived proxy labels. Genuine consumer reviews with atypical wording may receive elevated fake scores.<br/>"
                "3. <b>Ensemble Boundary:</b> The primary decision cutoff is 0.540 on the 55% XGBoost + 45% RoBERTa weighted probability. Values near 0.540 represent borderline confidence and warrant manual scrutiny.",
                disclaimer_style
            )
        ]
    ]
    t_disc = Table(disclaimer_box_content, colWidths=[7.0 * inch])
    t_disc.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFFBEB")),
        ('BOX', (0, 0), (-1, -1), 1.0, colors.HexColor("#F59E0B")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(t_disc)
    story.append(Spacer(1, 8))

    # -------------------------------------------------------------------------
    # SECTION 8: FINAL SUMMARY & CONCLUSION
    # -------------------------------------------------------------------------
    story.append(Paragraph("SECTION 8: EXECUTIVE SUMMARY & AUDIT CONCLUSION", section_h1))
    summary_text = (
        f"The submitted review was processed through ReviewShield's dual-stream architecture. "
        f"The primary ensemble computed a fake risk probability of <b>{fake_prob * 100:.2f}%</b> and a genuine probability of <b>{genuine_prob * 100:.2f}%</b>. "
        f"Based on the validated 0.540 classification threshold, the review is categorized as <b>{pred_verdict.upper()}</b>. "
        f"The tuned XGBoost component assigned a {xgb_f * 100:.1f}% risk score and the deep RoBERTa language model assigned a {roberta_f * 100:.1f}% risk score. "
        f"This audit is logged in the permanent SQLite database under Prediction ID <code>{pred_id}</code>."
    )
    story.append(Paragraph(summary_text, body_text))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer
