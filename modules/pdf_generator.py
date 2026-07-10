"""
pdf_generator.py
------------------
Builds a polished, professional PDF report summarizing a prescription
analysis session, using ReportLab's Platypus layout engine.

The report includes:
    - Branded header
    - Timestamp & session metadata
    - Side-by-side original vs. processed prescription images
    - Extracted medicines table
    - Analysis summary / KPIs
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import List, Optional

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PRIMARY_COLOR = colors.HexColor("#0E7C7B")
ACCENT_COLOR = colors.HexColor("#1A2B3C")
LIGHT_BG = colors.HexColor("#F4F8FA")


def _pil_to_rlimage(pil_image: Image.Image, max_width_cm: float = 8.0) -> RLImage:
    """Convert a PIL image into a ReportLab flowable, preserving aspect ratio."""
    buffer = io.BytesIO()
    rgb_image = pil_image.convert("RGB")
    rgb_image.save(buffer, format="JPEG", quality=88)
    buffer.seek(0)

    width, height = rgb_image.size
    max_width = max_width_cm * cm
    scale = max_width / width
    return RLImage(buffer, width=max_width, height=height * scale)


def generate_pdf_report(
    original_image: Image.Image,
    processed_image: Image.Image,
    medicines: List[dict],
    overall_confidence: int,
    prescription_quality: str,
    processing_time_s: float,
    doctor_notes: Optional[str] = None,
    patient_reference: Optional[str] = None,
    engine: Optional[str] = None,
    privacy_info: Optional[dict] = None,
) -> bytes:
    """
    Build the full PDF report and return it as raw bytes, ready to be
    served via st.download_button.

    `privacy_info`, if given, is a dict like:
        {"enabled": bool, "method": str|None, "fields_removed": List[str], "mode": str}
    used to render the Privacy Protection section.
    """
    engine_labels = {
        "tesseract": "Tesseract OCR (Offline)",
        "api": "Cloud API (direct extraction)",
        "hybrid-verified": "Tesseract + AI Verification (Hybrid)",
        "hybrid-ai-extract": "AI Vision Extraction (Hybrid fallback)",
    }
    engine_label = engine_labels.get(engine, engine or "Unknown")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        title="Prescription Analysis Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle", parent=styles["Title"], textColor=PRIMARY_COLOR, fontSize=20, spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle", parent=styles["Normal"], textColor=colors.grey, fontSize=10, spaceAfter=14
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"], textColor=ACCENT_COLOR, fontSize=13, spaceBefore=16, spaceAfter=8
    )
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10, leading=14)
    centered = ParagraphStyle("Centered", parent=styles["Normal"], alignment=TA_CENTER, fontSize=9, textColor=colors.grey)

    elements = []

    # --- Header -------------------------------------------------------- #
    elements.append(Paragraph("MediScan &mdash; Prescription Recognition System", title_style))
    elements.append(
        Paragraph(
            f"Medicine Extraction &amp; Analysis Report &mdash; Generated "
            f"{datetime.now().strftime('%d %B %Y, %I:%M %p')} &mdash; Engine: {engine_label}",
            subtitle_style,
        )
    )

    if patient_reference:
        elements.append(Paragraph(f"<b>Reference / Patient ID:</b> {patient_reference}", body_style))
        elements.append(Spacer(1, 8))

    # --- KPI summary row ------------------------------------------------ #
    kpi_data = [
        ["Medicines Detected", "Overall Confidence", "Prescription Quality", "Processing Time"],
        [
            str(len(medicines)),
            f"{overall_confidence}%",
            prescription_quality,
            f"{processing_time_s:.2f}s",
        ],
    ]
    kpi_table = Table(kpi_data, colWidths=[4 * cm] * 4)
    kpi_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_COLOR),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 1), (-1, 1), LIGHT_BG),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8E2E5")),
            ]
        )
    )
    elements.append(kpi_table)

    # --- Images side by side -------------------------------------------- #
    elements.append(Paragraph("Prescription Images", section_style))
    img_table = Table(
        [
            [_pil_to_rlimage(original_image, 7.5), _pil_to_rlimage(processed_image, 7.5)],
            [Paragraph("Original Upload", centered), Paragraph("Final Processed Image", centered)],
        ],
        colWidths=[8 * cm, 8 * cm],
    )
    img_table.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, 0), "MIDDLE"),
                ("TOPPADDING", (0, 1), (-1, 1), 4),
            ]
        )
    )
    elements.append(img_table)

    # --- Medicines table -------------------------------------------------- #
    elements.append(Paragraph("Extracted Medicines", section_style))
    if medicines:
        table_data = [["#", "Medicine Name", "Strength", "Frequency", "Confidence"]]
        for idx, med in enumerate(medicines, start=1):
            table_data.append(
                [
                    str(idx),
                    med.get("name", "-"),
                    med.get("strength") or "-",
                    med.get("frequency") or "-",
                    f"{med.get('confidence', 0)}%",
                ]
            )
        med_table = Table(table_data, colWidths=[1 * cm, 5.5 * cm, 3 * cm, 5 * cm, 2.5 * cm])
        med_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), ACCENT_COLOR),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8E2E5")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        elements.append(med_table)
    else:
        elements.append(Paragraph("No medicines were confidently identified in this prescription.", body_style))

    # --- Notes -------------------------------------------------------------- #
    if doctor_notes:
        elements.append(Paragraph("Additional Notes", section_style))
        elements.append(Paragraph(doctor_notes, body_style))

    # --- Privacy Protection -------------------------------------------------- #
    if privacy_info is not None:
        elements.append(Paragraph("Privacy Protection", section_style))
        if privacy_info.get("mode") == "not_applicable":
            elements.append(Paragraph(
                "Privacy Protection: N/A &mdash; Offline Tesseract mode was used, so no image data "
                "was ever transmitted anywhere.", body_style
            ))
        elif privacy_info.get("enabled"):
            fields = privacy_info.get("fields_removed") or []
            fields_html = "".join(f"&bull; {f}<br/>" for f in fields) if fields else "&bull; No PII detected on this image<br/>"
            elements.append(Paragraph("Privacy Protection: <b>&#10003; Enabled</b>", body_style))
            elements.append(Paragraph(f"Redaction Method: {privacy_info.get('method', '-')}", body_style))
            elements.append(Paragraph("Sensitive Fields Removed:<br/>" + fields_html, body_style))
            elements.append(Paragraph("Cloud AI Received: <b>Sanitized Image Only</b> &mdash; the original, "
                                       "unmodified image never left this device.", body_style))
            if privacy_info.get("low_confidence") and not privacy_info.get("manual_used"):
                elements.append(Paragraph(
                    "Caution: automatic detection read very little text from this image (common on "
                    "handwritten/degraded scans) and may not have located all sensitive information. "
                    "No manual region redaction was applied for this analysis.", body_style
                ))
            if privacy_info.get("overlaps_medicine"):
                elements.append(Paragraph(
                    "Caution: a manual/custom redaction region appeared to overlap text that looks like "
                    "a medicine or dosage entry. Manual regions are position-only and not content-aware, "
                    "so part of the clinical content may have been hidden from the AI as well.", body_style
                ))
            if privacy_info.get("high_coverage"):
                elements.append(Paragraph(
                    f"Caution: redaction covered approximately {privacy_info.get('coverage_pct', '?')}% of "
                    "this image. This is high enough that clinical content (medicine names, dosages) may "
                    "have been obscured along with patient information.", body_style
                ))
            if privacy_info.get("used_fallback_baseline"):
                elements.append(Paragraph(
                    "Note: no PII was detected by any active method, so a guaranteed top-10% fallback "
                    "header redaction was applied instead of sending the image completely unredacted.",
                    body_style
                ))
        else:
            elements.append(Paragraph("Privacy Protection: &#10007; Disabled", body_style))
            elements.append(Paragraph("Cloud AI Received: Original (unredacted) image.", body_style))

    # --- Disclaimer footer --------------------------------------------------- #
    elements.append(Spacer(1, 24))
    elements.append(
        Paragraph(
            "Disclaimer: This report was generated automatically by MediScan and is intended "
            "for informational / academic demonstration purposes only. Always verify medicine "
            "names, strengths, and dosing with a licensed pharmacist or physician before "
            "dispensing or consuming any medication.",
            centered,
        )
    )

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
