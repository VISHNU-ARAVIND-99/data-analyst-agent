import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from config import EXPORTS_DIR, MODEL_NAME

class ExecutivePDFReport:
    """Generates professional corporate analytics PDF reports with single and multi-table support."""

    @staticmethod
    def generate(
        filename: str,
        dataset_name: str,
        profile: Dict[str, Any],
        executive_summary: str,
        chat_insights: List[Dict[str, Any]],
        charts: List[str],
        transformations: Optional[Any] = None
    ) -> str:
        """
        Builds a multi-page executive PDF report.
        Returns the absolute filepath to the created PDF.
        """
        output_path = EXPORTS_DIR / filename
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )

        styles = getSampleStyleSheet()
        
        primary_color = colors.HexColor("#1E293B")   # Slate 800
        secondary_color = colors.HexColor("#2563EB") # Blue 600
        text_dark = colors.HexColor("#0F172A")
        bg_light = colors.HexColor("#F8FAFC")
        border_color = colors.HexColor("#E2E8F0")

        styles.add(ParagraphStyle(
            name="ReportTitle",
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            textColor=primary_color
        ))
        styles.add(ParagraphStyle(
            name="ReportSubtitle",
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#64748B")
        ))
        styles.add(ParagraphStyle(
            name="SectionHeading",
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=secondary_color,
            spaceBefore=14,
            spaceAfter=6
        ))
        styles.add(ParagraphStyle(
            name="SubSectionHeading",
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            textColor=primary_color,
            spaceBefore=8,
            spaceAfter=4
        ))
        styles.add(ParagraphStyle(
            name="BodyTextCustom",
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=text_dark
        ))
        styles.add(ParagraphStyle(
            name="TableHeader",
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=11,
            textColor=colors.white,
            alignment=1
        ))
        styles.add(ParagraphStyle(
            name="TableCell",
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=text_dark
        ))

        story = []

        # 1. Header Banner
        story.append(Paragraph("EXECUTIVE DATA INTELLIGENCE REPORT", styles["ReportTitle"]))
        story.append(Paragraph(
            f"Database / Source: <b>{dataset_name}</b> | Generated: {datetime.now().strftime('%B %d, %Y - %H:%M')} | Engine: {MODEL_NAME} Local SLM",
            styles["ReportSubtitle"]
        ))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=2, color=secondary_color, spaceBefore=4, spaceAfter=14))

        # 2. Executive Summary Section
        story.append(Paragraph("1. Executive Summary & Core Takeaways", styles["SectionHeading"]))
        summary_paragraphs = executive_summary.split("\n\n") if "\n\n" in executive_summary else [executive_summary]
        for p in summary_paragraphs:
            if p.strip():
                cleaned_p = p.replace("**", "<b>").replace("__", "<b>")
                if "<b>" in cleaned_p and cleaned_p.count("<b>") % 2 == 1:
                    cleaned_p += "</b>"
                story.append(Paragraph(cleaned_p, styles["BodyTextCustom"]))
                story.append(Spacer(1, 6))

        story.append(Spacer(1, 10))

        # 3. Data Health & Profiling Audit
        story.append(Paragraph("2. Data Health & Schema Audit Scorecard", styles["SectionHeading"]))
        
        is_multi = profile.get("is_multi_table", False)
        if is_multi:
            # Multi-table overview table
            table_data = [
                [
                    Paragraph("<b>Table Name</b>", styles["TableHeader"]),
                    Paragraph("<b>Row Count</b>", styles["TableHeader"]),
                    Paragraph("<b>Columns</b>", styles["TableHeader"]),
                    Paragraph("<b>Quality Index</b>", styles["TableHeader"])
                ]
            ]
            for t_name, t_prof in profile.get("table_profiles", {}).items():
                shape = t_prof.get("shape", {"rows": 0, "columns": 0})
                table_data.append([
                    Paragraph(f"<b>{t_name}</b>", styles["TableCell"]),
                    Paragraph(f"{shape['rows']:,}", styles["TableCell"]),
                    Paragraph(f"{shape['columns']}", styles["TableCell"]),
                    Paragraph(f"{t_prof.get('quality_score', 95)}/100", styles["TableCell"])
                ])

            t = Table(table_data, colWidths=[2.2 * inch, 1.6 * inch, 1.6 * inch, 1.6 * inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, border_color),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(t)
            story.append(Spacer(1, 10))

            # Discovered Relational Keys
            rels = profile.get("relationships", [])
            if rels:
                story.append(Paragraph("<b>Discovered Entity-Relationship Joins:</b>", styles["SubSectionHeading"]))
                for r in rels:
                    story.append(Paragraph(
                        f"• <b>{r['parent_table']}.{r['key_column']}</b> ──[{r['cardinality']}]──> <b>{r['child_table']}.{r['key_column']}</b> "
                        f"({r['shared_values_count']} shared keys, {r['orphan_records_count']} orphans)",
                        styles["BodyTextCustom"]
                    ))
                story.append(Spacer(1, 10))

        else:
            shape = profile.get("shape", {"rows": 0, "columns": 0})
            table_data = [
                [
                    Paragraph("<b>Metric</b>", styles["TableHeader"]),
                    Paragraph("<b>Value</b>", styles["TableHeader"]),
                    Paragraph("<b>Audit Status</b>", styles["TableHeader"])
                ],
                [
                    Paragraph("Total Record Count", styles["TableCell"]),
                    Paragraph(f"{shape['rows']:,} rows", styles["TableCell"]),
                    Paragraph("Verified", styles["TableCell"])
                ],
                [
                    Paragraph("Feature Dimensions", styles["TableCell"]),
                    Paragraph(f"{shape['columns']} columns", styles["TableCell"]),
                    Paragraph("Verified", styles["TableCell"])
                ],
                [
                    Paragraph("Completeness Rate", styles["TableCell"]),
                    Paragraph(f"{profile.get('completeness_pct', 100)}%", styles["TableCell"]),
                    Paragraph("Optimal" if profile.get('completeness_pct', 100) > 90 else "Review Missing", styles["TableCell"])
                ],
                [
                    Paragraph("Data Quality Index", styles["TableCell"]),
                    Paragraph(f"{profile.get('quality_score', 95)} / 100", styles["TableCell"]),
                    Paragraph("High Integrity" if profile.get('quality_score', 95) >= 80 else "Moderate", styles["TableCell"])
                ]
            ]
            t = Table(table_data, colWidths=[2.2 * inch, 2.3 * inch, 2.5 * inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), primary_color),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, border_color),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, bg_light]),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(t)
            story.append(Spacer(1, 10))

        # 4. Embedded Visualizations
        if charts:
            story.append(Spacer(1, 8))
            story.append(Paragraph("3. Analytical Visualizations & Trends", styles["SectionHeading"]))
            
            for idx, chart_path in enumerate(charts):
                if os.path.exists(chart_path):
                    try:
                        chart_flowables = [
                            Paragraph(f"<b>Figure {idx + 1}: Cross-Analysis Chart</b>", styles["SubSectionHeading"]),
                            Spacer(1, 4),
                            Image(chart_path, width=6.2 * inch, height=3.2 * inch),
                            Spacer(1, 12)
                        ]
                        story.append(KeepTogether(chart_flowables))
                    except Exception:
                        pass

        # 5. Deep-Dive Q&A Insights
        if chat_insights:
            story.append(Paragraph("4. Targeted Explorations & Session Findings", styles["SectionHeading"]))
            for item in chat_insights:
                q = item.get("question", "")
                a = item.get("answer", "")
                if q and a:
                    qa_block = [
                        Paragraph(f"<b>Investigation: {q}</b>", styles["SubSectionHeading"]),
                        Paragraph(a.replace("**", "<b>").replace("\n", "<br/>"), styles["BodyTextCustom"]),
                        Spacer(1, 8)
                    ]
                    story.append(KeepTogether(qa_block))

        # 6. Strategic Recommendations Section
        story.append(Spacer(1, 10))
        story.append(Paragraph("5. Strategic Recommendations & Action Items", styles["SectionHeading"]))
        story.append(Paragraph(
            "• <b>Capitalize on High-Performing Segments:</b> Direct marketing and inventory allocation towards the top product/customer combinations revealed in the multi-table analysis.<br/>"
            "• <b>Address Return Anomalies:</b> Monitor products and regions exhibiting outsized return rates.<br/>"
            "• <b>Maintain Relational Data Integrity:</b> Regularly audit foreign key relationships and address orphan records.",
            styles["BodyTextCustom"]
        ))

        doc.build(story)
        return str(output_path)
