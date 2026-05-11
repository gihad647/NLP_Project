"""
Convert TECHNICAL_REPORT.md to TECHNICAL_REPORT.pdf using reportlab.
Usage:  python scripts/generate_report_pdf.py
"""

import re
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted,
    Table, TableStyle, HRFlowable, KeepTogether,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

ROOT = Path(__file__).parent.parent
SRC  = ROOT / "TECHNICAL_REPORT.md"
DST  = ROOT / "TECHNICAL_REPORT.pdf"

# ── Styles ────────────────────────────────────────────────────────────────────
base_styles = getSampleStyleSheet()

H1 = ParagraphStyle("H1", parent=base_styles["Heading1"],
                    fontSize=20, spaceAfter=10, spaceBefore=16,
                    textColor=colors.HexColor("#1a1a2e"))
H2 = ParagraphStyle("H2", parent=base_styles["Heading2"],
                    fontSize=15, spaceAfter=6, spaceBefore=14,
                    textColor=colors.HexColor("#16213e"),
                    borderPad=2)
H3 = ParagraphStyle("H3", parent=base_styles["Heading3"],
                    fontSize=12, spaceAfter=4, spaceBefore=10,
                    textColor=colors.HexColor("#0f3460"))
BODY = ParagraphStyle("Body", parent=base_styles["Normal"],
                      fontSize=10, leading=14, spaceAfter=4)
BULLET = ParagraphStyle("Bullet", parent=BODY,
                        leftIndent=18, bulletIndent=6, spaceAfter=2)
CODE = ParagraphStyle("Code", fontName="Courier", fontSize=8,
                      leading=11, leftIndent=12, backColor=colors.HexColor("#f4f4f4"),
                      borderColor=colors.HexColor("#cccccc"), borderWidth=0.5,
                      borderPad=4, spaceAfter=6, spaceBefore=4)
CAPTION = ParagraphStyle("Caption", parent=BODY, fontSize=8,
                         textColor=colors.grey, alignment=TA_CENTER)


def _escape(text: str) -> str:
    """Escape XML special chars for Paragraph (reportlab uses XML parser)."""
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return text


def _inline(text: str) -> str:
    """Apply inline markdown: **bold**, `code`."""
    text = _escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'`([^`]+)`',
                  lambda m: f'<font name="Courier" size="9">{m.group(1)}</font>',
                  text)
    return text


# ── Table renderer ────────────────────────────────────────────────────────────
def _parse_table(lines: list[str]):
    rows = []
    for line in lines:
        if re.match(r'^\s*\|[-: |]+\|\s*$', line):
            continue  # separator row
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return None

    col_count = max(len(r) for r in rows)
    data = []
    for i, row in enumerate(rows):
        # pad short rows
        row = row + [""] * (col_count - len(row))
        if i == 0:
            data.append([Paragraph(f"<b>{_inline(c)}</b>", BODY) for c in row])
        else:
            data.append([Paragraph(_inline(c), BODY) for c in row])

    col_width = (A4[0] - 4 * cm) / col_count
    tbl = Table(data, colWidths=[col_width] * col_count, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  colors.HexColor("#e8eaf6")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  colors.HexColor("#1a1a2e")),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f9f9f9")]),
        ("VALIGN",       (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    return tbl


# ── Main converter ────────────────────────────────────────────────────────────
def convert(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    lines = text.splitlines()

    doc = SimpleDocTemplate(
        str(dst), pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2*cm,
    )

    story = []
    i = 0
    n = len(lines)

    def flush_paragraph(buf: list[str]):
        joined = " ".join(buf).strip()
        if joined:
            story.append(Paragraph(_inline(joined), BODY))
            story.append(Spacer(1, 2))

    para_buf: list[str] = []

    while i < n:
        line = lines[i]

        # ── Fenced code block ──────────────────────────────────────────────
        if line.strip().startswith("```"):
            flush_paragraph(para_buf); para_buf = []
            code_lines = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            code_text = "\n".join(code_lines)
            story.append(Preformatted(code_text, CODE))
            i += 1
            continue

        # ── Heading ───────────────────────────────────────────────────────
        m = re.match(r'^(#{1,3})\s+(.*)', line)
        if m:
            flush_paragraph(para_buf); para_buf = []
            level = len(m.group(1))
            heading_text = _inline(m.group(2))
            style = [H1, H2, H3][level - 1]
            story.append(Paragraph(heading_text, style))
            i += 1
            continue

        # ── Horizontal rule ───────────────────────────────────────────────
        if re.match(r'^---+\s*$', line):
            flush_paragraph(para_buf); para_buf = []
            story.append(HRFlowable(width="100%", thickness=0.5,
                                    color=colors.HexColor("#cccccc"),
                                    spaceAfter=6, spaceBefore=6))
            i += 1
            continue

        # ── Table block ───────────────────────────────────────────────────
        if line.strip().startswith("|"):
            flush_paragraph(para_buf); para_buf = []
            table_lines = []
            while i < n and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            tbl = _parse_table(table_lines)
            if tbl:
                story.append(tbl)
                story.append(Spacer(1, 6))
            continue

        # ── Bullet list item ──────────────────────────────────────────────
        m = re.match(r'^(\s*)[*\-]\s+(.*)', line)
        if m:
            flush_paragraph(para_buf); para_buf = []
            indent = len(m.group(1))
            bullet_style = ParagraphStyle(
                f"Bullet{indent}", parent=BULLET,
                leftIndent=18 + indent * 12, bulletIndent=6 + indent * 12,
            )
            story.append(Paragraph(f"•&nbsp;{_inline(m.group(2))}", bullet_style))
            i += 1
            continue

        # ── Blank line ────────────────────────────────────────────────────
        if not line.strip():
            flush_paragraph(para_buf); para_buf = []
            story.append(Spacer(1, 4))
            i += 1
            continue

        # ── Regular text ──────────────────────────────────────────────────
        para_buf.append(line.strip())
        i += 1

    flush_paragraph(para_buf)

    doc.build(story)
    print(f"PDF written to: {dst}")


if __name__ == "__main__":
    convert(SRC, DST)
