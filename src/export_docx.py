"""
Exports report/report_draft.md to report/report_draft.docx (Word), so the
team can edit and submit it. Numbers are not touched: the .docx is a direct
conversion of the Markdown produced by stage7_report.py.

Supported Markdown (all that the template uses):
  # / ## / ### headings, paragraphs, **bold**, *italic*, `code`,
  "- " bullets (two levels), "1. " numbered items, "> " notes,
  | tables |, and ![caption](path) figures.

Run (after stage7_report.py):  .venv/bin/python src/export_docx.py
"""
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "report" / "report_draft.md", ROOT / "report" / "report_draft.docx"
INLINE = re.compile(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)")


def add_runs(par, text, bold=False, italic=False):
    """Write text with **bold**, *italic* and `code` into a paragraph."""
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**"):
            r = par.add_run(part[2:-2])
            r.bold = True
        elif part.startswith("`"):
            r = par.add_run(part[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(9.5)
        elif part.startswith("*"):
            r = par.add_run(part[1:-1])
            r.italic = True
        else:
            r = par.add_run(part)
        r.bold = r.bold or bold
        r.italic = r.italic or italic
    return par


def shade(par, hex_fill="FFF4CE"):
    """Light background for the 'notes for the team' box."""
    ppr = par._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill)
    ppr.append(shd)


def add_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], [r for r in cells[1:] if not set("".join(r)) <= set("-: ")]
    t = doc.add_table(rows=1 + len(body), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate([header] + body):
        for j, val in enumerate(row):
            p = t.cell(i, j).paragraphs[0]
            add_runs(p, val, bold=(i == 0))
            for r in p.runs:
                r.font.size = Pt(9)
    doc.add_paragraph()


def convert(src=SRC, dst=DST):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    for side in ("left_margin", "right_margin", "top_margin", "bottom_margin"):
        setattr(sec, side, Cm(2.3))
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Calibri", Pt(11)

    lines = src.read_text().splitlines()
    i, n_fig = 0, 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if s.startswith("|"):                                   # table block
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            add_table(doc, block)
            continue
        m = re.fullmatch(r"!\[(.*)\]\((.*)\)", s)
        if m:                                                   # figure + caption
            path = (src.parent / m.group(2)).resolve()
            doc.add_picture(str(path), width=Cm(16))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = add_runs(doc.add_paragraph(style="Caption"), m.group(1))
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            n_fig += 1
        elif s.startswith("#"):
            level = len(s) - len(s.lstrip("#"))
            text = s[level:].strip()
            if level == 1:
                doc.add_heading(text, level=0)
            else:
                add_runs(doc.add_heading(level=level - 1), text)
        elif s.startswith(">"):                                 # note box
            body = s.lstrip(">").strip()
            p = add_runs(doc.add_paragraph(), body)
            p.paragraph_format.left_indent = Cm(0.5)
            shade(p)
        elif re.match(r"\d+\. ", s):                            # numbered item (number kept as text)
            num, body = s.split(" ", 1)
            p = doc.add_paragraph()
            p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Cm(0.9), Cm(-0.6)
            p.add_run(num + "\t")
            add_runs(p, body)
        elif s.startswith("- "):                                # bullets, two levels
            indent = len(line) - len(line.lstrip())
            style = "List Bullet 2" if indent >= 2 else "List Bullet"
            add_runs(doc.add_paragraph(style=style), s[2:])
        else:
            add_runs(doc.add_paragraph(), s)
        i += 1

    for h in ("Heading 1", "Heading 2"):
        doc.styles[h].font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
    doc.save(dst)
    return n_fig


def main():
    n_fig = convert()
    print(f"wrote {DST.relative_to(ROOT)} ({n_fig} figures)")


if __name__ == "__main__":
    main()
