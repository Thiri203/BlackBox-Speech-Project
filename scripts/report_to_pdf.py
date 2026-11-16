"""Turn a Markdown report into an A4 portrait PDF (plain look) with headless Chrome or Edge.

    python scripts/report_to_pdf.py reports/training_all/report.md            # -> report.pdf next to it
    python scripts/report_to_pdf.py reports/training_all/report.md --check    # also render pages to PNG

Text is converted as written (python-markdown, "tables" extension); images are referenced relative to
the .md file and shown at full text width. Tables with many columns get a smaller font so they fit.
Each "### " section starts on a new page so an image and its table stay together.
--check writes page images to <pdf name>_pages/ (not committed) for a visual check.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "chrome", "msedge", "chromium", "google-chrome",
]

CSS = """
@page { size: A4 portrait; margin: 18mm 16mm 18mm 16mm; }
html { background: #fff; }
body { font-family: "Segoe UI", Arial, Helvetica, sans-serif; font-size: 10pt; line-height: 1.45;
       color: #000; background: #fff; margin: 0; }
h1 { font-size: 17pt; margin: 0 0 10pt; line-height: 1.25; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; padding-bottom: 2pt; border-bottom: 0.6pt solid #000;
     break-after: avoid; }
h3 { font-size: 11.5pt; margin: 0 0 6pt; break-before: page; break-after: avoid; }
h2.newpage { break-before: page; margin-top: 0; }
h2.newpage + p + h3 { break-before: auto; margin-top: 12pt; }
p, li { orphans: 3; widows: 3; }
ul { padding-left: 16pt; }
code { font-family: Consolas, "Courier New", monospace; font-size: 9pt; }
img { display: block; width: 100%; height: auto; margin: 4pt 0 10pt; break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 12pt; break-inside: auto; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th, td { border: 0.5pt solid #000; padding: 2.5pt 4pt; text-align: left; vertical-align: top;
         overflow-wrap: anywhere; }
th { font-weight: 600; }
table.wide { font-size: 8pt; }
table.wide th, table.wide td { padding: 2pt 3pt; }
table.xwide { font-size: 7.2pt; }
table.xwide th, table.xwide td { padding: 1.8pt 2.5pt; }
"""


def find_browser() -> str:
    for b in BROWSERS:
        p = shutil.which(b) or (b if Path(b).exists() else None)
        if p:
            return p
    sys.exit("no Chrome / Edge found for printing to PDF")


def to_html(md_path: Path) -> str:
    body = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables"])

    def tag_table(m: re.Match) -> str:  # smaller font for tables with many columns
        n_cols = m.group(0).split("</tr>", 1)[0].count("<th")
        cls = "xwide" if n_cols >= 9 else "wide" if n_cols >= 6 else ""
        return f'<table class="{cls}">' if cls else "<table>"

    body = re.sub(r"<table>.*?</tr>", lambda m: tag_table(m) + m.group(0)[len("<table>"):], body, flags=re.S)
    # A level-2 section made of "### " subsections (e.g. one per fold) starts on a new page.
    body = re.sub(r"<h2>(?=(?:(?!<h2>).)*?<h3>)", '<h2 class="newpage">', body, flags=re.S)
    base = md_path.parent.resolve().as_uri() + "/"
    title = md_path.stem
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><base href="{base}">'
            f"<title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>")


def print_pdf(html: str, pdf: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "report.html"
        src.write_text(html, encoding="utf-8")
        cmd = [find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
               "--allow-file-access-from-files", f"--user-data-dir={Path(tmp) / 'profile'}",
               f"--print-to-pdf={pdf.resolve()}", src.as_uri()]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    if not pdf.exists() or pdf.stat().st_size == 0:
        sys.exit("printing failed: no PDF written")


def render_pages(pdf: Path, scale: float = 1.4) -> list[Path]:
    import pypdfium2 as pdfium

    out = pdf.with_name(pdf.stem + "_pages")
    out.mkdir(exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf))
    paths = []
    for i, page in enumerate(doc, 1):
        p = out / f"page{i:02d}.png"
        page.render(scale=scale).to_pil().save(p)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("md", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    pdf = a.out or a.md.with_suffix(".pdf")
    print_pdf(to_html(a.md), pdf)
    print(f"wrote {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")
    if a.check:
        for p in render_pages(pdf):
            print(f"  page image {p}")


if __name__ == "__main__":
    main()
