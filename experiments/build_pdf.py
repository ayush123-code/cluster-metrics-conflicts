"""Build the submission PDF from results/report.md plus a source-code appendix.

Usage:  python experiments/build_pdf.py   (run experiments/run_all.py first)

The HTML is printed to PDF with headless Microsoft Edge or Google Chrome.
"""

from __future__ import annotations

import datetime as dt
import html
import io
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

import markdown
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
HTML_OUT = RESULTS / "report.html"
PDF_OUT = ROOT / "Clustering_Evaluation_Metrics_Report.pdf"

TITLE = "When Clustering Metrics Disagree"
SUBTITLE = (
    "A contingency-matrix study of purity, maximum-matching accuracy, precision, recall "
    "and F-measure under fragmentation, merging and class imbalance"
)
FIELDS = ["Name", "Roll / Student ID", "Course", "Instructor", "Date of submission"]
SOURCE_FILES = [
    "clustermetrics/__init__.py",
    "clustermetrics/contingency.py",
    "clustermetrics/perturb.py",
    "clustermetrics/ranking.py",
    "experiments/run_all.py",
    "experiments/build_pdf.py",
    "tests/test_metrics.py",
]
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge",
    "google-chrome",
    "chromium",
]

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
* { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Segoe UI", Calibri, Arial, sans-serif; font-size: 10.2pt;
       line-height: 1.45; color: #1d1d1f; }
h1, h2, h3 { color: #1f3a5f; line-height: 1.25; }
h2 { font-size: 16pt; border-bottom: 2px solid #1f3a5f; padding-bottom: 3px;
     break-before: page; margin-top: 0; }
h3 { font-size: 12pt; margin-top: 18px; break-after: avoid; }
p, li { text-align: justify; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.8pt;
       background: #f2f4f7; padding: 0 2px; border-radius: 2px; }
table { border-collapse: collapse; margin: 8px 0 12px; font-size: 7.8pt; width: 100%; }
th, td { border: 1px solid #c7cdd6; padding: 2.5px 5px; text-align: left; vertical-align: top; }
th { background: #e6ecf4; }
tr { break-inside: avoid; }
tr:nth-child(even) td { background: #f8f9fb; }
img { max-width: 100%; display: block; margin: 10px auto; break-inside: avoid; }
em { color: #333; }

.title-page { height: 255mm; display: flex; flex-direction: column; justify-content: center;
              text-align: center; break-after: page; }
.title-page h1 { font-size: 28pt; margin-bottom: 8px; border: none; }
.title-page .subtitle { font-size: 13pt; color: #444; max-width: 150mm; margin: 0 auto 26px;
                        text-align: center; }
.title-page .question { font-size: 11.5pt; margin: 0 auto 50px; max-width: 150mm;
                        padding: 10px 14px; border-left: 4px solid #1f3a5f;
                        background: #f2f5fa; text-align: left; }
.fields { width: 125mm; margin: 0 auto; border: none; font-size: 11.5pt; }
.fields td { border: none; background: none !important; padding: 9px 4px; }
.fields td.label { width: 45mm; font-weight: 600; color: #1f3a5f; }
.fields td.blank { border-bottom: 1px solid #555; }

h2#introduction-and-method, h2#references { break-before: auto; margin-top: 26px; }

.toc-page { break-after: page; }
.toc-page h2 { break-before: auto; }
.toc ul { list-style: none; padding-left: 14px; }
.toc > ul { padding-left: 0; }
.toc-page li { text-align: left; margin: 3px 0; }
.toc-page a { color: #1d1d1f; text-decoration: none; }

.appendix h3 { font-family: Consolas, monospace; font-size: 10.5pt; background: #1f3a5f;
               color: white; padding: 4px 8px; margin: 18px 0 0; break-after: avoid; }
.appendix .highlight pre { font-size: 7.3pt; line-height: 1.32; margin: 0 0 10px;
               padding: 6px 8px; border: 1px solid #d0d5dd; background: #fbfbfc;
               white-space: pre-wrap; word-break: break-word; }
.appendix .highlight .linenos { color: #9aa3ad; padding-right: 6px; }
pre.output { font-size: 8pt; background: #f2f4f7; padding: 8px; border: 1px solid #d0d5dd;
             white-space: pre-wrap; }
"""


def title_page() -> str:
    rows = "\n".join(
        f'<tr><td class="label">{f}</td><td class="blank">&nbsp;</td></tr>' for f in FIELDS
    )
    return f"""
<section class="title-page">
  <h1>{TITLE}</h1>
  <p class="subtitle">{SUBTITLE}</p>
  <p class="question"><strong>Research question.</strong> How can evaluation metrics produce
  conflicting conclusions about clustering quality?</p>
  <table class="fields">{rows}</table>
</section>"""


def report_body() -> tuple[str, str]:
    text = (RESULTS / "report.md").read_text(encoding="utf-8")
    lines = text.splitlines()
    # The title page already carries the H1 title and the research question.
    body_start = next(i for i, ln in enumerate(lines) if ln.startswith("## "))
    body = "\n".join(lines[body_start:])
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "toc", "attr_list"],
        extension_configs={"toc": {"toc_depth": "2-3"}},
    )
    return md.convert(body), md.toc


def run_tests() -> str:
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    status = "OK" if result.wasSuccessful() else "FAILED"
    return f"{stream.getvalue().strip()}\n\nStatus: {status}"


def appendix() -> str:
    formatter = HtmlFormatter(linenos="inline", style="friendly")
    parts = [
        '<section class="appendix">',
        "<h2 id='appendix-a'>Appendix A. Reproducing the results</h2>",
        "<p>All numbers, tables and figures in this document are produced by the code in "
        "Appendix B. From the project root:</p>",
        "<pre class='output'>pip install -r requirements.txt\n"
        "python -m unittest discover -s tests -t .\n"
        "python experiments/run_all.py\n"
        "python experiments/build_pdf.py</pre>",
        "<p>The unit tests check every metric against hand-computed textbook values "
        "(Manning, Raghavan &amp; Sch&uuml;tze, Fig. 16.4) and against scikit-learn's "
        "ARI, NMI, Fowlkes&ndash;Mallows and Rand index. Output of the test run used for "
        "this document:</p>",
        f"<pre class='output'>{html.escape(run_tests())}</pre>",
        "<h2 id='appendix-b'>Appendix B. Source code</h2>",
    ]
    for rel in SOURCE_FILES:
        code = (ROOT / rel).read_text(encoding="utf-8")
        parts.append(f"<h3>{rel}</h3>")
        parts.append(highlight(code, PythonLexer(), formatter))
    parts.append("</section>")
    return "\n".join(parts), formatter.get_style_defs(".highlight")


def build_html() -> Path:
    body, toc = report_body()
    appendix_html, pygments_css = appendix()
    toc_html = (
        '<section class="toc-page"><h2>Contents</h2>'
        f'<div class="toc">{toc}</div>'
        "<ul style='list-style:none;padding-left:0'>"
        "<li><a href='#appendix-a'>Appendix A. Reproducing the results</a></li>"
        "<li><a href='#appendix-b'>Appendix B. Source code</a></li></ul>"
        "</section>"
    )
    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>{TITLE}</title>
<style>{CSS}\n{pygments_css}</style></head>
<body>
{title_page()}
{toc_html}
{body}
{appendix_html}
<p style="margin-top:20px;color:#888;font-size:8pt">Generated {dt.date.today():%d %B %Y}.</p>
</body></html>"""
    HTML_OUT.write_text(doc, encoding="utf-8")
    return HTML_OUT


def find_browser() -> str:
    for candidate in BROWSERS:
        path = candidate if os.path.isabs(candidate) else shutil.which(candidate)
        if path and Path(path).exists():
            return path
    raise SystemExit("No Edge/Chrome found; open results/report.html and print it to PDF.")


def print_pdf(html_path: Path, pdf_path: Path) -> None:
    if pdf_path.exists():
        pdf_path.unlink()
    subprocess.run(
        [
            find_browser(),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--virtual-time-budget=15000",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=180,
    )
    if not pdf_path.exists():
        raise SystemExit("Browser exited without writing the PDF.")


def main():
    if not (RESULTS / "report.md").exists():
        raise SystemExit("results/report.md missing; run experiments/run_all.py first.")
    html_path = build_html()
    print_pdf(html_path, PDF_OUT)
    print(f"PDF written to {PDF_OUT} ({PDF_OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    sys.exit(main())
