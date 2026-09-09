"""
Convert fig4e markdown spec to a polished PDF.

Pipeline: markdown → HTML (with tables + Chinese fonts) → PDF (weasyprint).
"""

from __future__ import annotations

from pathlib import Path

import markdown
from weasyprint import HTML, CSS


PAPER_CODE_DIR = Path(__file__).resolve().parent
DOC_DIR = PAPER_CODE_DIR / "output" / "m4_economics" / "nature_figure"
MD_PATH = DOC_DIR / "fig4e_baseline_vs_lowcost_osr_说明文档.md"
PDF_PATH = DOC_DIR / "fig4e_baseline_vs_lowcost_osr_说明文档.pdf"


CSS_STYLE = """
@page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans", sans-serif;
        font-size: 9pt;
        color: #888;
    }
}

body {
    font-family: "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans", sans-serif;
    font-size: 10pt;
    line-height: 1.55;
    color: #1a1a1a;
}

h1 {
    font-size: 18pt;
    color: #0F4D92;
    border-bottom: 2px solid #0F4D92;
    padding-bottom: 4mm;
    margin-top: 0;
    page-break-after: avoid;
}

h2 {
    font-size: 14pt;
    color: #1F4E79;
    border-left: 4px solid #1F4E79;
    padding-left: 3mm;
    margin-top: 8mm;
    margin-bottom: 3mm;
    page-break-after: avoid;
}

h3 {
    font-size: 12pt;
    color: #2A6099;
    margin-top: 6mm;
    margin-bottom: 2mm;
    page-break-after: avoid;
}

h4 {
    font-size: 11pt;
    color: #444;
    margin-top: 4mm;
    margin-bottom: 1mm;
    page-break-after: avoid;
}

p {
    text-align: justify;
    margin: 2mm 0;
}

blockquote {
    border-left: 3px solid #ccc;
    padding: 1mm 4mm;
    margin: 3mm 0;
    color: #555;
    background: #f8f8f8;
    font-size: 9.5pt;
}

code {
    font-family: "Courier New", monospace;
    background: #f0f0f0;
    padding: 0.5mm 1mm;
    border-radius: 1mm;
    font-size: 9pt;
    color: #b03030;
}

pre {
    background: #f5f5f5;
    border: 1px solid #ddd;
    padding: 3mm;
    border-radius: 1.5mm;
    font-size: 9pt;
    overflow-x: auto;
    page-break-inside: avoid;
}

pre code {
    background: none;
    color: #1a1a1a;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 3mm 0;
    font-size: 9pt;
    page-break-inside: avoid;
}

th, td {
    border: 1px solid #bbb;
    padding: 1.5mm 2mm;
    text-align: left;
    vertical-align: top;
}

th {
    background: #1F4E79;
    color: white;
    font-weight: 600;
}

tr:nth-child(even) {
    background: #f6f8fa;
}

td strong {
    color: #b03030;
}

ul, ol {
    margin: 2mm 0 2mm 6mm;
    padding-left: 4mm;
}

li {
    margin: 0.8mm 0;
}

hr {
    border: none;
    border-top: 1px dashed #aaa;
    margin: 5mm 0;
}

a {
    color: #2A6099;
    text-decoration: none;
    word-break: break-all;
}

a:hover {
    text-decoration: underline;
}

img {
    max-width: 100%;
    margin: 4mm auto;
    display: block;
    page-break-inside: avoid;
    border: 1px solid #ddd;
    border-radius: 2mm;
}

.figure-caption {
    text-align: center;
    font-size: 9pt;
    color: #666;
    margin-top: -2mm;
    margin-bottom: 4mm;
    font-style: italic;
}
"""


def build_html(md_text: str) -> str:
    """Convert markdown to HTML, inject figure preview and CSS."""
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "nl2br"],
    )

    # Inject the figure as a visual preview right after the intro section
    fig_path = (DOC_DIR / "fig4e_baseline_vs_lowcost_osr.png").resolve()
    fig_block = (
        f'<div>'
        f'<img src="file://{fig_path}" alt="Fig 4e" />'
        f'<div class="figure-caption">Fig 4e — Baseline vs Low-Cost LDES Scenarios — '
        f'OSR shifts and CAPEX gap</div>'
        f'</div>'
    )

    # Place figure between <h2>0.</h2> intro and <h2>1.</h2> 数据口径
    # by inserting before the first "<h2" occurrence after the title.
    head_html = (
        '<!DOCTYPE html>'
        '<html lang="zh-CN"><head>'
        '<meta charset="utf-8">'
        '<title>Fig 4e 说明文档</title>'
        f'<style>{CSS_STYLE}</style>'
        '</head><body>'
    )
    tail_html = "</body></html>"

    # Inject figure after the first paragraph after the H1
    parts = html_body.split("</p>", 1)
    if len(parts) == 2 and "<h1" not in parts[0]:
        body_with_fig = parts[0] + "</p>" + fig_block + parts[1]
    else:
        body_with_fig = fig_block + html_body

    return head_html + body_with_fig + tail_html


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    html = build_html(md_text)

    HTML(string=html, base_url=str(DOC_DIR)).write_pdf(PDF_PATH)
    print(f"PDF saved: {PDF_PATH}")
    print(f"size: {PDF_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
