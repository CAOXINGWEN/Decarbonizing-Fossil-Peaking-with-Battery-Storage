#!/usr/bin/env python3
"""Render FIGURE_EXPLANATION.md to a Chinese-readable PDF with figures embedded."""

from __future__ import annotations

from pathlib import Path

import markdown
from weasyprint import HTML


MODULE_DIR = Path(__file__).resolve().parent
MD_PATH = MODULE_DIR / "FIGURE_EXPLANATION.md"
PDF_PATH = MODULE_DIR / "FIGURE_EXPLANATION.pdf"
HTML_PATH = MODULE_DIR / "FIGURE_EXPLANATION.html"

FIGURE_INSERTS = {
    "## Figure 1: `fig_m4_lfp_ldes_residual_summary`": (
        "figures/fig_m4_lfp_ldes_residual_summary.png"
    ),
    "## Figure 2: `fig_m4_event_cost_scatter`": (
        "figures/fig_m4_event_cost_scatter.png"
    ),
    "## Figure 3: `fig_m4_best_ldes_tech_map`": (
        "figures/fig_m4_best_ldes_tech_map.png"
    ),
    "## Figure 4: `fig_m4_cost_saving_by_window`": (
        "figures/fig_m4_cost_saving_by_window.png"
    ),
}


CSS = """
@page {
  size: A4;
  margin: 18mm 16mm 18mm 16mm;
}
body {
  font-family: "Noto Sans CJK SC", "Noto Sans CJK TC", "DejaVu Sans", sans-serif;
  font-size: 10.5pt;
  line-height: 1.55;
  color: #222;
}
h1 {
  font-size: 20pt;
  margin: 0 0 12pt 0;
  padding-bottom: 6pt;
  border-bottom: 1.2pt solid #333;
}
h2 {
  font-size: 15pt;
  margin: 20pt 0 8pt 0;
  page-break-before: auto;
}
h2:nth-of-type(n+2) {
  page-break-before: always;
}
h3 {
  font-size: 12pt;
  margin: 14pt 0 6pt 0;
}
p {
  margin: 0 0 7pt 0;
}
ul {
  margin: 0 0 8pt 18pt;
  padding: 0;
}
li {
  margin-bottom: 3pt;
}
code {
  font-family: "DejaVu Sans Mono", "Liberation Mono", monospace;
  font-size: 9pt;
  background: #f3f3f3;
  padding: 1pt 3pt;
  border-radius: 2pt;
}
pre {
  font-family: "DejaVu Sans Mono", "Liberation Mono", monospace;
  font-size: 8.5pt;
  line-height: 1.35;
  background: #f6f6f6;
  padding: 8pt;
  border-left: 2.5pt solid #999;
  white-space: pre-wrap;
}
img.figure-preview {
  display: block;
  width: 100%;
  max-height: 165mm;
  object-fit: contain;
  margin: 8pt auto 12pt auto;
  border: 0.5pt solid #ddd;
}
.figure-note {
  font-size: 9pt;
  color: #555;
  margin-top: -5pt;
  margin-bottom: 10pt;
}
strong {
  font-weight: 700;
}
"""


def inject_figures(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    output: list[str] = []
    for line in lines:
        output.append(line)
        if line in FIGURE_INSERTS:
            image_path = FIGURE_INSERTS[line]
            output.extend(
                [
                    "",
                    f'<img class="figure-preview" src="{image_path}" alt="{line}">',
                    f'<p class="figure-note">图源文件：<code>{image_path}</code></p>',
                    "",
                ]
            )
    return "\n".join(output)


def main() -> None:
    raw_md = MD_PATH.read_text(encoding="utf-8")
    md_with_figures = inject_figures(raw_md)
    body = markdown.markdown(
        md_with_figures,
        extensions=["extra", "tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>M4 LFP-LDES residual demand 图件解释</title>
  <style>{CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")
    HTML(filename=str(HTML_PATH), base_url=str(MODULE_DIR)).write_pdf(str(PDF_PATH))
    print(PDF_PATH)


if __name__ == "__main__":
    main()
