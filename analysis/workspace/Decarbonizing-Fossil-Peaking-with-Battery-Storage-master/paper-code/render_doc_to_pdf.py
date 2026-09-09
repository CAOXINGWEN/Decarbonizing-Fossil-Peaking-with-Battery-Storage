"""
通用 markdown → PDF 渲染脚本（带中文字体、表格、嵌入对应图片）。

用法：
  python render_doc_to_pdf.py <md_path> [--figure <png_path>]
  python render_doc_to_pdf.py --batch  # 渲染预设的 m4 nature_figure 目录下所有说明文档
"""

from __future__ import annotations

import argparse
from pathlib import Path

import markdown
from weasyprint import HTML


PAPER_CODE_DIR = Path(__file__).resolve().parent
FIG_DIR = PAPER_CODE_DIR / "output" / "m4_economics" / "nature_figure"


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

p { text-align: justify; margin: 2mm 0; }

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

pre code { background: none; color: #1a1a1a; }

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

tr:nth-child(even) { background: #f6f8fa; }

td strong { color: #b03030; }

ul, ol { margin: 2mm 0 2mm 6mm; padding-left: 4mm; }
li { margin: 0.8mm 0; }

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

a:hover { text-decoration: underline; }

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


def build_html(md_text: str, base_dir: Path, fig_path: Path | None,
               caption: str | None) -> str:
    html_body = markdown.markdown(
        md_text, extensions=["tables", "fenced_code", "nl2br"]
    )

    fig_block = ""
    if fig_path is not None and fig_path.exists():
        fig_abs = fig_path.resolve()
        cap = caption or fig_path.stem
        fig_block = (
            f'<div>'
            f'<img src="file://{fig_abs}" alt="Figure" />'
            f'<div class="figure-caption">{cap}</div>'
            f'</div>'
        )

    head_html = (
        '<!DOCTYPE html>'
        '<html lang="zh-CN"><head>'
        '<meta charset="utf-8">'
        '<title>Document</title>'
        f'<style>{CSS_STYLE}</style>'
        '</head><body>'
    )
    tail_html = "</body></html>"

    if fig_block:
        parts = html_body.split("</p>", 1)
        if len(parts) == 2 and "<h1" not in parts[0]:
            body = parts[0] + "</p>" + fig_block + parts[1]
        else:
            body = fig_block + html_body
    else:
        body = html_body

    return head_html + body + tail_html


def render(md_path: Path, fig_path: Path | None = None,
           caption: str | None = None) -> Path:
    pdf_path = md_path.with_suffix(".pdf")
    md_text = md_path.read_text(encoding="utf-8")
    html = build_html(md_text, md_path.parent, fig_path, caption)
    HTML(string=html, base_url=str(md_path.parent)).write_pdf(pdf_path)
    print(f"PDF saved: {pdf_path}  ({pdf_path.stat().st_size / 1024:.1f} KB)")
    return pdf_path


BATCH_JOBS = [
    ("fig4b_m4_clustering_overlay_说明文档.md",
     "fig4b_m4_clustering_overlay.png",
     "Fig 4b — M4 经济边界 × 极端事件聚集度叠加图"),
    ("fig4b_诊断_长时储能效果差的原因.md",
     "fig4b_m4_clustering_overlay.png",
     "Fig 4b 诊断 — 长时储能 baseline 失利的成本结构溯源"),
    ("fig4e_baseline_vs_lowcost_osr_说明文档.md",
     "fig4e_baseline_vs_lowcost_osr.png",
     "Fig 4e — Baseline vs Low-Cost LDES Scenarios — OSR shifts and CAPEX gap"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("md_path", nargs="?", help="markdown 文件路径")
    parser.add_argument("--figure", help="嵌入的 png 路径")
    parser.add_argument("--caption", help="图注")
    parser.add_argument("--batch", action="store_true",
                        help="批量渲染 m4 nature_figure 目录预设的所有说明文档")
    args = parser.parse_args()

    if args.batch:
        for md_name, fig_name, caption in BATCH_JOBS:
            md = FIG_DIR / md_name
            fig = FIG_DIR / fig_name if fig_name else None
            if not md.exists():
                print(f"SKIP (missing): {md}")
                continue
            render(md, fig, caption)
    else:
        if not args.md_path:
            parser.error("must provide md_path or --batch")
        md = Path(args.md_path).resolve()
        fig = Path(args.figure).resolve() if args.figure else None
        render(md, fig, args.caption)


if __name__ == "__main__":
    main()
