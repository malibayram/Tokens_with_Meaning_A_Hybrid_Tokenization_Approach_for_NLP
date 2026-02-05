#!/usr/bin/env python3
"""
Generate color-aware TurBLiMP LaTeX + inline-styled HTML from the Pandas Styler HTML.

Input:
  turblimp_results_tables/results_table_centroid.html

Outputs:
  turblimp_results_tables/results_table_centroid_colored.tex
  turblimp_results_tables/results_table_centroid_inlined.html
"""

from __future__ import annotations

import os
import re
from pathlib import Path


INPUT_HTML = Path("turblimp_results_tables/results_table_centroid.html")
OUTPUT_TEX = Path("turblimp_results_tables/results_table_centroid_colored.tex")
OUTPUT_INLINE_HTML = Path("turblimp_results_tables/results_table_centroid_inlined.html")

MODEL_LABELS = {
    "mft": "TurkishTokenizer",
    "mursit": "Mursit",
    "newmindaimursit": "Mursit",
    "cosmosgpt2": "CosmosGPT2",
    "tabi": "Tabi",
}


CSS_BLOCK_RE = re.compile(r"<style[^>]*>(?P<css>.*?)</style>", re.DOTALL | re.IGNORECASE)
CSS_RULE_RE = re.compile(r"(?P<selectors>[^{}]+)\{(?P<body>[^{}]+)\}", re.DOTALL)
RGB_RE = re.compile(r"background-color\s*:\s*rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*;?")

THEAD_RE = re.compile(r"<thead>(?P<thead>.*?)</thead>", re.DOTALL | re.IGNORECASE)
TH_TEXT_RE = re.compile(r"<th[^>]*>(?P<text>.*?)</th>", re.DOTALL | re.IGNORECASE)

TBODY_RE = re.compile(r"<tbody>(?P<tbody>.*?)</tbody>", re.DOTALL | re.IGNORECASE)
TR_RE = re.compile(r"<tr>(?P<tr>.*?)</tr>", re.DOTALL | re.IGNORECASE)
ROW_TH_RE = re.compile(r"<th[^>]*>(?P<label>.*?)</th>", re.DOTALL | re.IGNORECASE)
TD_RE = re.compile(r'<td[^>]*id="(?P<id>[^"]+)"[^>]*>(?P<val>.*?)</td>', re.DOTALL | re.IGNORECASE)


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def parse_css_cell_colors(html: str) -> dict[str, tuple[int, int, int]]:
    m = CSS_BLOCK_RE.search(html)
    if not m:
        return {}
    css = m.group("css")

    colors: dict[str, tuple[int, int, int]] = {}
    for rule in CSS_RULE_RE.finditer(css):
        body = rule.group("body")
        m_rgb = RGB_RE.search(body)
        if not m_rgb:
            continue
        rgb = (int(m_rgb.group(1)), int(m_rgb.group(2)), int(m_rgb.group(3)))
        selectors = rule.group("selectors")
        for sel in selectors.split(","):
            sel = sel.strip()
            if not sel.startswith("#"):
                continue
            cell_id = sel[1:]
            colors[cell_id] = rgb
    return colors


def parse_table(html: str) -> tuple[list[str], list[tuple[str, list[tuple[str, str]]]]]:
    thead_m = THEAD_RE.search(html)
    if not thead_m:
        raise ValueError("Could not find <thead> in HTML.")
    thead = thead_m.group("thead")
    th_texts = [strip_tags(t) for t in TH_TEXT_RE.findall(thead)]
    # First th is the blank index header; remaining are model columns.
    col_headers = [t for t in th_texts[1:] if t]
    if not col_headers:
        raise ValueError("Could not parse column headers from HTML.")

    tbody_m = TBODY_RE.search(html)
    if not tbody_m:
        raise ValueError("Could not find <tbody> in HTML.")
    tbody = tbody_m.group("tbody")

    rows: list[tuple[str, list[tuple[str, str]]]] = []
    for tr_m in TR_RE.finditer(tbody):
        tr = tr_m.group("tr")
        row_th = ROW_TH_RE.search(tr)
        if not row_th:
            continue
        label = strip_tags(row_th.group("label"))
        cells = [(m.group("id"), strip_tags(m.group("val"))) for m in TD_RE.finditer(tr)]
        if not cells:
            continue
        rows.append((label, cells))
    return col_headers, rows


def write_colored_tex(
    out_path: Path,
    col_headers: list[str],
    rows: list[tuple[str, list[tuple[str, str]]]],
    cell_colors: dict[str, tuple[int, int, int]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Map header labels if we recognize them.
    headers = [MODEL_LABELS.get(h.lower(), h) for h in col_headers]
    col_spec = "l" + ("r" * len(headers))

    lines: list[str] = []
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & " + " & ".join(headers) + " \\\\")
    lines.append("\\midrule")

    for label, cells in rows:
        parts = [label]
        for cell_id, value in cells:
            rgb = cell_colors.get(cell_id)
            if rgb is None:
                parts.append(value)
            else:
                r, g, b = rgb
                parts.append(f"\\cellcolor[RGB]{{{r},{g},{b}}} {value}")
        lines.append(" & ".join(parts) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_inlined_html(
    out_path: Path,
    html: str,
    cell_colors: dict[str, tuple[int, int, int]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove the <style> block and apply inline styles to each <td id="...">.
    html_wo_style = CSS_BLOCK_RE.sub("", html)

    def repl_td(match: re.Match) -> str:
        cell_id = match.group("id")
        val = match.group("val")
        rgb = cell_colors.get(cell_id)
        if rgb is None:
            return match.group(0)
        r, g, b = rgb
        return f'<td id="{cell_id}" style="background-color: rgb({r}, {g}, {b});">{val}</td>'

    html_inlined = re.sub(r'<td([^>]*?)id="(?P<id>[^"]+)"([^>]*?)>(?P<val>.*?)</td>', repl_td, html_wo_style, flags=re.DOTALL | re.IGNORECASE)
    out_path.write_text(html_inlined, encoding="utf-8")


def main() -> None:
    if not INPUT_HTML.exists():
        raise FileNotFoundError(f"Missing {INPUT_HTML}")
    html = INPUT_HTML.read_text(encoding="utf-8")

    cell_colors = parse_css_cell_colors(html)
    col_headers, rows = parse_table(html)

    write_colored_tex(OUTPUT_TEX, col_headers, rows, cell_colors)
    print(f"✓ Wrote {OUTPUT_TEX}")

    write_inlined_html(OUTPUT_INLINE_HTML, html, cell_colors)
    print(f"✓ Wrote {OUTPUT_INLINE_HTML}")


if __name__ == "__main__":
    main()

