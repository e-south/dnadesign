"""Formatting helpers shared by review report renderers."""

from __future__ import annotations

import os
from html import escape
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not pd.notna(number):
        return ""
    if abs(number) >= 100:
        return f"{number:.1f}"
    if abs(number) >= 10:
        return f"{number:.2f}"
    if abs(number) >= 0.01:
        return f"{number:.3f}"
    return f"{number:.3g}"


def _gate_observed(row: Mapping[str, Any]) -> Any:
    gate = row.get("gate")
    if gate == "H-NULL-CONTROL":
        return row.get("null_lift")
    if gate == "H-NULL-ROUND-DYNAMICS":
        return row.get("max_lift")
    if gate == "H-POSITIVE-SEPARATION":
        return row.get("positive_minus_null_lift")
    if gate == "H-TRAJECTORY-SEPARATION":
        return row.get("paired_auc_delta")
    return row.get("observed", "")


def _gate_threshold(row: Mapping[str, Any]) -> Any:
    gate = row.get("gate")
    if gate == "H-NULL-CONTROL":
        return row.get("null_lift_attention_baseline")
    if gate == "H-NULL-ROUND-DYNAMICS":
        return row.get("threshold")
    if gate == "H-POSITIVE-SEPARATION":
        return 0.0
    if gate == "H-TRAJECTORY-SEPARATION":
        return 0.0
    return row.get("threshold", "")


def _e(value: Any) -> str:
    return escape("" if value is None else str(value), quote=True)


def _rel(path: Any, *, base_dir: Path) -> str:
    return os.path.relpath(str(path), str(base_dir))


def _metric_card(label: str, value: Any) -> str:
    return f'<article class="metric"><span>{_e(label)}</span><strong>{_e(value)}</strong></article>'


def _html_document(*, title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_e(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #1f2528;
      --muted: #667074;
      --line: #d8ddd7;
      --accent: #8c4e4a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header, main {{ width: min(1180px, calc(100vw - 40px)); margin: 0 auto; }}
    header {{ padding: 34px 0 16px; }}
    header > p:first-child {{
      color: var(--accent);
      font-size: 0.82rem;
      font-weight: 700;
      margin: 0 0 6px;
      text-transform: uppercase;
    }}
    .lede {{ color: var(--muted); margin: 8px 0 0; }}
    h1 {{ font-size: clamp(1.8rem, 2.8vw, 3rem); margin: 0; overflow-wrap: anywhere; }}
    h2 {{
      border-bottom: 1px solid var(--line);
      font-size: 1.18rem;
      margin: 30px 0 14px;
      padding-bottom: 8px;
    }}
    code {{ background: #eef1ef; border-radius: 4px; padding: 1px 5px; }}
    .summary-grid {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }}
    .metric, .plot-grid article {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 1px 2px rgb(0 0 0 / 4%);
    }}
    .metric {{ min-height: 86px; padding: 14px; }}
    .metric span {{ color: var(--muted); display: block; font-size: 0.78rem; text-transform: uppercase; }}
    .metric strong {{ display: block; font-size: 1.25rem; margin-top: 8px; overflow-wrap: anywhere; }}
    dl {{ display: grid; gap: 8px 16px; grid-template-columns: minmax(160px, max-content) 1fr; }}
    dt {{ color: var(--muted); font-weight: 700; }}
    dd {{ margin: 0; overflow-wrap: anywhere; }}
    .plot-grid {{ display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); }}
    .plot-grid article {{ padding: 12px; }}
    .plot-grid h3 {{ font-size: 0.95rem; margin: 0 0 10px; text-transform: capitalize; }}
    .plot-thumb-grid {{ display: grid; gap: 10px; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); }}
    figure {{ margin: 10px 0 0; }}
    figcaption {{ color: var(--muted); font-size: 0.78rem; margin-top: 5px; overflow-wrap: anywhere; }}
    img {{ display: block; height: auto; max-width: 100%; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase; }}
    li {{ margin: 6px 0; }}
    @media (max-width: 640px) {{
      header, main {{ width: min(100vw - 24px, 1180px); }}
      dl {{ grid-template-columns: 1fr; }}
      .plot-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""
