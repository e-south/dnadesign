"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/formatting.py

CLI-wide human-output formatting helpers and renderers.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

# -----------------------
# Core formatting helpers
# -----------------------


def _indent(s: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in s.splitlines())


def _fmt_multiline(v: object) -> str:
    """
    Pretty-print values inside kv blocks. If a value spans multiple lines,
    indent the continuation lines so nested JSON (e.g., selection.params)
    is easy to read in human stdout.
    """
    s = v if isinstance(v, str) else str(v)
    return ("\n" + _indent(s, 2)) if "\n" in s else s


def kv_block(title: str, items: Mapping[str, object]) -> str:
    """
    Render a simple key/value block. Keys are printed in the order provided
    to make caller control of layout deterministic (avoid implicit sorting).
    """
    lines = [str(title)]
    for k in items.keys():
        v = _fmt_multiline(items[k])
        lines.append(f"  {k:24s}: {v}")
    return "\n".join(lines)


def bullet_list(title: str, rows: Iterable[str]) -> str:
    rows = [str(r) for r in rows]
    if not rows:
        return f"{title}\n  • (none)"
    return title + "\n  • " + "\n  • ".join(rows)


def short_array(a, maxlen: int = 8) -> str:
    try:
        import numpy as np  # local import; optional

        arr = np.asarray(a).ravel().tolist()
    except Exception:
        try:
            arr = list(a)
        except Exception:
            return str(a)
    if len(arr) <= maxlen:
        return str(arr)
    head = ", ".join(
        f"{x:.4g}" if isinstance(x, float) else str(x) for x in arr[:maxlen]
    )
    return f"[{head}, …] (len={len(arr)})"


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    # Best-effort: reflect attribute dicts
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    # Fallback to string representation wrapped
    return {"value": str(obj)}


def _fmt_params(params: Any) -> str:
    try:
        return json.dumps(params, indent=2, sort_keys=True)
    except Exception:
        return str(params)


def _truncate(s: str, n: int = 48) -> str:
    s = str(s)
    return s if len(s) <= n else (s[: n - 1] + "…")


def _sha_short(sha: str, n: int = 12) -> str:
    sha = (sha or "").strip()
    return sha[:n] if sha else ""


# -----------------------
# Renderers per command
# -----------------------


def render_explain_human(info: Mapping[str, Any]) -> str:
    """Human summary for `opal explain`."""
    sel = info.get("selection", {}) or {}
    obj = sel.get("objective", {}) or {}
    yops = info.get("training_y_ops") or info.get("training_y_ops", []) or []
    yops_str = ", ".join(str(p.get("name")) for p in yops) if yops else "(none)"

    header = kv_block(
        f"Round r={info.get('round_index')}",
        {
            "X column": info.get("x_column_name"),
            "Y column": info.get("y_column_name"),
            "Vector dim (X)": info.get("representation_vector_dimension"),
        },
    )

    model_block = kv_block(
        "Model",
        {
            "name": (info.get("model") or {}).get("name"),
            "params": _fmt_params((info.get("model") or {}).get("params")),
            "Y-ops": yops_str,
        },
    )

    selection_block = kv_block(
        "Selection & Objective",
        {
            "strategy": sel.get("strategy"),
            "selection.params": _fmt_params(sel.get("params")),
            "objective": obj.get("name"),
            "objective.params": _fmt_params(obj.get("params")),
        },
    )

    counts = kv_block(
        "Counts",
        {
            "training labels used": info.get(
                "number_of_training_examples_used_in_round"
            ),
            "candidates scored": info.get("number_of_candidates_scored_in_round"),
        },
    )

    warnings = bullet_list("Warnings", info.get("warnings") or [])

    return "\n".join(
        [header, "", model_block, "", selection_block, "", counts, "", warnings]
    )


def render_ingest_preview_human(
    preview: Any,
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    transform_name: Optional[str] = None,
) -> str:
    """Human preview for `opal ingest-y` prior to confirmation."""
    p = _as_dict(preview)
    head = kv_block(
        f"[Preview] ingest-y (transform={transform_name or '(from YAML)'})",
        {
            "rows in input table": p.get("total_rows_in_csv"),
            "columns present": f"id={'yes' if p.get('rows_with_id') else 'no'}, "
            f"sequence={'yes' if p.get('rows_with_sequence') else 'no'}",
            "resolved ids by sequence": p.get("resolved_ids_by_sequence"),
            "unknown sequences": p.get("unknown_sequences"),
            "y_expected_length": p.get("y_expected_length"),
            "y_length_ok (sampled)": p.get("y_length_ok"),
            "y_length_bad (sampled)": p.get("y_length_bad"),
            "warnings": ", ".join(p.get("warnings") or []) or "(none)",
        },
    )
    if not sample_rows:
        return head

    # show at most 5 compact rows
    lines: List[str] = []
    for r in sample_rows[:5]:
        seq = _truncate(r.get("sequence", ""))
        rid = r.get("id", "")
        y = r.get("y", "")
        y_str = (
            short_array(y, maxlen=6)
            if isinstance(y, (list, tuple))
            else _truncate(str(y), 64)
        )
        lines.append(f"id={rid}  sequence={seq}  y={y_str}")

    return head + "\n\n" + bullet_list("Sample (first 5)", lines)


def render_ingest_commit_human(
    *,
    round_index: int,
    labels_appended: int,
    y_column_updated: str,
    events_sha256: str,
) -> str:
    return kv_block(
        "[Committed] ingest-y",
        {
            "round": round_index,
            "labels appended": labels_appended,
            "y column updated": y_column_updated,
            "events sha256": _sha_short(events_sha256),
        },
    )


def render_init_human(*, workdir: Path) -> str:
    return kv_block(
        "[ok] Initialized campaign workspace",
        {
            "workdir": str(Path(workdir).resolve()),
            "directories": "inputs/, outputs/",
            "marker": ".opal/config",
        },
    )


def render_model_show_human(info: Mapping[str, Any]) -> str:
    head = kv_block(
        "Model",
        {
            "type": info.get("model_type"),
            "params": _fmt_params(info.get("params", {})),
        },
    )
    top = info.get("feature_importance_top20") or []
    if not top:
        return head

    lines = []
    for i, row in enumerate(top, start=1):
        fi = row.get("feature_index")
        w = row.get("feature_importance")
        rk = row.get("feature_rank")
        lines.append(f"{i:2d}. feature_index={fi}  weight={w:.6g}  rank={rk}")
    return (
        head + "\n\n" + bullet_list("Top-20 feature importance (if available)", lines)
    )


def render_record_report_human(report: Mapping[str, Any]) -> str:
    if report.get("error"):
        return kv_block("record-show", {"error": report.get("error")})

    # Optional sources (succinct, relative paths)
    sources = report.get("sources") or {}
    src_block = ""
    if sources:
        src_block = "\n\n" + kv_block(
            "Sources",
            {
                "records": sources.get("records", ""),
                "events": sources.get("events", ""),
            },
        )

    head = kv_block(
        "Record",
        {
            "id": report.get("id"),
            "sequence": (
                _truncate(report.get("sequence", ""))
                if report.get("sequence")
                else "(hidden)"
            ),
            "label_hist_column": report.get("label_hist_column"),
            "n_labels": (
                (
                    len(report.get("labels"))
                    if isinstance(report.get("labels"), list)
                    else 0
                )
            ),
        },
    )

    # compact per-run lines
    runs = report.get("runs") or []
    lines = []
    for r in runs:
        parts = [
            f"r={r.get('as_of_round')}",
            f"run_id={_truncate(r.get('run_id',''), 18)}",
        ]
        # Show common fields if present
        for k in ("sel__is_selected", "sel__rank_competition", "pred__y_obj_scalar"):

            if k in r:
                parts.append(f"{k.split('__', 1)[-1]}={r[k]}")
        lines.append(", ".join(parts))

    runs_block = bullet_list("Runs (per as_of_round)", lines)
    return "\n".join([head + src_block, "", runs_block])


def render_run_summary_human(summary: dict) -> str:
    rid = summary.get("run_id", "")
    requested = summary.get("top_k_requested")
    effective = summary.get("top_k_effective")
    # tie policy is logged in the round log; not in summary. If you prefer, plumb it through.
    tie_hint = summary.get("tie_handling") or "competition_rank"
    sel_line = f"selection: top_k={requested} (requested) → selected={effective} (effective after ties, tie_handling={tie_hint})"  # noqa
    lines = [
        f"run_id: {rid}",
        f"as_of_round: {summary.get('as_of_round')}",
        f"trained_on: {summary.get('trained_on')} | scored: {summary.get('scored')}",
        sel_line,
        f"events: {summary.get('events')}",
        f"top_k_source: {summary.get('top_k_source')}",
    ]
    return "\n".join(lines)


def render_status_human(st: Mapping[str, Any]) -> str:
    head = kv_block(
        "Campaign",
        {
            "name": st.get("campaign_name"),
            "slug": st.get("campaign_slug"),
            "workdir": st.get("workdir"),
            "X column": st.get("x_column_name"),
            "Y column": st.get("y_column_name"),
            "num_rounds": st.get("num_rounds"),
        },
    )

    latest = st.get("latest_round") or {}
    if not latest:
        return head + "\n\n" + "No completed rounds."

    latest_block = kv_block(
        "Latest round",
        {
            "r": latest.get("round_index"),
            "n_train": latest.get("number_of_training_examples_used_in_round"),
            "n_scored": latest.get("number_of_candidates_scored_in_round"),
            "top_k requested": latest.get("selection_top_k_requested"),
            "top_k effective": latest.get("selection_top_k_effective_after_ties"),
            "round_dir": latest.get("round_dir"),
        },
    )
    return "\n".join([head, "", latest_block])
