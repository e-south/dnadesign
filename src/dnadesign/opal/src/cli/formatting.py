"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/formatting.py

CLI-wide human-output formatting helpers and renderers.
Adds optional Rich markup (guarded by OPAL_CLI_MARKUP). JSON remains unstyled.
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

# -------------
# Markup gate
# -------------
_TRUTHY = {"1", "true", "yes", "on"}


def _markup_enabled() -> bool:
    val = os.getenv("OPAL_CLI_MARKUP", "").strip().lower()
    if val == "":
        # default on; app.py sets OPAL_CLI_MARKUP alongside --color
        return True
    return val in _TRUTHY


_M = _markup_enabled()


def _b(s: str) -> str:
    return f"[bold]{s}[/]" if _M else s


def _t(s: str) -> str:
    # title accent
    return f"[bold cyan]{s}[/]" if _M else s


def _dim(s: str) -> str:
    return f"[dim]{s}[/]" if _M else s


# -----------------------
# Core formatting helpers
# -----------------------


def _indent(s: str, n: int = 2) -> str:
    pad = " " * n
    return "\n".join(pad + line if line else line for line in s.splitlines())


def _fmt_multiline(v: object) -> str:
    s = v if isinstance(v, str) else str(v)
    return ("\n" + _indent(s, 2)) if "\n" in s else s


def kv_block(title: str, items: Mapping[str, object]) -> str:
    """
    Render a simple key/value block with a styled header and aligned keys.
    """
    lines = [_t(str(title))]
    for k in items.keys():
        v = _fmt_multiline(items[k])
        key = f"{_b(str(k))}"
        lines.append(f"  {key:24s}: {v}")
    return "\n".join(lines)


def bullet_list(title: str, rows: Iterable[str]) -> str:
    rows = [str(r) for r in rows]
    bullet = "•"
    if not rows:
        return f"{_t(title)}\n  {bullet} {_dim('(none)')}"
    return _t(title) + "\n  " + f"{bullet} " + f"\n  {bullet} ".join(rows)


def short_array(a, maxlen: int = 8) -> str:
    try:
        import numpy as np  # optional

        arr = np.asarray(a).ravel().tolist()
    except Exception:
        try:
            arr = list(a)
        except Exception:
            return str(a)
    if len(arr) <= maxlen:
        return str(arr)
    head = ", ".join(f"{x:.4g}" if isinstance(x, float) else str(x) for x in arr[:maxlen])
    return f"[{head}, …] (len={len(arr)})"


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": str(obj)}


def _fmt_params(params: Any) -> str:
    try:
        txt = json.dumps(params, indent=2, sort_keys=True)
        return txt if not _M else f"[white]{txt}[/]"
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
            "training labels used": info.get("number_of_training_examples_used_in_round"),
            "candidates scored": info.get("number_of_candidates_scored_in_round"),
            "candidate pool total": info.get("candidate_pool_total"),
            "candidate pool filtered out": info.get("candidate_pool_filtered_out"),
        },
    )

    warnings = bullet_list("Warnings", info.get("warnings") or [])

    return "\n".join([header, "", model_block, "", selection_block, "", counts, "", warnings])


def render_ingest_preview_human(
    preview: Any,
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    transform_name: Optional[str] = None,
) -> str:
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
            "duplicate policy": p.get("duplicate_policy"),
            "duplicate key": p.get("duplicate_key"),
            "duplicates found": p.get("duplicates_found"),
            "duplicates dropped": p.get("duplicates_dropped"),
            "warnings": ", ".join(p.get("warnings") or []) or "(none)",
        },
    )
    if not sample_rows:
        return head

    lines: List[str] = []
    for r in sample_rows[:5]:
        seq = _truncate(r.get("sequence", ""))
        rid = r.get("id", "")
        y = r.get("y", "")
        y_str = short_array(y, maxlen=6) if isinstance(y, (list, tuple)) else _truncate(str(y), 64)
        lines.append(f"id={rid}  sequence={seq}  y={y_str}")

    return head + "\n\n" + bullet_list("Sample (first 5)", lines)


def render_ingest_commit_human(
    *,
    round_index: int,
    labels_appended: int,
    y_column_updated: str,
) -> str:
    return kv_block(
        "[Committed] ingest-y",
        {
            "round": round_index,
            "labels appended": labels_appended,
            "y column updated": y_column_updated,
            "ledger labels appended": "yes",
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
    return head + "\n\n" + bullet_list("Top-20 feature importance (if available)", lines)


def render_record_report_human(report: Mapping[str, Any]) -> str:
    if report.get("error"):
        return kv_block("record-show", {"error": report.get("error")})

    sources = report.get("sources") or {}
    src_block = ""
    if sources:
        src_block = "\n\n" + kv_block(
            "Sources",
            {
                "records": sources.get("records", ""),
                "ledger_predictions": sources.get("ledger_predictions_dir", ""),
                "ledger_runs": sources.get("ledger_runs_path", ""),
            },
        )

    head = kv_block(
        "Record",
        {
            "id": report.get("id"),
            "sequence": (_truncate(report.get("sequence", "")) if report.get("sequence") else "(hidden)"),
            "label_hist_column": report.get("label_hist_column"),
            "n_labels": (len(report.get("labels")) if isinstance(report.get("labels"), list) else 0),
        },
    )

    runs = report.get("runs") or []
    lines = []
    for r in runs:
        parts = [
            f"r={r.get('as_of_round')}",
            f"run_id={_truncate(r.get('run_id', ''), 18)}",
        ]
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
    tie_hint = summary.get("tie_handling") or "competition_rank"
    obj_mode = summary.get("objective_mode") or "maximize"
    sel_line = (
        "selection: "
        f"objective={obj_mode} tie={tie_hint} | "
        f"top_k={requested} (requested) → selected={effective} (effective after ties)"
    )
    lines = [
        f"{_b('run_id')}: {rid}",
        f"{_b('as_of_round')}: {summary.get('as_of_round')}",
        f"{_b('trained_on')}:{' '}{summary.get('trained_on')} | {_b('scored')}:{' '}{summary.get('scored')}",
        sel_line,
        f"{_b('ledger')}: {summary.get('ledger')}",
        f"{_b('top_k_source')}: {summary.get('top_k_source')}",
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
        return head + "\n\n" + _dim("No completed rounds.")

    def _round_block(label: str, round_info: Mapping[str, Any], ledger_info: Mapping[str, Any]) -> list[str]:
        run_id = round_info.get("run_id") or ledger_info.get("run_id")
        main = kv_block(
            label,
            {
                "r": round_info.get("round_index"),
                "run_id": run_id,
                "n_train": round_info.get("number_of_training_examples_used_in_round"),
                "n_scored": round_info.get("number_of_candidates_scored_in_round"),
                "top_k requested": round_info.get("selection_top_k_requested"),
                "top_k effective": round_info.get("selection_top_k_effective_after_ties"),
                "round_dir": round_info.get("round_dir"),
            },
        )
        blocks = [main]
        if ledger_info:
            summary = ledger_info.get("objective_summary_stats") or {}
            kv = {
                "model": ledger_info.get("model"),
                "objective": ledger_info.get("objective"),
                "selection": ledger_info.get("selection"),
                "y_ops": ", ".join([p.get("name") for p in (ledger_info.get("y_ops") or [])]) or "(none)",
                "score_min": summary.get("score_min"),
                "score_median": summary.get("score_median"),
                "score_max": summary.get("score_max"),
            }
            blocks.append(kv_block(f"{label} (ledger)", kv))
        return blocks

    parts = [head]
    # Latest round block
    latest_blocks = _round_block("Latest round", latest, st.get("latest_round_ledger") or {})
    for b in latest_blocks:
        parts.extend(["", b])

    # Selected round (if specified and distinct)
    selected = st.get("selected_round") or {}
    if selected and int(selected.get("round_index", -1)) != int(latest.get("round_index", -1)):
        sel_blocks = _round_block("Selected round", selected, st.get("selected_round_ledger") or {})
        for b in sel_blocks:
            parts.extend(["", b])

    return "\n".join(parts)


def render_runs_list_human(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return _dim("No runs found.")
    lines = []
    for r in rows:
        parts = [
            f"r={r.get('as_of_round')}",
            f"run_id={_truncate(r.get('run_id', ''), 18)}",
            f"model={r.get('model')}",
            f"objective={r.get('objective')}",
            f"selection={r.get('selection')}",
            f"n_train={r.get('stats_n_train')}",
            f"n_scored={r.get('stats_n_scored')}",
        ]
        lines.append(", ".join(parts))
    return bullet_list("Runs", lines)


def render_run_meta_human(row: Mapping[str, Any]) -> str:
    y_ops = row.get("training__y_ops") or []
    y_ops_str = ", ".join([p.get("name") for p in y_ops]) if y_ops else "(none)"
    head = kv_block(
        "Run",
        {
            "run_id": row.get("run_id"),
            "as_of_round": row.get("as_of_round"),
            "model": row.get("model__name"),
            "objective": row.get("objective__name"),
            "selection": row.get("selection__name"),
            "y_ops": y_ops_str,
            "n_train": row.get("stats__n_train"),
            "n_scored": row.get("stats__n_scored"),
        },
    )
    obj_stats = row.get("objective__summary_stats") or {}
    stats_block = kv_block("Objective summary", obj_stats) if obj_stats else _dim("No objective summary stats.")
    artifacts = row.get("artifacts") or {}
    artifacts_block = kv_block("Artifacts", artifacts) if artifacts else _dim("No artifacts recorded.")
    return "\n".join([head, "", stats_block, "", artifacts_block])


def render_round_log_summary_human(summary: Mapping[str, Any]) -> str:
    head = kv_block(
        "Round log",
        {
            "round": summary.get("round_index"),
            "path": summary.get("path"),
            "events": summary.get("events"),
            "predict_batches": summary.get("predict_batches"),
            "predict_rows": summary.get("predict_rows"),
            "duration_total_s": summary.get("duration_sec_total"),
            "duration_fit_s": summary.get("duration_sec_fit"),
        },
    )
    stages = summary.get("stage_counts") or {}
    stage_lines = [f"{k}: {v}" for k, v in sorted(stages.items())] if stages else []
    stages_block = bullet_list("Stages", stage_lines)
    return "\n".join([head, "", stages_block])
