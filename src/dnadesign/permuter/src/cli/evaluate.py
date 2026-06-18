"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/cli/evaluate.py

CLI wiring for evaluate Permuter CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import shlex
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from rich.console import Console

from dnadesign.permuter.src.cli.output import emit_json
from dnadesign.permuter.src.contracts.metrics import (
    interaction_metric_column,
    observed_metric_column,
)
from dnadesign.permuter.src.core.paths import normalize_data_path, resolve_workspace_config_hint
from dnadesign.permuter.src.core.registry import get_evaluator
from dnadesign.permuter.src.core.storage import (
    append_record_md,
    atomic_write_parquet,
    read_parquet,
    read_ref_fasta,
)
from dnadesign.permuter.src.evaluators.results import normalize_scores
from dnadesign.permuter.src.workspaces.datasets import resolve_workspace_dataset_path
from dnadesign.permuter.src.workspaces.loader import load_workspace

console = Console()
_LOG = logging.getLogger("permuter.evaluate")


def _load_workspace_metrics(workspace_hint: Optional[Union[str, Path]]) -> List[Dict]:
    if not workspace_hint:
        return []
    config_path = resolve_workspace_config_hint(workspace_hint)
    cfg = load_workspace(config_path).config
    metrics: List[Dict] = []
    if cfg.scope.evaluate and cfg.scope.evaluate.metrics:
        for m in cfg.scope.evaluate.metrics:
            metrics.append(
                {
                    "id": m.id,
                    "evaluator": m.evaluator,
                    "metric": m.metric,
                    "params": m.params,
                }
            )
    return metrics


def _parse_cli_with(args: List[str]) -> List[Dict]:
    # each item: id:evaluator[:metric]; metric defaults to id
    out = []
    for s in args:
        parts = [p.strip() for p in s.split(":") if p.strip()]
        if len(parts) < 2:
            raise ValueError(f"--with expects id:evaluator[:metric], got {s!r}")
        mid, ev = parts[0], parts[1]
        metric = parts[2] if len(parts) >= 3 else mid
        out.append({"id": mid, "evaluator": ev, "metric": metric, "params": {}})
    return out


def _derive_records_from_workspace(workspace_hint: str, ref: Optional[str], out: Optional[Path]) -> Path:
    return resolve_workspace_dataset_path(workspace_hint=workspace_hint, ref=ref, out=out).records


def _argv() -> str:
    try:
        return shlex.join(sys.argv)
    except Exception:
        return " ".join(sys.argv)


def _summarize_first_column_for_log(
    s: pd.Series,
) -> tuple[float, float, float, float, float, Optional[int]]:
    """
    Robust summary for logging:
      • If `s` is numeric → compute mean, sd, min, p50, max directly.
      • If `s` has per-row sequences (list/ndarray/Series of numbers) → compute the
        same statistics over each row's mean; also return a vector dimension when
        consistent across rows (else None).
    No silent fallbacks: non-numeric, non-sequence values become NaN in the summary.
    """
    # Numeric scalar column
    if pd.api.types.is_numeric_dtype(s):
        a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(a)
        if finite.sum() == 0:
            return (float("nan"),) * 5 + (None,)
        mean = float(np.nanmean(a))
        sd = float(np.nanstd(a, ddof=1)) if finite.sum() > 1 else float("nan")
        mn = float(np.nanmin(a))
        p50 = float(np.nanmedian(a))
        mx = float(np.nanmax(a))
        return mean, sd, mn, p50, mx, None

    # Sequence-of-numbers per row
    vec_lens: list[int] = []
    row_means: list[float] = []
    for x in s:
        if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            arr = np.array(list(x), dtype=float)
            vec_lens.append(int(len(arr)))
            if arr.size:
                row_means.append(float(np.nanmean(arr)))
            else:
                row_means.append(np.nan)
        else:
            row_means.append(np.nan)
    a = np.array(row_means, dtype=float)
    finite = np.isfinite(a)
    if finite.sum() == 0:
        vec_dim = vec_lens[0] if vec_lens else None
        return (float("nan"),) * 5 + (vec_dim,)
    mean = float(np.nanmean(a))
    sd = float(np.nanstd(a, ddof=1)) if finite.sum() > 1 else float("nan")
    mn = float(np.nanmin(a))
    p50 = float(np.nanmedian(a))
    mx = float(np.nanmax(a))
    vec_dim = None
    if vec_lens:
        u = set(vec_lens)
        if len(u) == 1:
            vec_dim = int(next(iter(u)))
    return mean, sd, mn, p50, mx, vec_dim


def evaluate(
    data: Path | None,
    metric_ids: List[str] | None = None,
    with_spec: List[str] | None = None,
    workspace: Optional[str] = None,
    ref: Optional[str] = None,
    out: Optional[Path] = None,
    as_json: bool = False,
) -> dict[str, object]:
    # Resolve records path from either --data or --workspace/--ref
    if data is not None:
        records = normalize_data_path(data)
    elif workspace:
        try:
            records = _derive_records_from_workspace(workspace, ref, out)
        except Exception as e:
            raise ValueError(
                f"Unable to derive dataset from --workspace. {e}\n"
                "Hint: supply --ref if your refs CSV has multiple rows."
            ) from e
    else:
        raise ValueError("Provide either --data (file or dataset dir) or --workspace/--ref.")

    if not records.exists():
        # Primary, actionable hint
        workspace_hint = f"--workspace {workspace} --ref {ref}" if workspace else "(your workspace)"
        raise FileNotFoundError(
            f"Dataset not found: {records}\n"
            f"Generate it first with:\n"
            f"  permuter run {workspace_hint}\n"
            f"Then re-run:\n"
            f"  permuter evaluate --data {records.parent}\n"
        )

    df = read_parquet(records)
    if "sequence" not in df.columns or "bio_type" not in df.columns:
        raise ValueError("records.parquet missing USR core columns (sequence, bio_type)")

    # Explicit CLI metrics replace workspace defaults so smoke/adversarial probes
    # do not accidentally invoke closed-loop evaluators from config.
    metrics = [] if (with_spec or metric_ids) else _load_workspace_metrics(workspace)
    if with_spec:
        metrics.extend(_parse_cli_with(with_spec))
    if metric_ids:
        # convenience: if only ids given, assume placeholder with metric==id
        for mid in metric_ids:
            metrics.append({"id": mid, "evaluator": "placeholder", "metric": mid, "params": {}})
    if not metrics:
        raise ValueError(
            "No metrics specified. Use --with id:evaluator[:metric] or provide a workspace config with evaluate.metrics"
        )
    # De-duplicate by id (last one wins, so CLI overrides workspace config)
    uniq: Dict[str, Dict] = {}
    for m in metrics:
        mid = m.get("id")
        if not mid:
            raise ValueError("Metric entries must have an 'id'")
        uniq[str(mid)] = m
    metrics = list(uniq.values())

    # get reference seq, required for some evaluators (e.g. evo2_llr)
    ref = read_ref_fasta(records.parent)
    ref_sequence = ref[1] if ref else None
    if ref and ref_sequence and len(ref_sequence) > 0:
        _LOG.info("evaluate: REF loaded for baseline • name=%s • length=%d nt/aa", ref[0], len(ref_sequence))

    sequences = df["sequence"].astype(str).tolist()

    # Accumulate new metric columns here and add them in a single concat to avoid fragmentation
    new_metric_frames: list[pd.DataFrame] = []

    for mc in metrics:
        ev_cls = get_evaluator(mc["evaluator"])
        ev = ev_cls(**(mc.get("params") or {}))
        try:
            scores = ev.score(
                sequences,
                metric=mc["metric"],
                ref_sequence=ref_sequence,
                ref_embedding=None,
            )
        except Exception as e:
            # Friendlier guidance for common pitfalls
            msg = str(e)
            if "requires ref_sequence" in msg or "ref_sequence" in msg:
                raise RuntimeError(
                    "This evaluator requires a reference sequence (REF.fa).\n"
                    f"Missing sidecar: {records.parent / 'REF.fa'}\n"
                    "Generate the dataset with:\n"
                    f"  permuter run --workspace {workspace or '<workspace>'} --ref {ref or '<ref>'}\n"
                ) from e
            raise

        # Normalize evaluator output directly into the canonical observed namespace.
        cols = normalize_scores(scores, n=len(sequences), metric_id=mc["id"])
        cols_df = pd.DataFrame(cols)  # aligns on RangeIndex 0..n-1
        new_metric_frames.append(cols_df)
        # Log quick stats for the first column (without mutating df yet)
        first_col = cols_df.iloc[:, 0]
        mean, sd, mn, p50, mx, vec_dim = _summarize_first_column_for_log(first_col)
        p = mc.get("params") or {}
        red = p.get("reduction", None)
        # evaluator-specific flavor for evo2
        extra = ""
        if str(mc["evaluator"]).startswith("evo2_"):
            extra = (
                f" model={p.get('model_id', '?')} device={p.get('device', '?')}"
                f" prec={p.get('precision', '?')} alpha={p.get('alphabet', '?')}"
            )
        if vec_dim is not None:
            extra = f"{extra} vecdim={vec_dim}"
        _LOG.info(
            "evaluate: id=%s eval=%s metric=%s%s%s n=%d mean=%.4f sd=%.4f min=%.4f p50=%.4f max=%.4f",
            mc["id"],
            mc["evaluator"],
            mc["metric"],
            (f" reduction={red}" if red else ""),
            extra,
            len(first_col),
            mean,
            sd,
            mn,
            p50,
            mx,
        )
        if str(mc["evaluator"]).strip() == "evo2_llr":
            _LOG.info("evaluate: LLR computed as log P(variant) - log P(reference) using REF.fa")

    # Single concat to add all metric columns at once (zero/low-copy when possible)
    if new_metric_frames:
        df = pd.concat([df] + new_metric_frames, axis=1)
        # Defragment once so downstream ops (parquet write / slicing) stay fast
        df = df.copy()

    # ---- Canonical interaction metric (generic, non-protocol-specific) ------
    # If a protocol emitted exactly one expected column, compute epistasis.
    exp_cols = [c for c in df.columns if c.startswith("permuter__expected__")]
    if len(exp_cols) == 1:
        exp_col = exp_cols[0]
        metric_for_exp = exp_col[len("permuter__expected__") :]
        obs_col = observed_metric_column(metric_for_exp)
        if obs_col not in df.columns:
            raise RuntimeError(
                "evaluate: expected column present but matching observed column is missing.\n"
                f"  expected: {exp_col}\n"
                f"  missing : {obs_col}\n"
                "Ensure the evaluator id matches the protocol's singles_metric_id."
            )
        epi_col = interaction_metric_column("epistasis", metric_for_exp)
        df[epi_col] = df[obs_col].astype("float64") - df[exp_col].astype("float64")
        _LOG.info(
            "evaluate: attached epistasis=%s using observed=%s and expected=%s",
            epi_col,
            obs_col,
            exp_col,
        )
    elif len(exp_cols) > 1:
        raise RuntimeError(
            "evaluate: multiple 'permuter__expected__*' columns found; ambiguous for a single epistasis column.\n"
            f"Found: {exp_cols}\nEmit exactly one expected metric in your protocol."
        )

    atomic_write_parquet(df, records)
    summary: dict[str, object] = {
        "schema": "permuter.evaluate.v1",
        "records": records,
        "dataset_dir": records.parent,
        "row_count": len(df),
        "metrics": [str(m["id"]) for m in metrics],
        "observed_columns": sorted(c for c in df.columns if str(c).startswith("permuter__observed__")),
    }
    if as_json:
        emit_json(summary)
    else:
        console.print(f"[green]✔[/green] Appended metrics: {', '.join(m['id'] for m in metrics)} → {records}")
    append_record_md(records.parent, "evaluate", _argv())
    return summary
