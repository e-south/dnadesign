"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/cli/evaluate.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import numbers
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from rich.console import Console

from dnadesign.permuter.src.core.config import JobConfig
from dnadesign.permuter.src.core.registry import get_evaluator
from dnadesign.permuter.src.core.storage import (
    atomic_write_parquet,
    read_parquet,
    read_ref_fasta,
)

console = Console()
_LOG = logging.getLogger("permuter.evaluate")


def _load_job_metrics(job_path: Optional[Path]) -> List[Dict]:
    if not job_path:
        return []
    data = yaml.safe_load(Path(job_path).read_text(encoding="utf-8"))
    cfg = JobConfig.model_validate(data)
    metrics = []
    if cfg.job.evaluate and cfg.job.evaluate.metrics:
        for m in cfg.job.evaluate.metrics:
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


def evaluate(
    data: Path,
    metric_ids: List[str] | None = None,
    with_spec: List[str] | None = None,
    job: Optional[Path] = None,
):
    df = read_parquet(data)
    if "sequence" not in df.columns or "bio_type" not in df.columns:
        raise ValueError(
            "records.parquet missing USR core columns (sequence, bio_type)"
        )

    # gather metrics
    metrics = _load_job_metrics(job)
    if with_spec:
        metrics.extend(_parse_cli_with(with_spec))
    if metric_ids:
        # convenience: if only ids given, assume placeholder with metric==id
        for mid in metric_ids:
            metrics.append(
                {"id": mid, "evaluator": "placeholder", "metric": mid, "params": {}}
            )
    if not metrics:
        raise ValueError(
            "No metrics specified. Use --with id:evaluator[:metric] or provide a job YAML with evaluate.metrics"
        )

    # get reference seq, required for some evaluators (e.g. evo2_llr)
    ref = read_ref_fasta(data.parent)
    ref_sequence = ref[1] if ref else None

    sequences = df["sequence"].astype(str).tolist()

    for mc in metrics:
        ev_cls = get_evaluator(mc["evaluator"])
        ev = ev_cls(**(mc.get("params") or {}))
        scores = ev.score(
            sequences,
            metric=mc["metric"],
            ref_sequence=ref_sequence,
            ref_embedding=None,
        )
        # Normalize and write columns
        cols = _normalize_scores(scores, n=len(sequences), metric_id=mc["id"])
        for col_name, series in cols.items():
            df[col_name] = series
        # Log quick stats for the first column of this metric
        first_col = next(iter(cols.values()))
        desc = first_col.describe(percentiles=[0.25, 0.5, 0.75])
        _LOG.info(
            "evaluate: id=%s evaluator=%s metric=%s n=%d mean=%.4f std=%.4f min=%.4f p50=%.4f max=%.4f",
            mc["id"],
            mc["evaluator"],
            mc["metric"],
            len(first_col),
            float(desc["mean"]),
            float(desc["std"]) if pd.notna(desc["std"]) else float("nan"),
            float(desc["min"]),
            float(desc["50%"]),
            float(desc["max"]),
        )

    atomic_write_parquet(df, data)
    console.print(
        f"[green]✔[/green] Appended metrics: {', '.join(m['id'] for m in metrics)} → {data}"
    )


def _normalize_scores(scores: Any, *, n: int, metric_id: str) -> Dict[str, pd.Series]:
    """
    Accept one record per sequence. Each record may be:
      - scalar number → single column
      - dict[str, number] (flat) → one column per key, stable order by key
      - list/tuple/np.ndarray of numbers with constant length K → __0..__K-1
    Return mapping of column_name → Series[n].
    """

    def _err(msg: str) -> ValueError:
        return ValueError(f"Evaluator output for metric '{metric_id}' invalid: {msg}")

    if not isinstance(scores, (list, tuple)):
        raise _err(f"expected a list/tuple of length {n}, got {type(scores).__name__}")
    if len(scores) != n:
        raise _err(f"expected length {n}, got {len(scores)}")

    # determine record shape
    if n == 0:
        return {f"permuter__metric__{metric_id}": pd.Series([], dtype="float64")}

    first = scores[0]
    # Scalar case
    if isinstance(first, numbers.Number):
        try:
            ser = pd.Series(
                [float(x) if x is not None else float("nan") for x in scores],
                dtype="float64",
            )
        except Exception as e:
            raise _err(f"unable to coerce scalars to float: {e}") from e
        return {f"permuter__metric__{metric_id}": ser}

    # Dict-of-scalars case
    if isinstance(first, dict):
        # collect keys and validate
        keys = sorted(first.keys())
        if not keys:
            raise _err("empty dict records")
        for i, rec in enumerate(scores):
            if not isinstance(rec, dict) or set(rec.keys()) != set(keys):
                raise _err(
                    f"record {i} has keys {sorted(getattr(rec, 'keys', lambda: [])())}, expected {keys}"
                )
            for k, v in rec.items():
                if not (v is None or isinstance(v, numbers.Number)):
                    raise _err(
                        f"record {i} key '{k}' has non-numeric value {type(v).__name__}"
                    )
        out: Dict[str, pd.Series] = {}
        for k in keys:
            ser = pd.Series(
                [
                    float(rec[k]) if rec[k] is not None else float("nan")
                    for rec in scores
                ],
                dtype="float64",
            )
            out[f"permuter__metric__{metric_id}__{k}"] = ser
        return out

    # Fixed-length sequence of numbers case
    if isinstance(first, (list, tuple, pd.Series)) or hasattr(first, "__array__"):
        try:
            lens = [len(x) for x in scores]
        except Exception:
            raise _err("variable records must be sized sequences of numbers")
        if len(set(lens)) != 1:
            raise _err(f"inconsistent inner lengths {sorted(set(lens))}")
        k = lens[0]
        out: Dict[str, pd.Series] = {}
        for j in range(k):
            col = []
            for i, rec in enumerate(scores):
                v = rec[j]
                if not (v is None or isinstance(v, numbers.Number)):
                    raise _err(
                        f"record {i}[{j}] has non-numeric value {type(v).__name__}"
                    )
                col.append(float(v) if v is not None else float("nan"))
            out[f"permuter__metric__{metric_id}__{j}"] = pd.Series(col, dtype="float64")
        return out

    raise _err(f"unsupported record type: {type(first).__name__}")
