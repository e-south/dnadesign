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
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml
from rich.console import Console

from dnadesign.permuter.src.core.config import JobConfig
from dnadesign.permuter.src.core.paths import (
    normalize_data_path,
    resolve,
    resolve_job_hint,
)
from dnadesign.permuter.src.core.registry import get_evaluator
from dnadesign.permuter.src.core.storage import (
    append_record_md,
    atomic_write_parquet,
    read_parquet,
    read_ref_fasta,
)

console = Console()
_LOG = logging.getLogger("permuter.evaluate")


def _load_job_metrics(job_hint: Optional[Union[str, Path]]) -> List[Dict]:
    """
    Load evaluate.metrics[] from a job YAML given either a path or a PRESET name.
    Uses resolve_job_hint() so this works from any working directory.
    """
    if not job_hint:
        return []
    job_path = resolve_job_hint(job_hint)
    data = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    cfg = JobConfig.model_validate(data)
    metrics: List[Dict] = []
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


def _pick_reference(
    df: pd.DataFrame, name_col: str, seq_col: str, desired: Optional[str]
) -> tuple[str, str]:
    if desired:
        sub = df[df[name_col] == desired]
        if sub.empty:
            raise ValueError(f"Reference '{desired}' not found in '{name_col}'")
        if len(sub) > 1:
            raise ValueError(f"Reference '{desired}' not unique in CSV")
        row = sub.iloc[0]
        return str(row[name_col]), str(row[seq_col])
    if len(df) == 1:
        row = df.iloc[0]
        return str(row[name_col]), str(row[seq_col])
    raise ValueError("--ref is required because the refs CSV has multiple rows")


def _derive_records_from_job(
    job_hint: str, ref: Optional[str], out: Optional[Path]
) -> Path:
    """
    Resolve the dataset path from a job preset/path and an optional ref.
    Tries nested, flat-job, then flat-jobref.
    """
    job_path = resolve_job_hint(job_hint)
    data = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    cfg = JobConfig.model_validate(data)
    # First resolve refs CSV relative to the job
    jp0 = resolve(
        job_yaml=job_path,
        refs=cfg.job.input.refs,
        output_dir=cfg.job.output.dir,
        ref_name="__PENDING__",
        out_override=out,
    )
    # 1) flat-job: <output_root>/records.parquet (preferred when it already exists)
    flat_job = (jp0.output_root / "records.parquet").resolve()
    if flat_job.exists():
        return flat_job

    # Need a ref name for the other layouts
    df_refs = pd.read_csv(jp0.refs_csv, dtype=str)
    desired = ref or getattr(cfg.job.input, "reference_sequence", None)
    ref_name, ref_seq = _pick_reference(
        df_refs, cfg.job.input.name_col, cfg.job.input.seq_col, desired
    )
    # 2) nested: <output_root>/<ref>/records.parquet
    jp = resolve(
        job_yaml=job_path,
        refs=cfg.job.input.refs,
        output_dir=cfg.job.output.dir,
        ref_name=ref_name,
        out_override=out,
    )
    nested = jp.records_parquet
    if nested.exists():
        return nested

    # 3) flat-jobref: <parent>/<output_root.name>__<ref>/records.parquet
    flat_jobref = (
        jp.output_root.parent / f"{jp.output_root.name}__{ref_name}" / "records.parquet"
    ).resolve()
    if flat_jobref.exists():
        return flat_jobref

    # Fall back to nested path (caller will raise a helpful error)
    return nested


def _argv() -> str:
    try:
        return shlex.join(sys.argv)
    except Exception:
        return " ".join(sys.argv)


def evaluate(
    data: Path | None,
    metric_ids: List[str] | None = None,
    with_spec: List[str] | None = None,
    job: Optional[str] = None,
    ref: Optional[str] = None,
    out: Optional[Path] = None,
):
    # Resolve records path from either --data or --job/--ref
    if data is not None:
        records = normalize_data_path(data)
    elif job:
        try:
            records = _derive_records_from_job(job, ref, out)
        except Exception as e:
            raise ValueError(
                f"Unable to derive dataset from --job. {e}\n"
                "Hint: supply --ref if your refs CSV has multiple rows."
            ) from e
    else:
        raise ValueError("Provide either --data (file or dataset dir) or --job/--ref.")

    if not records.exists():
        # Primary, actionable hint
        job_hint = f"--job {job} --ref {ref}" if job else "(your job preset)"
        raise FileNotFoundError(
            f"Dataset not found: {records}\n"
            f"Generate it first with:\n"
            f"  permuter run {job_hint}\n"
            f"Then re-run:\n"
            f"  permuter evaluate --data {records.parent}\n"
        )

    df = read_parquet(records)
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
    # De-duplicate by id (last one wins, so CLI overrides job YAML)
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

    sequences = df["sequence"].astype(str).tolist()

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
                    f"  permuter run --job {job or '<job>'} --ref {ref or '<ref>'}\n"
                ) from e
            raise

        # Normalize and write columns
        cols = _normalize_scores(scores, n=len(sequences), metric_id=mc["id"])
        for col_name, series in cols.items():
            df[col_name] = series
        # Log quick stats for the first column of this metric
        first_col = next(iter(cols.values()))
        desc = first_col.describe(percentiles=[0.25, 0.5, 0.75])
        p = mc.get("params") or {}
        red = p.get("reduction", None)
        # evaluator-specific flavor for evo2
        extra = ""
        if str(mc["evaluator"]).startswith("evo2_"):
            extra = (
                f" model={p.get('model_id','?')} device={p.get('device','?')}"
                f" prec={p.get('precision','?')} alpha={p.get('alphabet','?')}"
            )
        _LOG.info(
            "evaluate: id=%s eval=%s metric=%s%s%s n=%d mean=%.4f sd=%.4f min=%.4f p50=%.4f max=%.4f",
            mc["id"],
            mc["evaluator"],
            mc["metric"],
            (f" reduction={red}" if red else ""),
            extra,
            len(first_col),
            float(desc["mean"]),
            float(desc["std"]) if pd.notna(desc["std"]) else float("nan"),
            float(desc["min"]),
            float(desc["50%"]),
            float(desc["max"]),
        )

    atomic_write_parquet(df, records)
    console.print(
        f"[green]✔[/green] Appended metrics: {', '.join(m['id'] for m in metrics)} → {records}"
    )
    append_record_md(records.parent, "evaluate", _argv())


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
