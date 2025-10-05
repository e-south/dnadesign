"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/cli/run.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

from dnadesign.permuter.src.core.config import JobConfig
from dnadesign.permuter.src.core.ids import derive_seed64, variant_id
from dnadesign.permuter.src.core.paths import resolve, resolve_job_hint
from dnadesign.permuter.src.core.registry import get_protocol
from dnadesign.permuter.src.core.storage import (
    atomic_write_parquet,
    ensure_output_dir,
    write_ref_fasta,
)
from dnadesign.permuter.src.core.usr import make_usr_row

console = Console()
_LOG = logging.getLogger("permuter.run")


def _load_job(path: str | Path) -> JobConfig:
    job_path = resolve_job_hint(Path(path))
    data = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    try:
        return JobConfig.model_validate(data)
    except Exception as e:
        raise ValueError(f"Invalid job YAML ({job_path}): {e}") from e


def _load_refs(cfg: JobConfig, base_dir: Path) -> pd.DataFrame:
    refs_path = (
        (base_dir / cfg.job.input.refs)
        if not Path(cfg.job.input.refs).is_absolute()
        else Path(cfg.job.input.refs)
    )
    if not refs_path.exists():
        raise FileNotFoundError(f"Refs CSV not found: {refs_path}")
    df = pd.read_csv(refs_path, dtype=str)
    need = {cfg.job.input.name_col, cfg.job.input.seq_col}
    if not need.issubset(df.columns):
        raise ValueError(
            f"Refs CSV must contain columns {need}, got {list(df.columns)}"
        )
    return df


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


def _variants_stream(
    protocol_name: str,
    params: Dict[str, Any],
    ref_name: str,
    sequence: str,
    *,
    seed: int,
) -> Iterable[Dict[str, Any]]:
    proto_cls = get_protocol(protocol_name)
    proto = proto_cls()
    proto.validate_cfg(params=params)
    rng = np.random.default_rng(seed)
    yield from proto.generate(
        ref_entry={"ref_name": ref_name, "sequence": sequence}, params=params, rng=rng
    )


def run(job: str | Path, ref: Optional[str], out: Optional[Path]):
    t0 = time.time()
    cfg = _load_job(job)
    # Resolve all paths in one place
    jp = resolve(
        job_yaml=Path(str(job)),
        refs=cfg.job.input.refs,
        output_dir=cfg.job.output.dir,
        ref_name="__PENDING__",  # set after picking ref
        out_override=out,
    )
    df_refs = pd.read_csv(jp.refs_csv, dtype=str)
    ref_name, ref_seq = _pick_reference(
        df_refs, cfg.job.input.name_col, cfg.job.input.seq_col, ref
    )

    # Re-resolve with actual ref_name for dataset dir
    jp = resolve(
        job_yaml=Path(str(job)),
        refs=cfg.job.input.refs,
        output_dir=cfg.job.output.dir,
        ref_name=ref_name,
        out_override=out,
    )
    ensure_output_dir(jp.dataset_dir)

    # stable RNG seed derived from knobs (so hairpin protocol is reproducible)
    seed = derive_seed64(
        job=cfg.job.name,
        ref=ref_name,
        protocol=cfg.job.permute.protocol,
        params=cfg.job.permute.params or {},
    )

    console.rule(f"[bold]Permuter run[/bold] • job={cfg.job.name} • ref={ref_name}")
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        transient=True,
    ) as prog:
        t_load = prog.add_task("Generating variants", total=None)
        rows: list[dict] = []
        for var in _variants_stream(
            cfg.job.permute.protocol,
            cfg.job.permute.params or {},
            ref_name,
            ref_seq,
            seed=seed,
        ):
            row = make_usr_row(
                sequence=var["sequence"],
                bio_type=cfg.infer_bio_type(ref_seq),
                source=f"permuter run {cfg.job.name}/{ref_name}",
            )
            # variant identity (stable across rebuilds)
            mods = list(var.get("modifications", []))
            row["permuter__var_id"] = variant_id(
                job=cfg.job.name,
                ref=ref_name,
                protocol=cfg.job.permute.protocol,
                sequence=var["sequence"],
                modifications=mods,
            )
            # standard permuter columns
            row["permuter__job"] = cfg.job.name
            row["permuter__ref"] = ref_name
            row["permuter__protocol"] = cfg.job.permute.protocol
            row["permuter__modifications"] = mods
            row["permuter__round"] = 1  # single-pass DMS

            # flatten protocol meta under permuter__*
            for k, v in var.items():
                if k in ("sequence", "modifications"):
                    continue
                row[f"permuter__{k}"] = v

            rows.append(row)
        prog.update(t_load, description=f"Generated {len(rows)} variants")
        prog.stop_task(t_load)

    if not rows:
        raise RuntimeError("Protocol produced zero variants")

    df = pd.DataFrame(rows)
    atomic_write_parquet(df, jp.records_parquet)
    write_ref_fasta(jp.dataset_dir, ref_name, ref_seq)

    # Summaries
    n = len(df)
    nt_count = (
        len(df["permuter__nt_pos"].dropna().unique())
        if "permuter__nt_pos" in df.columns
        else 0
    )
    aa_count = (
        len(df["permuter__aa_pos"].dropna().unique())
        if "permuter__aa_pos" in df.columns
        else 0
    )
    hp_lens = (
        df["permuter__hp_length_paired"].describe().to_dict()
        if "permuter__hp_length_paired" in df.columns
        else {}
    )
    _LOG.info(
        "run: wrote %d variants (unique nt_pos=%d, aa_pos=%d) → %s",
        n,
        nt_count,
        aa_count,
        jp.records_parquet,
    )
    if hp_lens:
        _LOG.info(
            "run: hairpin paired length stats: %s",
            {k: float(v) for k, v in hp_lens.items() if isinstance(v, (int, float))},
        )

    console.print(f"[green]✔[/green] Variants: {len(df)} → {jp.records_parquet}")
    console.print(f"Elapsed: {time.time()-t0:.2f}s")
    console.print(
        "[dim]Hint:[/dim] Use "
        f"[bold]permuter evaluate --job {Path(str(jp.job_yaml)).name} --ref {ref_name}[/bold] "
        "to score variants, or pass --data with the dataset directory."
    )
