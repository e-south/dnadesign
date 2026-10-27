"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/cli/run.py

CLI wiring for run Permuter CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
from rich.console import Console

from dnadesign.permuter.src.cli.output import emit_json
from dnadesign.permuter.src.core.config import ScopeConfig
from dnadesign.permuter.src.core.ids import derive_seed64, variant_id
from dnadesign.permuter.src.core.paths import (
    expand_param_paths,
    resolve,
    resolve_workspace_config_hint,
)
from dnadesign.permuter.src.core.registry import get_protocol
from dnadesign.permuter.src.core.storage import (
    append_record_event,
    atomic_write_parquet,
    ensure_output_dir,
    init_record_md,
    write_ref_fasta,
    write_ref_protein_fasta,
)
from dnadesign.permuter.src.core.usr import make_usr_row
from dnadesign.permuter.src.workspaces.loader import load_workspace

console = Console()
_LOG = logging.getLogger("permuter.run")


def _load_config(path: str | Path) -> tuple[ScopeConfig, Path]:
    config_path = resolve_workspace_config_hint(Path(path))
    workspace = load_workspace(config_path)
    return workspace.config, workspace.config_path


def _pick_reference(df: pd.DataFrame, name_col: str, seq_col: str, desired: Optional[str]) -> tuple[str, str]:
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
    workspace_dir: Path,
    dataset_dir: Path,
) -> Iterable[Dict[str, Any]]:
    proto_cls = get_protocol(protocol_name)
    proto = proto_cls()
    params_resolved = expand_param_paths(params or {}, workspace_dir=workspace_dir)
    params_resolved["_workspace_dir"] = str(workspace_dir)
    params_resolved["_artifact_dir"] = str(dataset_dir)
    proto.validate_cfg(params=params_resolved)
    rng = np.random.default_rng(seed)
    params_resolved["_derived_seed"] = int(seed)
    yield from proto.generate(
        ref_entry={"ref_name": ref_name, "sequence": sequence},
        params=params_resolved,
        rng=rng,
    )


def _argv() -> str:
    try:
        return shlex.join(sys.argv)
    except Exception:
        return " ".join(sys.argv)


def run(
    workspace: str | Path,
    ref: Optional[str],
    out: Optional[Path],
    overwrite: bool = False,
    as_json: bool = False,
) -> dict[str, object]:
    t0 = time.time()
    cfg, config_path = _load_config(workspace)
    # Resolve all paths in one place
    jp = resolve(
        config_yaml=config_path,
        refs=cfg.scope.input.refs,
        output_dir=cfg.scope.output.dir,
        ref_name="__PENDING__",
        out_override=out,
        layout=getattr(cfg.scope.output, "layout", None),
    )
    df_refs = pd.read_csv(jp.refs_csv, dtype=str)
    if not as_json:
        console.print(f"[cyan]Using refs CSV[/cyan]: {jp.refs_csv}")
    desired = ref or getattr(cfg.scope.input, "reference_sequence", None)
    ref_name, ref_seq = _pick_reference(df_refs, cfg.scope.input.name_col, cfg.scope.input.seq_col, desired)
    if not as_json:
        console.print(f"[dim]Using reference[/dim] [bold]{ref_name}[/bold]")

    # Re-resolve with actual ref_name for dataset dir
    jp = resolve(
        config_yaml=config_path,
        refs=cfg.scope.input.refs,
        output_dir=cfg.scope.output.dir,
        ref_name=ref_name,
        out_override=out,
        layout=getattr(cfg.scope.output, "layout", None),
    )
    ensure_output_dir(jp.dataset_dir)
    # Existence & overwrite behavior
    if jp.records_parquet.exists():
        if not overwrite:
            raise FileExistsError(
                f"Dataset already exists for ref '{ref_name}': {jp.records_parquet}\n"
                "Refuse to overwrite. Re-run with --overwrite, or choose a different --out."
            )
        if not as_json:
            console.print(f"[yellow]Overwrite enabled[/yellow] → will replace {jp.records_parquet}")
    if not as_json:
        console.print(f"[cyan]Dataset dir[/cyan]: {jp.dataset_dir}")

    # stable RNG seed derived from knobs (so hairpin protocol is reproducible)
    seed = derive_seed64(
        scope=cfg.scope.name,
        ref=ref_name,
        protocol=cfg.scope.permute.protocol,
        params=cfg.scope.permute.params or {},
    )

    if not as_json:
        console.rule(f"[bold]Permuter run[/bold] • workspace={cfg.scope.name} • ref={ref_name}")
    rows: list[dict] = []
    stream = _variants_stream(
        cfg.scope.permute.protocol,
        cfg.scope.permute.params or {},
        ref_name,
        ref_seq,
        seed=seed,
        workspace_dir=jp.workspace_dir,
        dataset_dir=jp.dataset_dir,
    )

    def _append_rows() -> None:
        for var in stream:
            row = make_usr_row(
                sequence=var["sequence"],
                bio_type=cfg.infer_bio_type(ref_seq),
                source=f"permuter run {cfg.scope.name}/{ref_name}",
            )
            mods = list(var.get("modifications", []))
            row["permuter__var_id"] = variant_id(
                scope=cfg.scope.name,
                ref=ref_name,
                protocol=cfg.scope.permute.protocol,
                sequence=var["sequence"],
                modifications=mods,
            )
            row["permuter__scope"] = cfg.scope.name
            row["permuter__ref"] = ref_name
            row["permuter__protocol"] = cfg.scope.permute.protocol
            row["permuter__modifications"] = mods
            row["permuter__round"] = 1

            for k, v in var.items():
                if k in ("sequence", "modifications"):
                    continue
                key = k if str(k).startswith("permuter__") else f"permuter__{k}"
                row[key] = v

            rows.append(row)

    if as_json:
        _append_rows()
    else:
        with console.status("[bold]Generating variants[/bold] …", spinner="dots") as st:
            _append_rows()
            st.update(status=f"[bold]Generated[/bold] {len(rows)} variants")

    if not rows:
        raise RuntimeError("Protocol produced zero variants")

    df = pd.DataFrame(rows)
    atomic_write_parquet(df, jp.records_parquet)
    write_ref_fasta(jp.dataset_dir, ref_name, ref_seq)
    # Optional authoritative protein sidecar from refs.csv if configured
    aa_col = getattr(cfg.scope.input, "aa_col", None)
    if aa_col and aa_col in df_refs.columns:
        aa_row = df_refs[df_refs[cfg.scope.input.name_col] == ref_name]
        aa_seq = str(aa_row.iloc[0][aa_col]).strip() if not aa_row.empty else ""
        if aa_seq:
            write_ref_protein_fasta(jp.dataset_dir, ref_name, aa_seq)
    # Initialize RECORD.md and log the command
    init_record_md(
        dataset_dir=jp.dataset_dir,
        config_path=config_path,
        scope_id=cfg.scope.name,
        ref_name=ref_name,
        refs_csv=jp.refs_csv,
    )
    append_record_event(
        jp.dataset_dir,
        "RUN",
        [
            f"scope: {cfg.scope.name}",
            f"workspace_config: {config_path}",
            f"refs_csv: {jp.refs_csv}",
            f"ref: {ref_name}",
            f"protocol: {cfg.scope.permute.protocol}",
            f"dataset: {jp.records_parquet}",
        ],
        command=_argv(),
    )

    # Summaries
    n = len(df)
    nt_count = len(df["permuter__nt_pos"].dropna().unique()) if "permuter__nt_pos" in df.columns else 0
    aa_count = len(df["permuter__aa_pos"].dropna().unique()) if "permuter__aa_pos" in df.columns else 0
    hp_lens = (
        df["permuter__hp_length_paired"].describe().to_dict() if "permuter__hp_length_paired" in df.columns else {}
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

    elapsed = time.time() - t0
    summary: dict[str, object] = {
        "schema": "permuter.run.v1",
        "workspace": cfg.scope.name,
        "ref": ref_name,
        "dataset_dir": jp.dataset_dir,
        "records": jp.records_parquet,
        "row_count": len(df),
        "output_layout": getattr(cfg.scope.output, "layout", None) or "default",
        "elapsed_seconds": round(elapsed, 3),
    }
    if as_json:
        emit_json(summary)
    else:
        console.print(f"[green]✔[/green] Variants: {len(df)} → {jp.records_parquet}")
        console.print(f"Elapsed: {elapsed:.2f}s")
        console.print(f"[dim]Record:[/dim] {jp.dataset_dir / 'RECORD.md'}")
    return summary
